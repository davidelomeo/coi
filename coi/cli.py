"""Command line interface for the Cyanobacteria Occurrence Index (COI)."""

import argparse
import json
import logging
import os
import re
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy
import pandas
import xarray

from . import __version__
from .COI import COI
from .compute import select_bands, wavelength_dim

logger = logging.getLogger('coi')

# Names of variables that hold one band each
_BAND_VARIABLE = re.compile(r'Rw\d+(\.\d+)?(_mean)?')

_EPILOG = """\
The input holds either one variable per band (Rw412, Rw443, ...) or one variable with a
wavelength dimension, such as the output of satconv:

  satconv -i spectra.nc -o olci.nc -s OLCI_S3A -v rrs
  coi -i olci.nc -o coi.nc -s OLCI --valid-where rrs_QC=1 --keep rrs_QC latitude longitude
"""


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog='coi',
        description='Calculate the Cyanobacteria Occurrence Index (COI) from satellite data',
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument('--input', '-i', required=True,
                        help='Path to input netCDF file')
    parser.add_argument('--output', '-o', required=True,
                        help='Path to save output netCDF file')

    # Either sensor or bands must be provided
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--sensor', '-s', type=str.upper, choices=list(COI.SENSOR_BANDS),
                       help='Satellite sensor name (OLCI, MERIS or MODIS)')
    group.add_argument('--bands', '-b',
                       help='Comma-separated list of bands (e.g., "Rw443,Rw490,Rw560")')

    parser.add_argument('--variable', '-v',
                        help="Variable with a wavelength dimension to read the bands from; found "
                             "automatically if the input has no band variables ('Rw443', ...)")
    parser.add_argument('--dim',
                        help='Name of its wavelength dimension, if it cannot be recognised')
    parser.add_argument('--keep', nargs='+', default=[], metavar='NAME',
                        help='Input variables to copy into the output, e.g. rrs_QC latitude')
    parser.add_argument('--valid-where', action='append', type=_condition, default=[],
                        metavar='NAME=VALUE',
                        help='Only compute pixels where this input variable has this value, e.g. '
                             'rrs_QC=1 (several values: rrs_QC=1,2). Other pixels are NaN. '
                             'Repeat the option to require several conditions')
    parser.add_argument('--weights-file',
                        help='Path to JSON file containing custom OWT weights')
    parser.add_argument('--owt-file',
                        help='Path to CSV file containing reference OWT data '
                             '(with wavebands in the form "Rwxxx" as index)')
    parser.add_argument('--min-wsum', type=float, default=223.0,
                        help='Minimum weighted sum threshold (default: 223.0 - from OLCI)')
    parser.add_argument('--max-wsum', type=float, default=321.0,
                        help='Maximum weighted sum threshold (default: 321.0 - from OLCI)')
    parser.add_argument('--exp', type=float, default=1.0,
                        help='Exponent used in the computation of the proportionality factor (default: 1.0)')
    parser.add_argument('--return-all', action='store_true',
                        help='Also write the input bands and all OWT scores, not only Wsum and COI')
    parser.add_argument('--local-coi', action='store_true',
                        help='Also write coi_local, the COI between the smallest and largest Wsum of '
                             'the input file (coi still uses --min-wsum and --max-wsum)')

    return parser.parse_args(args)


def _condition(text: str) -> tuple[str, list[float]]:
    """A --valid-where condition, 'NAME=VALUE[,VALUE...]', as the name and the values."""
    name, _, values = text.partition('=')
    try:
        numbers = [float(value) for value in values.split(',')]
    except ValueError:
        numbers = []
    if not name or not numbers:
        raise argparse.ArgumentTypeError(f'expected NAME=VALUE, such as rrs_QC=1, not {text!r}')
    return name, numbers


def _valid_mask(ds: xarray.Dataset, conditions: list[tuple[str, list[float]]],
                dims: tuple) -> xarray.DataArray:
    """True for the pixels that meet every --valid-where condition.

    ``dims`` are the dimensions of the pixels; a condition cannot use any other.
    """
    mask = xarray.DataArray(True)
    for name, values in conditions:
        if name not in ds.variables:
            raise KeyError(f'--valid-where: no variable {name} in the input')
        data = ds[name]
        if data.dtype.kind not in 'biuf':
            raise ValueError(f'--valid-where: {name} is not a number variable')
        extra = [str(dim) for dim in data.dims if dim not in dims]
        if extra:
            raise ValueError(f'--valid-where: {name} has dimension {", ".join(extra)}, which the '
                             'pixels do not have')
        mask = mask & data.isin(values)
    return mask


def _ascii_text_as_bytes(ds: xarray.Dataset) -> xarray.Dataset:
    """Convert in-memory ASCII text stored as characters ('S1') to bytes, ready to be written.

    xarray converts text to bytes one value at a time, which takes seconds on large scenes.
    For ASCII text (such as OWT names) the whole array can be converted at once. The file
    is the same either way: the '_Encoding' attribute makes xarray read the values back as
    text. Anything else is left for xarray to convert.
    """
    ds = ds.copy()
    for name, var in ds.data_vars.items():
        if (var.encoding.get('dtype') != 'S1' or var.dtype.kind != 'U' or var.chunks is not None
                or var.ndim == 0 or var.dtype.itemsize == 0):
            continue
        code_points = numpy.ascontiguousarray(var.values).view(numpy.uint32)
        if code_points.size and code_points.max() >= 128:
            continue
        as_bytes = code_points.astype(numpy.uint8).view(f'S{var.dtype.itemsize // 4}')
        ds[name] = xarray.Variable(var.dims, as_bytes, attrs={**var.attrs, '_Encoding': 'utf-8'},
                                   encoding=var.encoding)
    return ds


def _wavelength_variable(ds: xarray.Dataset, dim: str | None) -> str | None:
    """The variable to read bands from, or None if the input has one variable per band."""
    if dim is None and any(_BAND_VARIABLE.fullmatch(str(name)) for name in ds.data_vars):
        return None
    found = []
    for name, data in ds.data_vars.items():
        try:
            wavelength_dim(data, dim)
        except ValueError:
            continue
        found.append(name)
    if len(found) > 1:
        raise ValueError(f'Several variables have wavelengths; choose with --variable: {", ".join(found)}')
    if dim is not None and not found:
        raise ValueError(f'No variable has a dimension {dim!r}')
    return found[0] if found else None


def _with_stored_variables(result: xarray.Dataset, path: str, ds: xarray.Dataset,
                           keep: list[str]) -> xarray.Dataset:
    """The result, with everything that comes from the input written exactly as stored there.

    That covers the coordinates carried over from the input and the variables named in
    ``--keep`` with their coordinates. They are read a second time, without decoding.
    """
    carried = [name for name in result.coords if name in ds.variables]
    added = [name for kept in keep if kept not in result.variables
             for name in (kept, *ds[kept].coords) if name not in result.variables]
    with xarray.open_dataset(path, decode_cf=False) as stored:
        for name in dict.fromkeys([*carried, *added]):
            variable = stored.variables[name]
            if not _can_copy(variable):
                raise ValueError(f'{name} has a compound or variable-length type, '
                                 'which cannot be copied')
            copied = _as_stored(variable).load()
            if name in result.coords:
                result = result.assign_coords({name: copied})
            else:
                result[name] = copied
    return result


def _can_copy(variable: xarray.Variable) -> bool:
    """Whether a variable's type can be written again (not compound or variable-length)."""
    if variable.dtype.names:
        return False
    # A variable-length type only shows once a value is read
    first = variable[tuple(slice(0, 1) for _ in variable.dims)].values
    return not (first.dtype == object and first.size and isinstance(first.flat[0], numpy.ndarray))


def _as_stored(variable: xarray.Variable) -> xarray.Variable:
    """An undecoded input variable, set up so that writing it reproduces what is in the file."""
    dims, data = variable.dims, variable.data
    encoding = {key: value for key, value in variable.encoding.items() if key == 'dtype'}
    if variable.dtype == 'S1' and dims and variable.shape[-1]:
        # Join the characters into strings here: otherwise writing them adds a dimension
        characters = numpy.ascontiguousarray(variable.values)
        data = characters.view(f'S{characters.shape[-1]}').reshape(characters.shape[:-1])
        encoding, dims = {'char_dim_name': dims[-1]}, dims[:-1]
    if '_FillValue' not in variable.attrs:
        encoding['_FillValue'] = None  # otherwise xarray adds one to floating point variables
    return xarray.Variable(dims, data, variable.attrs, encoding)


@contextmanager
def _replacing(path: Path) -> Iterator[Path]:
    """A scratch path that takes the place of ``path`` only if the block finishes without error.

    A write that fails therefore leaves an existing file as it was, and no partial file.
    """
    target = path.resolve()  # writing to a link updates the file it points to
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.coi-', dir=target.parent) as folder:
        scratch = Path(folder) / target.name
        yield scratch
        scratch.replace(target)


def main(argv: list[str] | None = None) -> None:
    """Run the COI calculation from the command line."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        _run(args)
    except (OSError, KeyError, ValueError, RuntimeError) as e:
        logger.error(f'Error during processing: {e}')
        sys.exit(1)


def _run(args: argparse.Namespace) -> None:
    # The shell does not expand '~' in --input=~/file
    input_path = os.path.expanduser(args.input)  # kept as text: it can also be a URL
    output_path = Path(args.output).expanduser()
    logger.info(f'Loading input file: {input_path}')
    with xarray.open_dataset(input_path) as ds:
        input_is_file = os.path.exists(input_path)
        if input_is_file and output_path.exists() and output_path.samefile(input_path):
            raise ValueError('The output file must not be the input file')
        if 'wavelength' in ds.data_vars and ds['wavelength'].ndim == 1:
            ds = ds.set_coords('wavelength')
        unknown = [name for name in args.keep if name not in ds.variables]
        if unknown:
            raise KeyError(f'--keep: no variable {", ".join(unknown)} in the input')

        w_owt = None
        if args.weights_file:
            logger.info(f'Loading custom weights from: {args.weights_file}')
            with open(args.weights_file) as f:
                w_owt = json.load(f)

        reference_owt = None
        if args.owt_file:
            logger.info(f'Loading custom OWT reference from: {args.owt_file}')
            reference_owt = pandas.read_csv(args.owt_file, index_col=0)

        custom_bands = [band.strip() for band in args.bands.split(',')] if args.bands else None

        coi_calc = COI(
            ds=ds,
            sensor=args.sensor,
            bands=custom_bands,
            w_owt=w_owt,
            reference_owt=reference_owt,
            min_wsum=args.min_wsum,
            max_wsum=args.max_wsum,
            exp=args.exp,
            return_all=args.return_all,
            local_coi=args.local_coi)

        variable = args.variable or _wavelength_variable(ds, args.dim)
        if variable is not None:
            if variable not in ds.data_vars:
                raise KeyError(f'No variable {variable} in the input')
            logger.info(f'Reading bands from variable: {variable}')
            coi_calc.ds = select_bands(ds[variable], coi_calc.bands, dim=args.dim)

        if args.valid_where:
            pixel_dims = coi_calc.ds[coi_calc._band_variables()[0]].dims
            coi_calc.valid_mask = _valid_mask(ds, args.valid_where, pixel_dims)
            logger.info(f'--valid-where: {int(coi_calc.valid_mask.sum())} of '
                        f'{coi_calc.valid_mask.size} values meet the conditions')

        logger.info('Processing input data')
        result_ds = coi_calc.process_spectral_data()
        result_ds = _with_stored_variables(result_ds, input_path, ds, args.keep)

        now = datetime.now(UTC)
        entry = f'{now:%Y-%m-%dT%H:%M:%SZ} coi {__version__}'
        history = ds.attrs.get('history')
        result_ds.attrs['processing_date'] = now.isoformat()
        result_ds.attrs['history'] = f'{history}\n{entry}' if history else entry

        logger.info(f'Saving to {output_path}')
        result_ds = _ascii_text_as_bytes(result_ds)
        # Keep each variable's own encoding (e.g. how text is stored) and add compression,
        # which netCDF does not allow for variable-length text
        encoding = {name: {**data.encoding, 'zlib': True, 'complevel': 5}
                    for name, data in result_ds.data_vars.items()
                    if data.dtype.kind not in 'UO' or data.encoding.get('dtype') == 'S1'}
        with _replacing(output_path) as scratch:
            result_ds.to_netcdf(scratch, encoding=encoding)
        logger.info('Processing completed successfully')


if __name__ == '__main__':
    main()
