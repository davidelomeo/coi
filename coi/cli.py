"""Command line interface for the Cyanobacteria Occurrence Index (COI)."""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy
import pandas
import xarray

from . import __version__
from .COI import COI

logger = logging.getLogger('coi')


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog='coi',
        description='Calculate the Cyanobacteria Occurrence Index (COI) from satellite data')
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

    return parser.parse_args(args)


def _ascii_text_as_bytes(ds: xarray.Dataset) -> xarray.Dataset:
    """Convert in-memory ASCII text variables to bytes, ready to be written to netCDF.

    xarray converts text to bytes one value at a time, which takes seconds on large scenes.
    For ASCII text (such as OWT names) the whole array can be converted at once. The file
    is the same either way: the '_Encoding' attribute makes xarray read the values back as
    text. Anything else is left for xarray to convert.
    """
    ds = ds.copy()
    for name, var in ds.data_vars.items():
        if var.dtype.kind != 'U' or var.chunks is not None or var.ndim == 0 or var.dtype.itemsize == 0:
            continue
        code_points = numpy.ascontiguousarray(var.values).view(numpy.uint32)
        if code_points.size and code_points.max() >= 128:
            continue
        as_bytes = code_points.astype(numpy.uint8).view(f'S{var.dtype.itemsize // 4}')
        ds[name] = xarray.Variable(var.dims, as_bytes, attrs={**var.attrs, '_Encoding': 'utf-8'},
                                   encoding=var.encoding)
    return ds


def main(argv: Optional[List[str]] = None) -> None:
    """Run the COI calculation from the command line."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        logger.info(f'Loading input file: {args.input}')
        ds = xarray.open_dataset(args.input)

        w_owt = None
        if args.weights_file:
            logger.info(f'Loading custom weights from: {args.weights_file}')
            with open(args.weights_file, 'r') as f:
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
            return_all=args.return_all)

        logger.info('Processing input data')
        result_ds = coi_calc.process_spectral_data()
        result_ds.attrs['processing_date'] = datetime.now().isoformat()

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f'Saving to {output_path}')
        result_ds = _ascii_text_as_bytes(result_ds)
        # Keep each variable's own encoding (e.g. how text is stored) and add compression
        encoding = {var: {**result_ds[var].encoding, 'zlib': True, 'complevel': 5}
                    for var in result_ds.data_vars}
        result_ds.to_netcdf(output_path, encoding=encoding)
        logger.info('Processing completed successfully')

    except Exception as e:
        logger.error(f'Error during processing: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
