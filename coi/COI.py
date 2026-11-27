"""Cyanobacteria Occurrence Index (COI) from satellite water reflectance."""

import json
import logging
from importlib.resources import files
from typing import ClassVar, NamedTuple

import numpy
import pandas
import xarray

from .calculate_wsum_and_coi import coi_from_own_range, owt_weights, wsum_and_coi
from .compute_spectral_angle import band_wavelengths, prepare_reference, spectral_scores

logger = logging.getLogger(__name__)

_CORE_FILES = files('coi') / 'core_files'

# Pixels are processed in blocks of this many, so memory use does not grow with the scene size
_BLOCK_SIZE = 262_144

# Number of most similar OWTs reported for each pixel (dominant_owt_1, dominant_owt_2, ...)
N_DOMINANT = 3


def load_reference_owt() -> pandas.DataFrame:
    """The reference OWT library, used by ``COI`` unless ``reference_owt`` is given.

    Returns
    -------
    pandas.DataFrame
        One row per OLCI band (index 'wl', from 'Rw412' to 'Rw779') and one column per
        optical water type ('OWT_<name>') holding its standardised mean spectrum. Each
        call returns a new table, which can be changed and passed as ``reference_owt``.
    """
    with (_CORE_FILES / 'OWT_inland_mean_standardised_extended.csv').open('r') as f:
        return pandas.read_csv(f, index_col='wl')


class _Model(NamedTuple):
    """Everything the calculation needs apart from the data, prepared once."""

    sowt_names: list[str]
    sort_idx: numpy.ndarray
    sorted_wavelengths: numpy.ndarray
    ref_standardised: numpy.ndarray  # (n_bands, n_owt)
    weights: numpy.ndarray  # (n_owt,)
    min_wsum: float
    max_wsum: float
    exp: float

    @property
    def n_dominant(self) -> int:
        return min(N_DOMINANT, len(self.sowt_names))

    @property
    def name_lookup(self) -> numpy.ndarray:
        """OWT names as in 'sowt_<name>'; the extra '' is looked up for pixels without a result."""
        return numpy.array([name[len('sowt_'):] for name in self.sowt_names] + [''])


class COI:
    """Compute the Cyanobacteria Occurrence Index (COI) from satellite data.

    Each pixel's spectrum is compared with a library of reference (standardised) optical
    water types (OWTs), built from Ocean and Land Colour Instrument (OLCI) spectra, using the
    spectral angle. The resulting OWT membership scores (Sowt) are combined into a weighted
    sum (Wsum), which is rescaled between ``min_wsum`` and ``max_wsum`` to give the COI (0-1).

    Parameters
    ----------
    ds : xarray.Dataset
        Input reflectance, one variable per band named like 'Rw443' (or 'Rw443_mean').
    sensor : {'OLCI', 'MERIS', 'MODIS'}, optional
        Sensor whose bands are used (case-insensitive). Give either ``sensor`` or ``bands``.
    bands : list of str, optional
        Custom list of band names, e.g. ['Rw443', 'Rw490', 'Rw560']. They must be rows of
        the reference OWT table.
    w_owt : dict, optional
        Weight of each OWT, keyed 'sowt_<OWT>'. Defaults to the packaged weights.
    reference_owt : pandas.DataFrame, optional
        Reference OWT spectra, band names as index and one 'OWT_<name>' column per OWT.
        Defaults to the packaged 25-OWT inland water library (``load_reference_owt()``).
    min_wsum, max_wsum : float
        Wsum values mapped to COI = 0 and COI = 1 (defaults derived from OLCI).
    exp : float
        Exponent used in the fractionality term.
    valid_mask : xarray.DataArray, optional
        Boolean mask; pixels where it is False are returned as NaN.
    return_all : bool
        If True, also return the input 'Rw*' bands and every 'sowt_*' score.
    local_coi : bool
        If True, also return 'coi_local': the COI with the smallest and largest Wsum of this
        data as the boundaries instead of ``min_wsum`` and ``max_wsum``. With it come the
        Wsum it was computed from, 'wsum_local' (the same values as 'wsum'), and the two
        boundaries, 'wsum_local_min' and 'wsum_local_max'. Pixels that are masked or have
        no result do not count. 'coi' is unaffected.

    The output always contains 'wsum', 'coi' and the names of the three OWTs with the
    highest membership scores ('dominant_owt_1' to 'dominant_owt_3').
    """

    SENSOR_BANDS: ClassVar[dict[str, list[str]]] = {
        'OLCI': [
            'Rw412', 'Rw443', 'Rw490', 'Rw510', 'Rw560', 'Rw620',
            'Rw665', 'Rw674', 'Rw681', 'Rw709', 'Rw754', 'Rw779'
        ],
        'MERIS': [
            'Rw412', 'Rw443', 'Rw490', 'Rw510', 'Rw560', 'Rw620',
            'Rw665', 'Rw681', 'Rw709', 'Rw754', 'Rw779'
        ],
        'MODIS': [
            'Rw412', 'Rw443', 'Rw488', 'Rw555', 'Rw667',
            'Rw678', 'Rw748'
        ]
    }

    # Reference (OLCI) band -> MODIS band; only MODIS bands within 6 nm of an OLCI band are used
    MODIS_BAND_MAP: ClassVar[dict[str, str]] = {
        'Rw412': 'Rw412',
        'Rw443': 'Rw443',
        'Rw490': 'Rw488',
        'Rw560': 'Rw555',
        'Rw665': 'Rw667',
        'Rw681': 'Rw678',
        'Rw754': 'Rw748'
    }

    def __init__(
        self,
        ds: xarray.Dataset,
        sensor: str | None = None,
        bands: list[str] | None = None,
        w_owt: dict | None = None,
        reference_owt: pandas.DataFrame | None = None,
        min_wsum: float = 223.0,
        max_wsum: float = 321.0,
        exp: float = 1.0,
        valid_mask: xarray.DataArray | None = None,
        return_all: bool = False,
        local_coi: bool = False) -> None:

        self.ds = ds
        self.sensor = sensor.upper() if sensor is not None else None
        self.custom_bands = bands
        self.min_wsum = float(min_wsum)
        self.max_wsum = float(max_wsum)
        self.exp = float(exp)
        self.valid_mask = valid_mask
        self.return_all = return_all
        self.local_coi = local_coi

        self.bands = self._get_sensor_bands()
        self.w_owt = self._load_w_owt() if w_owt is None else w_owt

        reference = load_reference_owt() if reference_owt is None else reference_owt
        if self.sensor == 'MODIS':
            # Label the reference rows with the matching MODIS band names
            reference = reference.rename(index=self.MODIS_BAND_MAP)
        missing = [band for band in self.bands if band not in reference.index]
        if missing:
            raise KeyError(f'Reference OWT table has no rows for bands {missing}')
        self.reference_owt = reference.loc[self.bands]

        logger.info(f"Initialised with sensor: {self.sensor or 'custom bands'}")

    @staticmethod
    def _load_w_owt() -> dict:
        """Load the default OWT weights."""
        with (_CORE_FILES / 'w_owt.json').open('r') as f:
            return json.load(f)

    def _get_sensor_bands(self) -> list[str]:
        """Bands to use, from the sensor name or the custom band list."""
        if self.sensor is not None and self.custom_bands is not None:
            raise ValueError('Provide either a sensor or a custom bands list, not both')
        if self.sensor is not None:
            if self.sensor not in self.SENSOR_BANDS:
                raise ValueError(f'Unsupported sensor. Choose from: {list(self.SENSOR_BANDS)}')
            return list(self.SENSOR_BANDS[self.sensor])
        if self.custom_bands:
            bands = list(self.custom_bands)
            band_wavelengths(bands)  # validates the band names
            return bands
        raise ValueError('Either sensor type or custom bands list must be provided')

    def _band_variables(self) -> list[str]:
        """Names of the band variables in the input dataset ('Rw443' or 'Rw443_mean')."""
        if all(band in self.ds for band in self.bands):
            return self.bands
        suffixed = [band + '_mean' for band in self.bands]
        if all(band in self.ds for band in suffixed):
            logger.info("Using band variables with a '_mean' suffix")
            return suffixed
        missing = [band for band in self.bands if band not in self.ds]
        raise KeyError(f"Input dataset is missing bands {missing} (also tried with a '_mean' suffix)")

    def _sowt_names(self) -> list[str]:
        names = []
        for col in self.reference_owt.columns:
            if 'OWT_' not in str(col):
                raise ValueError(f"Reference OWT columns must be named 'OWT_<name>'; got {col!r}")
            names.append(f"sowt_{str(col).split('OWT_')[1]}")
        return names

    def _model(self) -> _Model:
        sowt_names = self._sowt_names()
        sort_idx, sorted_wavelengths, ref_standardised = prepare_reference(self.bands, self.reference_owt)
        return _Model(sowt_names, sort_idx, sorted_wavelengths, ref_standardised,
                      owt_weights(sowt_names, self.w_owt), self.min_wsum, self.max_wsum, self.exp)

    def process_spectral_data(self) -> xarray.Dataset:
        """Compute Wsum and COI for every pixel.

        Works on in-memory and dask-backed (lazy) datasets alike.

        Returns
        -------
        xarray.Dataset
            'wsum', 'coi' and 'dominant_owt_1' to 'dominant_owt_3' (names of the OWTs with
            the highest Sowt, '' where there is no result). With ``return_all=True`` also
            the input 'Rw*' bands and the 'sowt_*' scores. With ``local_coi=True`` also
            'wsum_local', 'coi_local', 'wsum_local_min' and 'wsum_local_max'.
        """
        model = self._model()
        sowt_names, n_dominant = model.sowt_names, model.n_dominant

        inputs = [self.ds[name] for name in self._band_variables()]
        if self.valid_mask is not None:
            inputs.append(self.valid_mask)

        # Outputs: wsum, coi, one array per dominant OWT rank and, with return_all, all scores
        output_core_dims = [[]] * (2 + n_dominant) + ([['owt']] if self.return_all else [])
        output_dtypes = [float, float] + [model.name_lookup.dtype] * n_dominant
        if self.return_all:
            output_dtypes.append(float)
        logger.info('Computing spectral angle, Wsum and COI')
        outputs = xarray.apply_ufunc(
            _coi_kernel, *inputs,
            kwargs={'has_mask': self.valid_mask is not None, 'model': model,
                    'return_scores': self.return_all},
            input_core_dims=[[] for _ in inputs],
            output_core_dims=output_core_dims,
            dask='parallelized',
            keep_attrs=False,  # input attributes (e.g. units) do not apply to the outputs
            output_dtypes=output_dtypes,
            dask_gufunc_kwargs={'output_sizes': {'owt': len(sowt_names)}} if self.return_all else None)

        wsum, coi = outputs[0], outputs[1]
        wsum.attrs['long_name'] = 'Weighted sum of OWT membership scores (Wsum)'
        coi.attrs['long_name'] = 'Cyanobacteria Occurrence Index (COI)'

        variables = {}
        if self.return_all:
            for var in self.ds.data_vars:
                if str(var).startswith('Rw'):
                    band = self.ds[var].copy(deep=False)
                    band.encoding = {}
                    variables[var] = band
            for i, name in enumerate(sowt_names):
                variables[name] = outputs[-1].isel(owt=i)
        variables['wsum'] = wsum
        variables['coi'] = coi
        if self.local_coi:
            coi_local, low, high = coi_from_own_range(wsum)
            wsum_local = wsum.copy()
            wsum_local.attrs = {'long_name': 'Wsum of the local COI (same values as wsum)'}
            coi_local.attrs = {'long_name': 'Local COI: Wsum rescaled between its smallest and '
                                            'largest value in this data'}
            low.attrs = {'long_name': 'Smallest Wsum in this data (coi_local = 0)'}
            high.attrs = {'long_name': 'Largest Wsum in this data (coi_local = 1)'}
            variables |= {'wsum_local': wsum_local, 'coi_local': coi_local,
                          'wsum_local_min': low, 'wsum_local_max': high}
        for rank in range(n_dominant):
            dominant = outputs[2 + rank]
            dominant.attrs['long_name'] = f'OWT with the {_ORDINALS[rank]} highest membership score (Sowt)'
            dominant.encoding = {'dtype': 'S1'}  # compact fixed-width characters in netCDF files
            variables[f'dominant_owt_{rank + 1}'] = dominant

        return xarray.Dataset(variables, coords=self.ds.coords)


_ORDINALS = ['1st', '2nd', '3rd'] + [f'{n}th' for n in range(4, 21)]


def _process(read_block, n_pixels: int, mask: numpy.ndarray | None, model: _Model,
             return_scores: bool):
    """Wsum, COI, dominant OWTs and optionally all scores for ``n_pixels`` spectra.

    ``read_block(slice)`` returns those spectra as (n, n_bands), in the order of the bands.
    Only pixels with finite values in every band (and inside the mask) are computed; the
    rest are NaN. Pixels are processed in blocks to keep memory use bounded.

    Returns wsum and coi (n_pixels,), the columns of the dominant OWTs (n_pixels, n_dominant),
    -1 where there is no result, and the scores (n_pixels, n_owt) or None.
    """
    wsum = numpy.full(n_pixels, numpy.nan)
    coi = numpy.full(n_pixels, numpy.nan)
    dominant = numpy.full((n_pixels, model.n_dominant), -1, dtype=numpy.int16)
    scores = numpy.full((n_pixels, len(model.sowt_names)), numpy.nan) if return_scores else None

    for start in range(0, n_pixels, _BLOCK_SIZE):
        block = slice(start, min(start + _BLOCK_SIZE, n_pixels))
        spectra = numpy.asarray(read_block(block), dtype=float)
        valid = numpy.isfinite(spectra).all(axis=1)
        if mask is not None:
            valid &= mask[block]
        if not valid.any():
            continue

        index = start + numpy.flatnonzero(valid)
        block_scores = spectral_scores(
            spectra[valid], model.sort_idx, model.sorted_wavelengths, model.ref_standardised)
        wsum[index], coi[index] = wsum_and_coi(
            block_scores, model.weights, model.min_wsum, model.max_wsum, model.exp)
        dominant[index] = _top_columns(block_scores, model.n_dominant).T
        if return_scores:
            scores[index] = block_scores

    return wsum, coi, dominant, scores


def _coi_kernel(*arrays, has_mask: bool, model: _Model, return_scores: bool):
    """``_process`` for one numpy array per band, of any shape (used by xarray.apply_ufunc)."""
    arrays = numpy.broadcast_arrays(*arrays)
    if has_mask:
        *bands, mask = arrays
        flat_mask = mask.reshape(-1).astype(bool)
    else:
        bands, flat_mask = arrays, None

    shape = bands[0].shape
    flat_bands = [band.reshape(-1) for band in bands]
    wsum, coi, dominant, scores = _process(
        lambda block: numpy.stack([band[block] for band in flat_bands], axis=1),
        flat_bands[0].size, flat_mask, model, return_scores)

    outputs = (wsum.reshape(shape), coi.reshape(shape))
    outputs += tuple(model.name_lookup[codes].reshape(shape) for codes in dominant.T)
    if return_scores:
        outputs += (scores.reshape((*shape, len(model.sowt_names))),)
    return outputs


def _top_columns(scores: numpy.ndarray, n: int) -> numpy.ndarray:
    """Columns of the ``n`` highest scores in each row, shape (n, rows); -1 for rows with NaN.

    Ties go to the first column, i.e. the order of the reference library.
    """
    remaining = scores.copy()
    rows = numpy.arange(len(scores))
    top = numpy.empty((n, len(scores)), dtype=numpy.int16)
    for rank in range(n):
        top[rank] = best = numpy.argmax(remaining, axis=1)
        remaining[rows, best] = -numpy.inf
    top[:, numpy.isnan(scores).any(axis=1)] = -1
    return top
