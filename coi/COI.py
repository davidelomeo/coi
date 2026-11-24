"""Cyanobacteria Occurrence Index (COI) from satellite water reflectance."""

import json
import logging
from importlib.resources import files
from typing import Dict, List, Optional

import numpy
import pandas
import xarray

from .calculate_wsum_and_coi import owt_weights, wsum_and_coi
from .compute_spectral_angle import band_wavelengths, prepare_reference, spectral_scores

logger = logging.getLogger(__name__)

_CORE_FILES = files('coi') / 'core_files'

# Pixels are processed in blocks of this many, so memory use does not grow with the scene size
_BLOCK_SIZE = 262_144


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
        Defaults to the packaged 25-OWT inland water library.
    min_wsum, max_wsum : float
        Wsum values mapped to COI = 0 and COI = 1 (defaults derived from OLCI).
    exp : float
        Exponent used in the fractionality term.
    valid_mask : xarray.DataArray, optional
        Boolean mask; pixels where it is False are returned as NaN.
    return_all : bool
        If True, also return the input 'Rw*' bands and every 'sowt_*' score.
    """

    SENSOR_BANDS = {
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
    MODIS_BAND_MAP = {
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
        sensor: Optional[str] = None,
        bands: Optional[List[str]] = None,
        w_owt: Optional[Dict] = None,
        reference_owt: Optional[pandas.DataFrame] = None,
        min_wsum: float = 223.0,
        max_wsum: float = 321.0,
        exp: float = 1.0,
        valid_mask: Optional[xarray.DataArray] = None,
        return_all: bool = False) -> None:

        self.ds = ds
        self.sensor = sensor.upper() if sensor is not None else None
        self.custom_bands = bands
        self.min_wsum = float(min_wsum)
        self.max_wsum = float(max_wsum)
        self.exp = float(exp)
        self.valid_mask = valid_mask
        self.return_all = return_all

        self.bands = self._get_sensor_bands()
        self.w_owt = self._load_w_owt() if w_owt is None else w_owt

        reference = self._load_reference_owt() if reference_owt is None else reference_owt
        if self.sensor == 'MODIS':
            # Label the reference rows with the matching MODIS band names
            reference = reference.rename(index=self.MODIS_BAND_MAP)
        missing = [band for band in self.bands if band not in reference.index]
        if missing:
            raise KeyError(f'Reference OWT table has no rows for bands {missing}')
        self.reference_owt = reference.loc[self.bands]

        logger.info(f"Initialised with sensor: {self.sensor or 'custom bands'}")

    @staticmethod
    def _load_w_owt() -> Dict:
        """Load the default OWT weights."""
        with (_CORE_FILES / 'w_owt.json').open('r') as f:
            return json.load(f)

    @staticmethod
    def _load_reference_owt() -> pandas.DataFrame:
        """Load the default reference OWT library."""
        with (_CORE_FILES / 'OWT_inland_mean_standardised_extended.csv').open('r') as f:
            return pandas.read_csv(f, index_col='wl')

    def _get_sensor_bands(self) -> List[str]:
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

    def _band_variables(self) -> List[str]:
        """Names of the band variables in the input dataset ('Rw443' or 'Rw443_mean')."""
        if all(band in self.ds for band in self.bands):
            return self.bands
        suffixed = [band + '_mean' for band in self.bands]
        if all(band in self.ds for band in suffixed):
            logger.info("Using band variables with a '_mean' suffix")
            return suffixed
        missing = [band for band in self.bands if band not in self.ds]
        raise KeyError(f"Input dataset is missing bands {missing} (also tried with a '_mean' suffix)")

    def _sowt_names(self) -> List[str]:
        names = []
        for col in self.reference_owt.columns:
            if 'OWT_' not in str(col):
                raise ValueError(f"Reference OWT columns must be named 'OWT_<name>'; got {col!r}")
            names.append(f"sowt_{str(col).split('OWT_')[1]}")
        return names

    def process_spectral_data(self) -> xarray.Dataset:
        """Compute Wsum and COI for every pixel.

        Works on in-memory and dask-backed (lazy) datasets alike.

        Returns
        -------
        xarray.Dataset
            'wsum' and 'coi'; with ``return_all=True`` also the input 'Rw*' bands and the
            'sowt_*' scores.
        """
        sowt_names = self._sowt_names()
        sort_idx, sorted_wavelengths, ref_standardised = prepare_reference(self.bands, self.reference_owt)

        inputs = [self.ds[name] for name in self._band_variables()]
        if self.valid_mask is not None:
            inputs.append(self.valid_mask)

        n_outputs = 3 if self.return_all else 2
        logger.info('Computing spectral angle, Wsum and COI')
        outputs = xarray.apply_ufunc(
            _coi_kernel, *inputs,
            kwargs=dict(
                has_mask=self.valid_mask is not None,
                sort_idx=sort_idx,
                sorted_wavelengths=sorted_wavelengths,
                ref_standardised=ref_standardised,
                weights=owt_weights(sowt_names, self.w_owt),
                min_wsum=self.min_wsum,
                max_wsum=self.max_wsum,
                exp=self.exp,
                return_scores=self.return_all),
            input_core_dims=[[] for _ in inputs],
            output_core_dims=[[], [], ['owt']][:n_outputs],
            dask='parallelized',
            output_dtypes=[float] * n_outputs,
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
                variables[name] = outputs[2].isel(owt=i)
        variables['wsum'] = wsum
        variables['coi'] = coi

        return xarray.Dataset(variables, coords=self.ds.coords)


def _coi_kernel(*arrays, has_mask, sort_idx, sorted_wavelengths, ref_standardised, weights,
                min_wsum, max_wsum, exp, return_scores):
    """Wsum, COI (and optionally all scores) for numpy band arrays of any shape.

    Only pixels with finite values in every band (and inside the mask) are computed; the
    rest are NaN. Pixels are processed in blocks to keep memory use bounded.
    """
    arrays = numpy.broadcast_arrays(*arrays)
    if has_mask:
        *bands, mask = arrays
        flat_mask = mask.reshape(-1).astype(bool)
    else:
        bands, flat_mask = arrays, None

    shape = bands[0].shape
    flat_bands = [band.reshape(-1) for band in bands]
    n_pixels, n_owt = flat_bands[0].size, ref_standardised.shape[1]

    wsum = numpy.full(n_pixels, numpy.nan)
    coi = numpy.full(n_pixels, numpy.nan)
    scores = numpy.full((n_pixels, n_owt), numpy.nan) if return_scores else None

    for start in range(0, n_pixels, _BLOCK_SIZE):
        block = slice(start, min(start + _BLOCK_SIZE, n_pixels))
        spectra = numpy.stack([band[block] for band in flat_bands], axis=1)
        valid = numpy.isfinite(spectra).all(axis=1)
        if flat_mask is not None:
            valid &= flat_mask[block]
        if not valid.any():
            continue

        index = start + numpy.flatnonzero(valid)
        block_scores = spectral_scores(spectra[valid], sort_idx, sorted_wavelengths, ref_standardised)
        wsum[index], coi[index] = wsum_and_coi(block_scores, weights, min_wsum, max_wsum, exp)
        if return_scores:
            scores[index] = block_scores

    outputs = (wsum.reshape(shape), coi.reshape(shape))
    if return_scores:
        outputs += (scores.reshape(shape + (n_owt,)),)
    return outputs
