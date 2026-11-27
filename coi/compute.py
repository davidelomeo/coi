"""COI for spectra with wavelength along one axis or dimension (NumPy, xarray or pandas).

``COI`` reads one variable per band ('Rw443', ...). ``compute_coi`` reads the other common
layout, such as the output of a spectral convolution: one array and its wavelengths.
"""

import warnings
from collections.abc import Hashable
from typing import Any, NamedTuple

import numpy
import pandas
import xarray
from numpy.typing import ArrayLike

from .calculate_wsum_and_coi import coi_from_own_range
from .COI import COI, _process
from .compute_spectral_angle import band_wavelengths

__all__ = ['CoiArrays', 'compute_coi']

# Input with more wavelengths than this, no further apart than this many nm, is hyperspectral
_MOST_SENSOR_BANDS = 30
_WIDEST_HYPERSPECTRAL_STEP = 10.0


class CoiArrays(NamedTuple):
    """Result of ``compute_coi`` for NumPy input; the leading shape is that of the input."""

    wsum: numpy.ndarray
    coi: numpy.ndarray
    dominant_owt: numpy.ndarray  # (..., 3) OWT names, most similar first; '' without a result
    scores: numpy.ndarray | None  # (..., n_owt) Sowt, only with return_scores=True
    coi_local: numpy.ndarray | None = None  # only with local_coi=True
    wsum_local: numpy.ndarray | None = None  # only with local_coi=True; same values as wsum


def compute_coi(
    spectra: Any,
    wavelengths: ArrayLike | None = None,
    *,
    sensor: str | None = None,
    bands: list[str] | None = None,
    dim: str | None = None,
    w_owt: dict | None = None,
    reference_owt: pandas.DataFrame | None = None,
    min_wsum: float = 223.0,
    max_wsum: float = 321.0,
    exp: float = 1.0,
    valid_mask: Any = None,
    return_scores: bool = False,
    local_coi: bool = False,
    tolerance: float = 1.5,
) -> Any:
    """Compute Wsum and COI from band reflectances arranged along wavelength.

    Parameters
    ----------
    spectra : array-like, xarray.DataArray or pandas.DataFrame
        Reflectance at the sensor's bands:

        * NumPy array (or array-like): wavelength is the **last** axis.
        * xarray.DataArray: any shape, with one wavelength dimension.
        * pandas.DataFrame: rows = observations, columns = wavelengths (443 or 'Rw443').
    wavelengths : array-like, optional
        Wavelengths (nm) of the input. Required for NumPy input. For xarray and pandas input
        they are read from the coordinate or column names; give them to override.
    sensor, bands, w_owt, reference_owt, min_wsum, max_wsum, exp
        As for ``COI``. Give either ``sensor`` or ``bands``.
    dim : str, optional
        Wavelength dimension of a DataArray, if it cannot be recognised by its name.
    valid_mask : array-like of bool, optional
        Pixels where it is False are returned as NaN.
    return_scores : bool
        Also return the membership score (Sowt) of every OWT.
    local_coi : bool
        Also return 'coi_local', the COI with the smallest and largest Wsum of ``spectra``
        (after ``valid_mask``) as the boundaries, and 'wsum_local'. As for ``COI``.
    tolerance : float
        Each band uses the input wavelength closest to it, which must be within this many nm.

    Returns
    -------
    CoiArrays, xarray.Dataset or pandas.DataFrame
        * NumPy input: ``CoiArrays(wsum, coi, dominant_owt, scores, coi_local, wsum_local)``.
        * DataArray input: Dataset like that of ``COI.process_spectral_data`` ('wsum', 'coi',
          'dominant_owt_1' to '_3', plus 'sowt_*' with ``return_scores`` and 'wsum_local',
          'coi_local', 'wsum_local_min' and 'wsum_local_max' with ``local_coi``), lazy if
          dask-backed.
        * DataFrame input: the same names as columns, one row per observation.
    """
    if isinstance(spectra, xarray.Dataset):
        raise TypeError("For a Dataset with one 'Rw*' variable per band, use "
                        "COI(ds, ...).process_spectral_data(), or pass one variable with wavelengths")
    is_dataarray = isinstance(spectra, xarray.DataArray)
    if dim is not None and not is_dataarray:
        raise ValueError('dim only applies to xarray input')

    calculator = COI(
        xarray.Dataset(),
        sensor=sensor,
        bands=bands,
        w_owt=w_owt,
        reference_owt=reference_owt,
        min_wsum=min_wsum,
        max_wsum=max_wsum,
        exp=exp,
        valid_mask=valid_mask,
        return_all=return_scores,
        local_coi=local_coi)

    if is_dataarray:
        calculator.ds = select_bands(
            spectra, calculator.bands, dim=dim, wavelengths=wavelengths, tolerance=tolerance)
        result = calculator.process_spectral_data()
        return result.drop_vars(calculator.bands) if return_scores else result

    is_dataframe = isinstance(spectra, pandas.DataFrame)
    if is_dataframe:
        if wavelengths is None:
            wavelengths = _wavelengths_from_labels(spectra.columns)
        values = spectra.to_numpy(dtype=float, na_value=numpy.nan)
    elif wavelengths is None:
        raise ValueError('Pass wavelengths= (in nm) for the last axis of the array')
    elif numpy.ma.isMaskedArray(spectra):
        values = spectra.astype(float).filled(numpy.nan)
    else:
        values = numpy.asarray(spectra)

    wavelengths = numpy.asarray(wavelengths, dtype=float)
    if values.ndim == 0 or values.shape[-1:] != wavelengths.shape:
        raise ValueError(f'The last axis of the spectra (shape {values.shape}) must match the '
                         f'{wavelengths.size} wavelengths')
    columns = _match_bands(calculator.bands, wavelengths, tolerance)

    model = calculator._model()
    shape = values.shape[:-1]
    flat = values.reshape(int(numpy.prod(shape)), values.shape[-1])
    mask = None if valid_mask is None else numpy.broadcast_to(valid_mask, shape).reshape(-1).astype(bool)
    wsum, coi, dominant, scores = _process(
        lambda block: flat[block][:, columns], flat.shape[0], mask, model, return_scores)

    result = CoiArrays(
        wsum.reshape(shape),
        coi.reshape(shape),
        model.name_lookup[dominant].reshape((*shape, model.n_dominant)),
        None if scores is None else scores.reshape((*shape, len(model.sowt_names))))
    if local_coi:
        own_range_coi, _, _ = coi_from_own_range(xarray.DataArray(result.wsum))
        result = result._replace(coi_local=own_range_coi.values, wsum_local=result.wsum.copy())
    if not is_dataframe:
        return result

    columns_out = {'wsum': result.wsum, 'coi': result.coi}
    if local_coi:
        columns_out |= {'wsum_local': result.wsum_local, 'coi_local': result.coi_local}
    if return_scores:
        columns_out |= dict(zip(model.sowt_names, result.scores.T, strict=True))
    columns_out |= {f'dominant_owt_{rank + 1}': result.dominant_owt[:, rank]
                    for rank in range(model.n_dominant)}
    return pandas.DataFrame(columns_out, index=spectra.index)


def select_bands(
    data: xarray.DataArray,
    bands: list[str],
    *,
    dim: str | None = None,
    wavelengths: ArrayLike | None = None,
    tolerance: float = 1.5,
) -> xarray.Dataset:
    """The input of ``COI``, one variable per band, from a DataArray with wavelengths."""
    dim = wavelength_dim(data, dim)
    if wavelengths is None:
        if 'wavelength' in data.coords and data['wavelength'].dims == (dim,):
            wavelengths = data['wavelength'].values
        elif dim in data.coords:
            wavelengths = data[dim].values
        else:
            raise ValueError(f'Dimension {dim!r} has no wavelength coordinate; pass wavelengths=')
    if len(wavelengths) != data.sizes[dim]:
        raise ValueError(f'Dimension {dim!r} has {data.sizes[dim]} values but '
                         f'{len(wavelengths)} wavelengths were given')
    columns = _match_bands(bands, wavelengths, tolerance)
    # Coordinates along wavelength differ from band to band, so the bands could not be combined
    data = data.drop_vars([name for name, coord in data.coords.items() if dim in coord.dims])
    return xarray.Dataset({band: data.isel({dim: int(column)})
                           for band, column in zip(bands, columns, strict=True)})


def wavelength_dim(data: xarray.DataArray, dim: str | None = None) -> Hashable:
    """The wavelength dimension of a DataArray: ``dim`` if given, otherwise found by name."""
    if dim is not None:
        if dim not in data.dims:
            raise ValueError(f'No dimension {dim!r}; the dimensions are {list(data.dims)}')
        return dim
    if 'wavelength' in data.coords and data['wavelength'].ndim == 1:
        return data['wavelength'].dims[0]
    matches = [d for d in data.dims
               if 'wavelength' in str(d).lower() or str(d).lower() in ('wl', 'wvl', 'lambda')]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f'Cannot tell which of the dimensions {list(data.dims)} is wavelength; pass dim=')


def _wavelengths_from_labels(labels: pandas.Index) -> numpy.ndarray:
    """Wavelengths from column names that are numbers (443) or band names ('Rw443')."""
    try:
        return numpy.asarray(labels, dtype=float)
    except (TypeError, ValueError):
        return band_wavelengths([str(label) for label in labels])


def _match_bands(bands: list[str], wavelengths: ArrayLike, tolerance: float) -> numpy.ndarray:
    """For each band, the position of the closest wavelength, within ``tolerance`` nm."""
    wavelengths = numpy.asarray(wavelengths, dtype=float)
    if wavelengths.ndim != 1 or wavelengths.size == 0 or not numpy.isfinite(wavelengths).all():
        raise ValueError('Wavelengths must be a 1-D list of finite numbers')

    # Sensors have few bands, far apart; many closely spaced wavelengths are a spectrometer's
    spacing = (float(numpy.median(numpy.diff(numpy.sort(wavelengths))))
               if wavelengths.size > _MOST_SENSOR_BANDS else numpy.inf)
    hyperspectral = spacing <= _WIDEST_HYPERSPECTRAL_STEP
    advice = (f'The input looks hyperspectral (a wavelength every {spacing:g} nm). COI expects '
              'sensor band reflectances, so convolve it first (e.g. with satconv).')

    distance = numpy.abs(wavelengths - band_wavelengths(bands)[:, numpy.newaxis])
    columns = distance.argmin(axis=1)
    missing = [band for band, gap in zip(bands, distance.min(axis=1), strict=True) if gap > tolerance]
    if missing:
        hint = f'. {advice}' if hyperspectral else ''
        raise KeyError(f'Input is missing bands {missing}: no wavelength within {tolerance:g} nm{hint}')
    if numpy.unique(columns).size < len(bands):
        raise ValueError('Two bands match the same input wavelength; check the bands and tolerance')

    if hyperspectral:
        warnings.warn(f'{advice} Using the closest wavelength to each band.', stacklevel=4)
    return columns
