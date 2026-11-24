"""Spectral angle between input spectra and the reference Optical Water Types (OWTs)."""

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = ['compute_spectral_angle']

# np.trapz was removed in NumPy 2.x in favour of np.trapezoid
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz


def band_wavelengths(bands: Sequence[str]) -> np.ndarray:
    """Wavelengths (nm) from band names such as 'Rw443' or 'Rw443_mean'."""
    try:
        return np.array([float(band[2:].replace('_mean', '')) for band in bands])
    except ValueError:
        raise ValueError(
            f"Band names must look like 'Rw443' (optionally with '_mean'); got {list(bands)}") from None


def prepare_reference(bands: List[str], reference_owt: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardise the reference OWT spectra once, for reuse on every block of pixels.

    Returns the band sort order, the sorted wavelengths and the area-standardised
    reference spectra (n_bands, n_owt).
    """
    missing = [band for band in bands if band not in reference_owt.index]
    if missing:
        raise KeyError(f'Reference OWT table has no rows for bands {missing}')

    wavelengths = band_wavelengths(bands)
    sort_idx = np.argsort(wavelengths)
    sorted_wavelengths = wavelengths[sort_idx]

    ref_values = reference_owt.loc[bands].values.astype(float)
    ref_areas = _trapezoid(ref_values[sort_idx], x=sorted_wavelengths, axis=0)
    return sort_idx, sorted_wavelengths, ref_values / ref_areas


def spectral_scores(spectra: np.ndarray, sort_idx: np.ndarray, sorted_wavelengths: np.ndarray,
                    ref_standardised: np.ndarray) -> np.ndarray:
    """OWT membership scores (1 - angle / pi) for spectra of shape (n_points, n_bands)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        areas = _trapezoid(spectra[:, sort_idx], x=sorted_wavelengths, axis=1)
        standardised = spectra / areas[:, np.newaxis]

        sum_xy = standardised @ ref_standardised
        sum_xx = np.sum(standardised * standardised, axis=1, keepdims=True)
        sum_yy = np.sum(ref_standardised * ref_standardised, axis=0)

        cosine = sum_xy / (np.sqrt(sum_xx) * np.sqrt(sum_yy)[np.newaxis, :])
        # Rounding can push the cosine of identical spectra just above 1
        angles = np.arccos(np.clip(cosine, -1.0, 1.0))
    return 1 - angles / np.pi


def compute_spectral_angle(spectra: np.ndarray, bands: List[str], reference_owt: pd.DataFrame) -> np.ndarray:
    """Compute spectral angle scores between input spectra and reference OWT spectra.

    Parameters
    ----------
    spectra : np.ndarray
        Input spectra, shape (n_points, n_bands), bands in the same order as ``bands``.
    bands : list of str
        Band names, e.g. ['Rw443', 'Rw490', ...]; they must be rows of ``reference_owt``.
    reference_owt : pd.DataFrame
        Reference OWT spectra, bands as index and one column per OWT.

    Returns
    -------
    np.ndarray
        Scores (1 - angle / pi), shape (n_points, n_owt).
    """
    return spectral_scores(np.asarray(spectra), *prepare_reference(bands, reference_owt))
