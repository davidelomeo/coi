"""Weighted sum of OWT membership scores (Wsum) and Cyanobacteria Occurrence Index (COI)."""

from typing import Dict, List, Tuple

import numpy as np
import xarray

__all__ = ['calculate_wsum_and_coi']


def owt_weights(sowt_names: List[str], w_owt: Dict) -> np.ndarray:
    """Weights in the order of ``sowt_names``, with a clear error if any are missing."""
    missing = [name for name in sowt_names if name not in w_owt]
    if missing:
        raise KeyError(f'OWT weights are missing entries for {missing}')
    return np.array([w_owt[name] for name in sowt_names], dtype=float)


def wsum_and_coi(scores: np.ndarray, weights: np.ndarray, min_wsum: float, max_wsum: float,
                 exp: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Wsum and COI from scores of shape (..., n_owt). Points with any NaN score are NaN."""
    with np.errstate(divide='ignore', invalid='ignore'):
        valid = ~np.isnan(scores).any(axis=-1)

        # Fractionality: each score's share of the (powered) total
        powered = scores ** exp
        fractions = powered / powered.sum(axis=-1, keepdims=True)

        wsum = np.where(valid, np.nansum(scores * weights * fractions, axis=-1), np.nan)
        coi = np.where(valid, np.clip((wsum - min_wsum) / (max_wsum - min_wsum), 0, 1), np.nan)
    return wsum, coi


def calculate_wsum_and_coi(ds: xarray.Dataset, w_owt: dict, min_wsum: float, max_wsum: float,
                           exp: float = 1.0) -> xarray.Dataset:
    """Calculate Wsum and COI from the 'sowt_*' variables of a Dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with one 'sowt_<OWT>' variable per optical water type.
    w_owt : dict
        Weight of each OWT, keyed by the 'sowt_<OWT>' variable names.
    min_wsum, max_wsum : float
        Wsum values mapped to COI = 0 and COI = 1 (COI is clipped to [0, 1]).
    exp : float, optional
        Exponent used in the fractionality term, by default 1.0.

    Returns
    -------
    xarray.Dataset
        The input dataset with 'wsum' and 'coi' added.
    """
    sowt_vars = [var for var in ds.data_vars if str(var).startswith('sowt_')]
    if not sowt_vars:
        raise ValueError("Dataset has no 'sowt_*' variables")
    weights = owt_weights(sowt_vars, w_owt)

    scores = xarray.concat([ds[var] for var in sowt_vars], dim='sowt')
    wsum, coi = xarray.apply_ufunc(
        wsum_and_coi, scores,
        kwargs=dict(weights=weights, min_wsum=float(min_wsum), max_wsum=float(max_wsum), exp=float(exp)),
        input_core_dims=[['sowt']], output_core_dims=[[], []],
        dask='parallelized', output_dtypes=[float, float])

    ds['wsum'] = wsum
    ds['coi'] = coi
    return ds
