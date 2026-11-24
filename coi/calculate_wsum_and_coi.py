"""Module for calculating Weighted Sum and Cyanobacteria Occurrence Index."""

import numpy
import xarray
import logging

logger = logging.getLogger(__name__)

def calculate_wsum_and_coi(
    ds: xarray.Dataset,
    w_owt: dict,
    min_wsum: float,
    max_wsum: float,
    exp: float = 1.0
) -> xarray.Dataset:
    """
    Calculate Weighted Sum and Cyano Occurrence Index (COI) for an xarray Dataset.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing spectral data
    w_owt : dict
        Dictionary containing weights for optical water types
    min_wsum : float
        Minimum weight sum threshold
    max_wsum : float
        Maximum weight sum threshold
    exp : float, optional
        Exponent used in computation, by default 1.0
        
    Returns
    -------
    xarray.Dataset
        Input dataset with added 'wsum' and 'coi' variables
    """
    try:
        # Get sowt variables
        sowt_vars = [var for var in ds.data_vars if var.startswith('sowt_')]
    
        # Create combined DataArray
        combined = xarray.concat([ds[var] for var in sowt_vars], dim='sowt')
        combined = combined.assign_coords(sowt=sowt_vars)
    
        # Create mask for valid data
        valid_mask = ~numpy.isnan(combined).any(dim='sowt')
        
        # Calculate fractionality
        powered = combined ** exp
        total_powered = powered.sum(dim='sowt')
        fractions = xarray.where(valid_mask, powered / total_powered, numpy.nan)
        
        # Prepare weights
        weights = xarray.DataArray(
            [w_owt[var] for var in sowt_vars],
            dims=['sowt'],
            coords={'sowt': sowt_vars}
        )
        
        # Calculate weighted sum and COI
        weighted_sum = xarray.where(
            valid_mask,
            (combined * weights * fractions).sum(dim='sowt'),
            numpy.nan
        )

        ds['wsum'] = weighted_sum
        ds['coi'] = (ds['wsum'] - min_wsum) / (max_wsum - min_wsum)
        
        # Clip COI values to valid range
        ds['coi'] = xarray.where(valid_mask, numpy.clip(ds['coi'], 0, 1), numpy.nan)

        return ds

    except Exception as e:
        logger.error(f"Error in WSUM and COI calculation: {str(e)}")
        raise