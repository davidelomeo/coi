"""Module for computing spectral angles between input and reference spectra."""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def compute_spectral_angle(
    spectra: np.ndarray, 
    bands: list, 
    reference_owt: pd.DataFrame
) -> np.ndarray:
    """
    Compute spectral angles between input spectra and reference spectra.
    
    Parameters
    ----------
    spectra : np.ndarray
        Input spectra array (n_points, n_bands)
    bands : list
        List of band names
    reference_owt : pd.DataFrame
        Reference optical water types data
        
    Returns
    -------
    np.ndarray
        Spectral angle scores (n_points, n_owt)
    """
    try:
        # Sort wavelengths for integration
        wavelengths = np.array([float(band[2:].replace('_mean', '')) for band in bands])
        sort_idx = np.argsort(wavelengths)
        sorted_wavelengths = wavelengths[sort_idx]
        
        # Calculate areas for input spectra
        areas = np.trapz(spectra[:, sort_idx], x=sorted_wavelengths, axis=1)
        standardised = spectra / areas[:, np.newaxis]
    
        # Prepare reference spectra
        ref_values = reference_owt.loc[bands].values
        ref_areas = np.trapz(ref_values[sort_idx], x=sorted_wavelengths, axis=0)
        ref_standardised = ref_values / ref_areas
        
        # Compute spectral angles
        sumXY = standardised @ ref_standardised
        sumXX = np.sum(standardised * standardised, axis=1, keepdims=True)
        sumYY = np.sum(ref_standardised * ref_standardised, axis=0)

        # Calculate scores
        with np.errstate(invalid='ignore'):
            angles = np.arccos(
                sumXY / (np.sqrt(sumXX) * np.sqrt(sumYY[np.newaxis, :]))
            )
            scores = 1 - (angles / np.pi)
        
        return scores

    except Exception as e:
        logger.error(f"Error in spectral angle computation: {str(e)}")
        raise