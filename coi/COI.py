"""Module to calculate Cyanobacteria Occurrence Index from hyperspectral data."""

from typing import List, Dict, Optional
from pkg_resources import resource_stream
import logging
import json
import numpy
import pandas
import xarray
from .compute_spectral_angle import compute_spectral_angle
from .calculate_wsum_and_coi import calculate_wsum_and_coi

logger = logging.getLogger(__name__)

class COI:
    """Class to compute the Cyanobacteria Occurrence Index from satellite data.
    
    This class implements the calculation of Cyanobacteria Occurrence Index (COI)
    using satellite spectra and a reference (standardised) optical water types (OWT)
    based on Ocean Land Colour Instrument (OLCI) spectra.
    """

    # Define sensor bands as class attributes
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

    # Define band mapping for MODIS - only bands within 6nm distance from OLCI wavebands are used
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
        """Initialise the COI calculator"""
    
        self.ds = ds
        self.sensor = sensor
        self.custom_bands = bands
        self.min_wsum = min_wsum
        self.max_wsum = max_wsum
        self.exp = exp
        self.valid_mask = valid_mask
        self.return_all = return_all
        
        # Set bands based on input parameters
        self.bands = self._get_sensor_bands()
        
        # Load default files if not provided
        if w_owt is None:
            self.w_owt = self._load_w_owt()
        else:
            self.w_owt = w_owt

        if reference_owt is None:
            self.reference_owt = self._load_reference_owt()
        else:
            self.reference_owt = reference_owt

        # Dropping bands from the reference OWT that are not available in the input dataset
        if self.sensor == 'MERIS':
            self.reference_owt = self.reference_owt.drop('Rw674')
        elif self.sensor == 'MODIS':
            self.reference_owt = self.reference_owt.loc[['Rw412', 'Rw443', 'Rw490', 'Rw560', 'Rw665', 'Rw681', 'Rw754']]
            self.reference_owt.index = self.reference_owt.index.map(self.MODIS_BAND_MAP)

        logger.info(f"Initialised with sensor: {sensor if sensor else 'custom'}")

    def _load_w_owt(self) -> Dict:
        """Load the OWT weights from the default JSON file"""
        with resource_stream('coi', 'core_files/w_owt.json') as f:
            return json.load(f)

    def _load_reference_owt(self) -> pandas.DataFrame:
        """Load the reference OWT data from the default CSV file"""
        with resource_stream('coi', 'core_files/OWT_inland_mean_standardised_extended.csv') as f:
            return pandas.read_csv(f, index_col='wl')

    def _get_sensor_bands(self) -> List[str]:
        """Get the appropriate bands based on sensor type or custom input"""  
        if self.sensor is not None:
            if self.sensor.upper() not in self.SENSOR_BANDS:
                raise ValueError(
                    f"Unsupported sensor. Choose from: {list(self.SENSOR_BANDS.keys())}")
            return self.SENSOR_BANDS[self.sensor.upper()]
    
        raise ValueError("Either sensor type or custom bands list must be provided")


    def process_spectral_data(self) -> xarray.Dataset:
        """Process spectral data to compute Sowt scores

        Returns
        -------
        xarray.Dataset
            Dataset containing wavebands and raw sowt scores
        """
        try:
            try:
                spectra_list = [self.ds[band] for band in self.bands]
            except:
                logger.info("Default band name not in dataset. Trying adding '_mean'... to waveband names")
                spectra_list = [self.ds[band+'_mean'] for band in self.bands]

            # Stack all bands into a single array
            stacked_spectra = numpy.stack([arr.values for arr in spectra_list], axis=-1)

            # Apply mask if provided
            if self.valid_mask is not None:
                stacked_spectra[~ self.valid_mask.values] = numpy.nan

            # Reshape for processing
            original_shape = stacked_spectra.shape[:-1]
            n_points = numpy.prod(original_shape)
            spectra_2d = stacked_spectra.reshape(n_points, -1)

            # Compute spectral angles
            logger.info("Computing spectral angle")
            scores_array = compute_spectral_angle(spectra_2d, self.bands, self.reference_owt)
            raw_scores = scores_array.reshape(original_shape + (len(self.reference_owt.columns),))
    
            # Create raw dataset with selected variables
            raw_variables = {}

            # Add all wavebands (variables starting with 'Rw')
            for var in self.ds.data_vars:
                if var.startswith('Rw'):
                    raw_variables[var] = (self.ds[var].dims, self.ds[var].values)   

            # Add raw sowt scores
            for i, col in enumerate(self.reference_owt.columns):
                raw_variables[f"sowt_{col.split('OWT_')[1]}"] = (
                    spectra_list[0].dims,
                    raw_scores[..., i]
                )

            # Create raw dataset
            new_ds = xarray.Dataset(raw_variables, coords=self.ds.coords)
            
            # Compute Wsum and COI
            new_ds = calculate_wsum_and_coi(new_ds, self.w_owt, self.min_wsum, self.max_wsum, self.exp)

            if not self.return_all:
                return new_ds[['wsum', 'coi']]
            else:
                return new_ds

        except Exception as e:
            logger.error(f"Error processing input dataset: {str(e)}")
            raise
