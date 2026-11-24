#!/usr/bin/env python
"""Command line interface for Cyanobacteria Occurrence Index calculation"""

import argparse
import sys
import xarray
import logging
import json
import pandas
from datetime import datetime
from pathlib import Path
from typing import Optional
from .COI import COI

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args(args: Optional[list] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Calculate Cyanobacteria Occurrence Index (COI) from satellite data'
    )
    parser.add_argument('--input', '-i', required=True,
                        help='Path to input netCDF file')
    parser.add_argument('--output', '-o', required=True,
                        help='Path to save output netCDF file')

    # Either sensor or bands must be provided
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--sensor', '-s', choices=['OLCI', 'MODIS', 'MERIS'],
                       help='Satellite sensor name')
    group.add_argument('--bands', '-b',
                       help='Comma-separated list of bands (e.g., "Rw443,Rw490,Rw560")')

    # Optional arguments
    # New arguments for custom weights and OWT files
    parser.add_argument('--weights-file',
                        help='Path to JSON file containing custom OWT weights')
    parser.add_argument('--owt-file',
                        help='Path to CSV file containing reference OWT data (with wavebands in the form "Rwxxx" as index)')
    parser.add_argument('--min-wsum', default=223.0,
                        help='Minimum weighted sum threshold (default: 223.0 - from OLCI)')
    parser.add_argument('--max-wsum', default=321.0,
                        help='Maximum weighted sum threshold (default: 321.0 - from OLCI)')
    parser.add_argument('--exp', default=1.0,
                        help='Exponent used in the computation of the proportionality factor (default: 1.0)')
    parser.add_argument('--return-all', default=False,
                        help='Flag to return all variables or only Wsum and COI (default: False)')

    return parser.parse_args(args)

def main():
    """Main function to run COI calculation from command line."""
    args = parse_args()
    
    try:
        # Load input data
        logger.info(f"Loading input file: {args.input}")
        ds = xarray.open_dataset(args.input)
        
        # Load custom weights if provided
        w_owt = None
        if args.weights_file:
            logger.info(f"Loading custom weights from: {args.weights_file}")
            with open(args.weights_file, 'r') as f:
                w_owt = json.load(f)
        
        # Load custom OWT reference if provided
        reference_owt = None
        if args.owt_file:
            logger.info(f"Loading custom OWT reference from: {args.owt_file}")
            reference_owt = pandas.read_csv(args.owt_file, index_col=0)

        # Prepare bands list if provided
        custom_bands = args.bands.split(',') if args.bands else None
        
        # Initialize COI calculator
        logger.info("Initializing COI calculator")
        coi_calc = COI(
            ds=ds,
            sensor=args.sensor,
            bands=custom_bands,
            w_owt=w_owt,
            reference_owt=reference_owt,
            min_wsum=args.min_wsum,
            max_wsum=args.max_wsum,
            exp=args.exp
        )

        # Process data and calculate indices
        logger.info("Processing input data")
        result_ds = coi_calc.process_spectral_data()
    
        # Save output
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Add processing metadata
        result_ds.attrs['processing_date'] = datetime.now().isoformat()
        # result_ds.attrs['sensor'] = args.sensor if args.sensor else 'custom'
        # result_ds.attrs['bands_used'] = ','.join(coi_calc.bands)
        
        logger.info(f"Saving...")
        encoding = {"zlib": True, "complevel": 5}
        for var in result_ds:
            result_ds[var].encoding.update(encoding)
        result_ds.to_netcdf(args.output, compute=True)
        logger.info("Processing completed successfully")
 
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()