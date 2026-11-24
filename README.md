# Calculate the Cyanobacteria Occurrence Index

Program to compute the Weighted Optical Water Type (OWT) membership score (Sowt) sum (Wsum) and the Cyanobacteria Occurrence Index
The extended of 25 OWT library for inland waters, and related weights in Lomeo __et al.__ (in press) are used as the reference (default) OWT and OWT weights
The program currently only supports MERIS, MODIS and OLCI.

## Usage Examples

### Command Line Interface (CLI)
```bash
# Basic usage with OLCI sensor (default settings)
coi -i input.nc -o output.nc -s OLCI

# Use MODIS with different wsum range
coi -i input.nc -o output.nc -s MODIS --min-wsum 200 --max-wsum 300

# Use custom bands
coi -i input.nc -o output.nc -b Rw443,Rw490,Rw560,Rw665,Rw681

# Use custom weights and reference OWT files
coi -i input.nc -o output.nc -s OLCI \
    --weights-file path/to/weights.json \
    --owt-file path/to/reference_owt.csv

# Return all variables instead of just wsum and coi
coi -i input.nc -o output.nc -s MERIS --return-all
```

### Python Package Usage
```python
import xarray
import pandas
from coi import COI

# Load your data
ds = xr.open_dataset('input.nc')

# Basic usage with default settings
coi_calc = COI(ds, sensor='OLCI')
results = coi_calc.calculate_wsum_and_coi()

# Using custom weights and OWT reference
custom_weights = {
    "sowt_1": 1000,
    "sowt_2": 100,
    "sowt_3": 1,
    # ... add all required weights (i.e., 25 matching the default OWT library if using it)
}

custom_owt = pd.read_csv('reference_owt.csv', index_col=0) # Ensure the index_col is the wavebands colum

coi_calc = COI(
    ds,
    sensor='OLCI',
    w_owt=custom_weights,
    reference_owt=custom_owt,
    min_wsum=200,
    max_wsum=345
)
results = coi_calc.calculate_wsum_and_coi()

# Save results
results.to_netcdf('output.nc')
```