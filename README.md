# FTMY: Future Typical Meteorological Year

## Overview

FTMY (Future Typical Meteorological Year) is a scientific pipeline for generating hourly weather files for future climate conditions based on climate projections and historical observations. The method selects representative months from future climate projections using FS (Finkelstein–Schafer) selection, matches projected days to historical observations, and assembles hourly time series from nearby meteorological stations. The output is compatible with building energy simulation formats (EPW files).

This implementation corresponds to the methodology described in the associated academic paper.

## Method Summary

The FTMY pipeline operates in six sequential steps:

1. **Projection Extraction**: Extracts daily climate variables (temperature, solar radiation, wind speed) from NetCDF projection files for a target location using nearest grid point selection.

2. **FS Selection**: Selects representative months from the projection period using temperature-based FS (Finkelstein–Schafer) selection. For each calendar month, the method identifies the year within the projection period that minimizes the FS distance against the long-term distribution.

3. **Station Selection**: Identifies meteorological stations within 10 km of the target location using historical station metadata.

4. **Historical Matching**: Matches each projected day to the most similar historical day from the same calendar date across available years. Matching uses a normalized distance formula based on daily mean temperature, solar radiation, and wind speed.

5. **Hourly Assembly and Gap Handling**: Constructs hourly time series by extracting data from the matched historical days. For each variable, the method selects the nearest available station within 10 km. Applies linear interpolation only for single-day gaps (24 consecutive missing hours) when both neighboring days are complete.

6. **EPW Generation**: Converts the final hourly dataset to EnergyPlus Weather (EPW) format with appropriate metadata.

## Repository Structure

```
FTMY/
├── Src/
│   ├── pipeline/
│   │   ├── config.py              # Configuration and path management
│   │   └── interface_cli.py       # Main pipeline orchestrator
│   ├── projection/
│   │   ├── projection_extractor.py
│   │   ├── fs_selector.py
│   │   └── projection_resolver.py
│   ├── matching/
│   │   ├── station_selector.py
│   │   └── daily_matcher.py
│   ├── assembly/
│   │   ├── hourly_assembler.py
│   │   └── interpolation.py
│   ├── visualization/
│   │   ├── climate_yearly_profiles.py
│   │   ├── station_contribution_plots.py
│   │   └── temperature_comparison.py
│   └── epw/
│       └── epw_writer.py
├── Output/results/                # Full mode outputs
├── Test_Output/results/           # Test mode outputs
└── find_netcdf_files.ps1          # Helper script for locating NetCDF files
```

## Data Requirements

The pipeline requires three types of external data, which must be downloaded manually and placed in directories specified by environment variables. These datasets are large and are not included in this repository.

### 1. Historical Daily Climate Data

**Source**: Danish Meteorological Institute (DMI) bulk download service  
**URL**: https://dmigw.govcloud.dk/v2/metObs/bulk/

**Format**: Daily text files organized by year:
```
<historical_base_dir>/
  YYYY/
    YYYY-MM-DD.txt
```

Each daily file contains 10 min raw observations from multiple stations. The pipeline reads these files to extract hourly data for matched historical days.

**Required variables**: Temperature, solar radiation, wind speed, precipitation, and other meteorological variables as available.

### 2. Climate Projection NetCDF Files

**Source**: DMI Climate Atlas (bias-corrected CORDEX data)  
**URL**: https://download.dmi.dk/Research_Projects/klimaatlas/v2024b/daily_bias_corrected/

**Format**: NetCDF files following CORDEX naming conventions. Files should be placed in a single directory accessible via `FTMY_PREDICTION_PATH`.

**File naming**: Files should follow CORDEX conventions, e.g.:
```
tas_KGDK-1_ICHEC-EC-EARTH_rcp85_r3i1p1_KNMI-RACMO22E_v1_day_20110101-20401231.nc
```

### 3. Daily Summary Pickle Files

**Format**: Pre-computed daily summary files (`.pkl` format) containing aggregated statistics for each calendar day across historical years.

**Structure**:
```
<daily_summaries_dir>/
  daily_summary_MM-DD.pkl
```

These files contain station metadata and daily statistics used for station selection and historical matching. They must be generated from the historical daily climate data prior to running the pipeline. These files are pre-generated daily summary datasets required by the FTMY pipeline.
They are generated once from raw historical meteorological data using a separate preprocessing script (preprocessing_historical_data.py) and are not created during FTMY execution. Users must generate these files in advance and set the environment variable FTMY_DAILY_SUMMARIES_PATH to the directory containing daily_summary_MM-DD.pkl files.

## Environment Setup

### Python Requirements

- Python 3.10 or higher
- Required packages: pandas, numpy, netCDF4, pathlib (standard library)

Install dependencies:
```bash
pip install pandas numpy netCDF4
```

### Environment Variables

The pipeline requires the following environment variables to be set before execution. No machine-specific defaults are provided.

**Required variables**:

- `FTMY_BASE_DIR`: Path to historical daily climate data directory (contains year subdirectories with `.txt` files)
- `FTMY_PREDICTION_PATH`: Path to directory containing NetCDF projection files
- `FTMY_DAILY_SUMMARIES_PATH`: Path to directory containing daily summary pickle files (`daily_summary_MM-DD.pkl`)

**Optional variables**:

- `FTMY_WORKSPACE_ROOT`: Path to repository root (auto-detected if not set)
- `FTMY_MODE`: Execution mode (`"test"` or `"full"`, defaults to `"full"`)

**Example setup (Windows PowerShell)**:
```powershell
$env:FTMY_BASE_DIR = "D:\Data\SyntheticWeatherFile"
$env:FTMY_PREDICTION_PATH = "D:\Data\SynthticWeatherFile\prediction"
$env:FTMY_DAILY_SUMMARIES_PATH = "D:\Data\SyntheticWeatherFile\historical_daily_climate_variables"
$env:FTMY_MODE = "test"  # or "full"
```

**Example setup (Linux/Mac)**:
```bash
export FTMY_BASE_DIR="/data/SyntheticWeatherFile"
export FTMY_PREDICTION_PATH="/data/SynthticWeatherFile/prediction"
export FTMY_DAILY_SUMMARIES_PATH="/data/SyntheticWeatherFile/historical_daily_climate_variables"
export FTMY_MODE="test"  # or "full"
```

## Running the Pipeline

### Command-Line Interface

The pipeline is executed via `Src/pipeline/interface_cli.py` with the following arguments:

```bash
python Src/pipeline/interface_cli.py \
    --lon <longitude> \
    --lat <latitude> \
    --start-year <start_year> \
    --end-year <end_year> \
    --projection-dir <path_to_netcdf_dir> \
    --scenario <scenario> \
    --gcm <gcm_name> \
    --rcm <rcm_name>
```

**Arguments**:
- `--lon`: Target longitude in decimal degrees (-180 to 180)
- `--lat`: Target latitude in decimal degrees (-90 to 90)
- `--start-year`: Start year of projection period (e.g., 2025)
- `--end-year`: End year of projection period (e.g., 2050)
- `--projection-dir`: Directory containing NetCDF files (can use `$env:FTMY_PREDICTION_PATH`)
- `--scenario`: Climate scenario (e.g.,`rcp85`)
- `--gcm`: Global Climate Model name (e.g., `ICHEC-EC-EARTH`)
- `--rcm`: Regional Climate Model name (e.g., `DMI-HIRHAM5`)


### Full Mode

Full mode processes all days from the FS-selected months, generating a complete synthetic weather year.

**Activate full mode** (default):
```powershell
# Windows PowerShell
$env:FTMY_MODE = "full"
# or simply omit FTMY_MODE (defaults to "full")
```

**Example full run**:
```bash
python Src/pipeline/interface_cli.py \
    --lon 12.5683 \
    --lat 55.6761 \
    --start-year 2025 \
    --end-year 2050 \
    --projection-dir "$env:FTMY_PREDICTION_PATH" \
    --scenario rcp85 \
    --gcm ICHEC-EC-EARTH \
    --rcm DMI-HIRHAM5
```

## Outputs

All outputs are written to `Output/results/` (full mode) or `Test_Output/results/` (test mode).

### Intermediate Data Files

- `daily_projection_data.csv`: Daily projection data for the target location
- `projection_selected_months.csv`: FS-selected months with scores and metadata
- `station_distances.csv`: Selected stations within 10 km with distances
- `historical_matches.csv`: Matched historical days for each projected day
- `historical_matching_diagnostics.csv`: Full candidate evaluation for transparency
- `synthetic_hourly_weather.csv`: Final hourly weather data
- `synthetic_hourly_sources.csv`: Source tracking (station ID and distance for each variable)
- `interpolation_log.csv`: Log of interpolated days (if any)

### Visualization Files

- `climate_yearly_profiles_complete.png`: Yearly profiles of key climate variables
- `station_contribution_plots_complete.png`: Contribution analysis by station

### Final Output

- `FTMY_<lat>_<lon>_<scenario>_<gcm>_<rcm>_<year>.epw`: EnergyPlus Weather file for building simulation

## Limitations

1. **Data Availability**: The pipeline requires manual download and placement of large external datasets (>20 GB). No automatic data acquisition is provided.

2. **Station Coverage**: The method requires at least one meteorological station within 10 km of the target location. If no stations are found, the pipeline will fail with an error.

3. **Missing Data Handling**: The pipeline applies interpolation only for single-day gaps (24 consecutive missing hours) when both neighboring days are complete. All other missing values remain as NaN in the output.

4. **Geographic Scope**: The current implementation is optimized for Danish meteorological data formats and CORDEX-style NetCDF projections. Adaptation may be required for other regions.

## Citation / Code Availability

This codebase implements the FTMY (Future Typical Meteorological Year) method as described in the associated academic publication. The repository is provided for reproducibility and research purposes. If you use this code, please cite the associated publication.

**Code Repository**: [future_weather_generation](https://github.com/AliRezaYaghoubi1998/future_weather_generation)

**Data Sources**:
- Historical observations: Danish Meteorological Institute (DMI) bulk download service
- Climate projections: DMI Climate Atlas (CORDEX bias-corrected data)

**License**: MIT License

## Author

Ali Reza Yaghoubi

## Supervisors
1. Rongling Li
2. Kristoffer Negendhal
