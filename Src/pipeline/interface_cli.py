"""
FTMY Pipeline CLI Interface

Main orchestrator for the FTMY pipeline. Coordinates all modules in sequence
without introducing fallback logic, ML steps, or gap-filling heuristics.

Reference behavior: references/interface.py (orchestration pattern only, not behavior)
"""

import sys
import os
import re
import argparse
from pathlib import Path
from typing import Tuple, Dict

# Add Src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from pipeline.config import PipelineConfig


def extract_epw_metadata(tas_filepath: str) -> Tuple[str, str, str]:
    """
    Extract scenario, GCM, and RCM metadata from NetCDF filename.
    
    Uses pattern-based parsing to handle various filename formats.
    Falls back to "UNKNOWN" if metadata cannot be resolved.
    
    Args:
        tas_filepath: Path to tas NetCDF file
        
    Returns:
        Tuple of (scenario, gcm, rcm)
    """
    tas_filename = os.path.basename(tas_filepath)
    
    # Initialize with defaults
    scenario = "UNKNOWN"
    gcm = "UNKNOWN"
    rcm = "UNKNOWN"
    
    # Pattern 1: Standard format
    # tas_KGDK-1_ICHEC-EC-EARTH_rcp85_r3i1p1_KNMI-RACMO22E_v1_day_20110101-20401231.nc
    parts = tas_filename.split('_')
    
    # Extract scenario (rcp26, rcp45, rcp85, ssp126, ssp245, ssp585, etc.)
    scenario_pattern = re.compile(r'(rcp\d+|ssp\d+)', re.IGNORECASE)
    for part in parts:
        match = scenario_pattern.search(part)
        if match:
            scenario = match.group(1).lower()
            break
    
    # Extract GCM (typically in position 2, but search for known patterns)
    # Known GCMs: ICHEC-EC-EARTH, MPI-ESM-LR, etc.
    known_gcms = ['ICHEC-EC-EARTH', 'MPI-ESM-LR', 'HadGEM2-ES', 'GFDL-ESM2M']
    for part in parts:
        for known_gcm in known_gcms:
            if known_gcm in part:
                gcm = known_gcm
                break
        if gcm != "UNKNOWN":
            break
    
    # If not found by pattern, try position-based (fallback)
    if gcm == "UNKNOWN" and len(parts) > 2:
        gcm = parts[2]  # Common position for GCM
    
    # Extract RCM (typically in position 5, but search for known patterns)
    # Known RCMs: KNMI-RACMO22E, DMI-HIRHAM5, etc.
    known_rcms = ['KNMI-RACMO22E', 'DMI-HIRHAM5', 'SMHI-RCA4']
    for part in parts:
        for known_rcm in known_rcms:
            if known_rcm in part:
                rcm = known_rcm
                break
        if rcm != "UNKNOWN":
            break
    
    # If not found by pattern, try position-based (fallback)
    if rcm == "UNKNOWN" and len(parts) > 5:
        rcm = parts[5]  # Common position for RCM
    
    # Log warning if any metadata is unresolved
    if scenario == "UNKNOWN" or gcm == "UNKNOWN" or rcm == "UNKNOWN":
        print(f"  WARNING: Could not fully resolve EPW metadata from filename: {tas_filename}")
        print(f"    Scenario: {scenario}, GCM: {gcm}, RCM: {rcm}")
        print(f"    EPW file will use 'UNKNOWN' for unresolved fields")
    
    return scenario, gcm, rcm


def parse_arguments() -> Tuple[float, float, int, int, Dict[str, str], str, str, str]:
    """
    Parse command-line arguments for the FTMY pipeline.
    
    Returns:
        Tuple of (lon, lat, start_year, end_year, netcdf_files, scenario, gcm, rcm)
        where netcdf_files is a dictionary with keys:
        tas, tasmax, tasmin, rsds, sfcWind, pr, potevap
    """
    parser = argparse.ArgumentParser(
        description="FTMY Pipeline - Generate synthetic weather files from climate projections"
    )
    
    parser.add_argument("--lon", type=float, required=True,
                       help="Target longitude (decimal degrees, -180 to 180)")
    parser.add_argument("--lat", type=float, required=True,
                       help="Target latitude (decimal degrees, -90 to 90)")
    parser.add_argument("--start-year", type=int, required=True,
                       help="Start year of projection period")
    parser.add_argument("--end-year", type=int, required=True,
                       help="End year of projection period")
    parser.add_argument("--projection-dir", type=Path, required=True,
                       help="Directory containing CORDEX-style NetCDF projection files")
    parser.add_argument("--scenario", type=str, required=True,
                       help="Climate scenario (e.g., rcp45, rcp85)")
    parser.add_argument("--gcm", type=str, required=True,
                       help="Global Climate Model name (e.g., ICHEC-EC-EARTH)")
    parser.add_argument("--rcm", type=str, required=True,
                       help="Regional Climate Model name (e.g., DMI-HIRHAM5)")
    
    args = parser.parse_args()
    
    # Extract values
    lon = args.lon
    lat = args.lat
    start_year = args.start_year
    end_year = args.end_year
    projection_dir = args.projection_dir
    scenario = args.scenario
    gcm = args.gcm
    rcm = args.rcm
    
    # Validate coordinates
    if not (-180 <= lon <= 180):
        raise ValueError("Longitude must be between -180 and 180")
    if not (-90 <= lat <= 90):
        raise ValueError("Latitude must be between -90 and 90")
    
    # Validate years
    if start_year < 1900 or end_year < 1900:
        raise ValueError("Years must be after 1900")
    if start_year > end_year:
        raise ValueError("Start year must be before or equal to end year")
    
    # Resolve projection files using projection_resolver
    from projection.projection_resolver import resolve_projection_files
    
    projection_files = resolve_projection_files(
        projection_dir=projection_dir,
        scenario=scenario,
        gcm=gcm,
        rcm=rcm,
        start_year=start_year,
        end_year=end_year
    )
    
    # Convert Dict[str, List[Path]] to Dict[str, str] for compatibility with projection_extractor
    # projection_extractor's find_relevant_files expects a single file path and will discover
    # related files, so we pass the first file from each list
    netcdf_files = {}
    for var, paths in projection_files.items():
        if not paths:
            raise ValueError(f"No files resolved for variable {var}")
        # Convert Path to string and use first file (find_relevant_files will discover others)
        netcdf_files[var] = str(paths[0])
    
    return lon, lat, start_year, end_year, netcdf_files, scenario, gcm, rcm


def main():
    """
    Main pipeline execution function.
    
    Orchestrates the complete FTMY pipeline:
    1. Configuration and validation
    2. FS selection + projection extraction
    3. Historical matching
    4. Hourly assembly
    5. Interpolation
    6. Visualization
    7. EPW generation
    """
    print("=" * 80)
    print("FTMY PIPELINE - CLEAN REBUILD")
    print("=" * 80)
    print()
    
    # Step 1: Parse arguments
    lon, lat, start_year, end_year, netcdf_files, scenario, gcm, rcm = parse_arguments()
    
    print(f"Target Location: {lat:.4f}°N, {lon:.4f}°E")
    print(f"Time Period: {start_year}-{end_year}")
    print()
    
    # Step 2: Print configuration and validate paths
    PipelineConfig.print_configuration()
    
    if not PipelineConfig.validate_paths():
        print("\nFATAL ERROR: Required input paths are missing.")
        print("Pipeline execution aborted.")
        sys.exit(1)
    
    results_path = PipelineConfig.get_results_path()
    print(f"\nResults will be saved to: {results_path}")
    print()
    
    # Step 3: FS Selection + Projection Extraction
    print("=" * 80)
    print("STEP 1: PROJECTION EXTRACTION AND FS SELECTION")
    print("=" * 80)
    try:
        from projection.projection_extractor import extract_daily_projections
        from projection.fs_selector import select_months_fs
        
        # Extract daily projections from NetCDF files
        print("\nExtracting daily projection data from NetCDF files...")
        daily_projection_data = extract_daily_projections(
            lon, lat, start_year, end_year, netcdf_files
        )
        
        # Enforce datetime type for 'date' column (Module 3 contract enforcement)
        if 'date' in daily_projection_data.columns:
            daily_projection_data['date'] = pd.to_datetime(daily_projection_data['date'])
        else:
            raise ValueError("Module 3 must return a 'date' column in daily_projection_data")
        
        # Save daily projection data
        projection_file = results_path / "daily_projection_data.csv"
        daily_projection_data.to_csv(projection_file, index=False)
        print(f"✓ Saved daily projection data: {projection_file}")
        
        # FS-based month selection (runs in both test and full mode)
        print("\nRunning FS-based month selection (temperature-based)...")
        selected_months = select_months_fs(daily_projection_data, start_year, end_year)
        
        # Save selected months
        months_file = results_path / "projection_selected_months.csv"
        selected_months.to_csv(months_file, index=False)
        print(f"✓ Saved selected months: {months_file}")
        print(f"  Selected {len(selected_months)} representative months")
        
    except ImportError as e:
        print(f"ERROR: Required module not found: {e}")
        print("This module must be implemented before pipeline can run.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR in projection extraction/FS selection: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 4: Station Selection
    print("\n" + "=" * 80)
    print("STEP 2: STATION SELECTION")
    print("=" * 80)
    try:
        from matching.station_selector import select_stations_within_10km
        
        print(f"\nSelecting stations within 10 km of ({lat:.4f}°N, {lon:.4f}°E)...")
        stations = select_stations_within_10km(lon, lat)
        
        if len(stations) == 0:
            print("FATAL ERROR: No stations found within 10 km.")
            print("Pipeline cannot proceed without valid stations.")
            sys.exit(1)
        
        print(f"✓ Found {len(stations)} stations within 10 km")
        
        # Save station distances
        stations_file = results_path / "station_distances.csv"
        stations.to_csv(stations_file, index=False)
        print(f"✓ Saved station distances: {stations_file}")
        
    except ImportError as e:
        print(f"ERROR: Required module not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR in station selection: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 5: Historical Matching
    print("\n" + "=" * 80)
    print("STEP 3: HISTORICAL DAY MATCHING")
    print("=" * 80)
    try:
        from matching.daily_matcher import match_historical_days
        
        # Filter projection data to FS-selected (month, year) pairs only
        # FS selection selects specific year for each calendar month, so we must filter
        # by both month AND selected_year to get ~365 days instead of thousands
        daily_projection_data['month'] = daily_projection_data['date'].dt.month
        daily_projection_data['year'] = daily_projection_data['date'].dt.year
        
        # Create set of (month, year) pairs from FS-selected months
        fs_selected_pairs = set(zip(selected_months['month'], selected_months['selected_year']))
        
        # Vectorized filtering: select only days where (month, year) matches FS-selected pairs
        month_year_pairs = list(zip(daily_projection_data['month'], daily_projection_data['year']))
        fs_selected_mask = pd.Series(month_year_pairs).isin(fs_selected_pairs)
        fs_selected_data = daily_projection_data[fs_selected_mask].copy()
        
        # Prepare FS-selected data for normalization (remove auxiliary columns)
        fs_selected_data_norm = fs_selected_data.drop(columns=['month', 'year']).copy()
        
        if PipelineConfig.is_test_mode():
            print("\n🧪 TEST MODE: Filtering to test days only...")
            print(f"  FS-selected (month, year) pairs: {sorted(fs_selected_pairs)}")
            print(f"  Test days to process: {PipelineConfig.TEST_DAYS}")
            
            # Further filter to test days only
            fs_selected_data['day'] = fs_selected_data['date'].dt.day
            test_days_set = set(PipelineConfig.TEST_DAYS)
            
            # Vectorized boolean mask: create string keys for efficient isin() lookup
            # Format: "MM-DD" as string for fast set membership check
            month_day_keys = (
                fs_selected_data['month'].astype(str).str.zfill(2) + '-' +
                fs_selected_data['day'].astype(str).str.zfill(2)
            )
            test_days_keys = {f"{m:02d}-{d:02d}" for m, d in test_days_set}
            mask = month_day_keys.isin(test_days_keys)
            
            filtered_data = fs_selected_data[mask].copy()
            filtered_data = filtered_data.drop(columns=['month', 'year', 'day'])
            print(f"  Filtered from {len(daily_projection_data)} to {len(fs_selected_data)} FS-selected days, then to {len(filtered_data)} test days")
        else:
            # Full mode: use all days from FS-selected (month, year) pairs
            filtered_data = fs_selected_data.copy()
            filtered_data = filtered_data.drop(columns=['month', 'year'])
            print(f"  Processing all days from FS-selected (month, year) pairs (~{len(filtered_data)} days)")
        
        print(f"\nMatching {len(filtered_data)} projected days to historical days...")
        # match_historical_days() receives filtered data for matching, and normalization
        # statistics are computed only from FS-selected projection days (~365 days)
        matches, diagnostics = match_historical_days(
            filtered_data, stations, full_projection_data=fs_selected_data_norm
        )
        
        # Save matching results
        matches_file = results_path / "historical_matches.csv"
        matches.to_csv(matches_file, index=False)
        print(f"✓ Saved historical matches: {matches_file}")
        
        # Save diagnostics
        diagnostics_file = results_path / "historical_matching_diagnostics.csv"
        diagnostics.to_csv(diagnostics_file, index=False)
        print(f"✓ Saved matching diagnostics: {diagnostics_file}")
        
    except ImportError as e:
        print(f"ERROR: Required module not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR in historical matching: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 6: Hourly Assembly
    print("\n" + "=" * 80)
    print("STEP 4: HOURLY ASSEMBLY")
    print("=" * 80)
    try:
        from assembly.hourly_assembler import assemble_hourly_data
        
        print(f"\nAssembling hourly data for {len(matches)} days...")
        hourly_data, hourly_sources = assemble_hourly_data(matches, stations)
        
        # Save hourly data
        hourly_file = results_path / "synthetic_hourly_weather.csv"
        hourly_data.to_csv(hourly_file, index=False)
        print(f"✓ Saved hourly weather data: {hourly_file}")
        
        sources_file = results_path / "synthetic_hourly_sources.csv"
        hourly_sources.to_csv(sources_file, index=False)
        print(f"✓ Saved hourly sources: {sources_file}")
        
    except ImportError as e:
        print(f"ERROR: Required module not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR in hourly assembly: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 7: Interpolation
    print("\n" + "=" * 80)
    print("STEP 5: INTERPOLATION")
    print("=" * 80)
    try:
        from assembly.interpolation import interpolate_missing_days
        
        print("\nApplying interpolation policy (one missing day only)...")
        # Interpolation must update both hourly_data and hourly_sources
        hourly_data, hourly_sources, interpolation_log = interpolate_missing_days(
            hourly_data, hourly_sources
        )
        
        # Save updated hourly data
        hourly_data.to_csv(hourly_file, index=False)
        print(f"✓ Updated hourly weather data: {hourly_file}")
        
        # Save updated hourly sources (interpolated values marked as "Interpolation")
        hourly_sources.to_csv(sources_file, index=False)
        print(f"✓ Updated hourly sources: {sources_file}")
        
        # Save interpolation log
        if len(interpolation_log) > 0:
            interp_file = results_path / "interpolation_log.csv"
            interpolation_log.to_csv(interp_file, index=False)
            print(f"✓ Saved interpolation log: {interp_file}")
            print(f"  Interpolated {len(interpolation_log)} missing days")
        else:
            print("  No interpolation applied (no single-day gaps found)")
        
    except ImportError as e:
        print(f"ERROR: Required module not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR in interpolation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 8: Visualization
    print("\n" + "=" * 80)
    print("STEP 6: VISUALIZATION")
    print("=" * 80)
    try:
        from visualization.climate_yearly_profiles import create_yearly_profiles
        from visualization.station_contribution_plots import create_contribution_plots
        from visualization.temperature_comparison import create_temperature_comparison
        
        print("\nGenerating visualizations...")
        
        # Climate yearly profiles
        create_yearly_profiles(hourly_data, results_path)
        print("✓ Created climate_yearly_profiles_complete.png")
        
        # Station contribution plots
        create_contribution_plots(hourly_sources, stations, results_path)
        print("✓ Created station_contribution_plots_complete.png")
        
        # Temperature comparison
        create_temperature_comparison(
            daily_projection_data, hourly_data, results_path
        )
        print("✓ Created temperature_comparison.png")
        
    except ImportError as e:
        print(f"ERROR: Required module not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR in visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 9: EPW Generation
    print("\n" + "=" * 80)
    print("STEP 7: EPW GENERATION")
    print("=" * 80)
    try:
        from epw.epw_writer import generate_epw_file
        
        # Use scenario/GCM/RCM from command-line arguments (already parsed)
        # No need to extract from filename since we have them directly
        
        print(f"\nGenerating EPW file...")
        print(f"  Scenario: {scenario}, GCM: {gcm}, RCM: {rcm}")
        
        epw_file = generate_epw_file(
            hourly_data, lon, lat, scenario, gcm, rcm, end_year, results_path
        )
        
        print(f"✓ Created EPW file: {epw_file}")
        
    except ImportError as e:
        print(f"ERROR: Required module not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR in EPW generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Final summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nAll outputs saved to: {results_path}")
    print("\nGenerated files:")
    print("  - projection_selected_months.csv")
    print("  - daily_projection_data.csv")
    print("  - station_distances.csv")
    print("  - historical_matches.csv")
    print("  - historical_matching_diagnostics.csv")
    print("  - synthetic_hourly_weather.csv")
    print("  - synthetic_hourly_sources.csv")
    print("  - interpolation_log.csv (if applicable)")
    print("  - climate_yearly_profiles_complete.png")
    print("  - station_contribution_plots_complete.png")
    print("  - temperature_comparison.png")
    print("  - FTMY_*.epw")
    print()
    
    if PipelineConfig.is_test_mode():
        print(PipelineConfig.TEST_MODE_DISCLAIMER)
        print()


if __name__ == "__main__":
    main()
