"""
Station Selector

Builds list of valid stations within 10 km of target location.
Reads station metadata from daily summary pickle files.

Reference behavior: references/Synthetic_File_Creator_Updated.py (station metadata extraction)
"""

import os
import pickle
import pandas as pd
import glob
from pathlib import Path
import sys

# Add Src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import PipelineConfig
from projection.projection_extractor import get_utm_zone_epsg, calculate_utm_distance


def find_daily_summary_file(daily_summaries_path: str, preferred_file: str = None) -> str:
    """
    Find a daily summary pickle file to use for station metadata extraction.
    
    Helper function to avoid duplication across modules that need to read
    daily summary files. Any daily summary file contains the same station metadata.
    
    Args:
        daily_summaries_path: Path to daily summaries directory
        preferred_file: Preferred filename (e.g., 'daily_summary_01-01.pkl')
                       If None or not found, uses first available file
        
    Returns:
        Path to a daily summary file
        
    Raises:
        FileNotFoundError: If no daily summary files found
    """
    if preferred_file:
        preferred_path = os.path.join(daily_summaries_path, preferred_file)
        if os.path.exists(preferred_path):
            return preferred_path
    
    # Try to find any daily summary file
    pattern = os.path.join(daily_summaries_path, 'daily_summary_*.pkl')
    sample_files = glob.glob(pattern)
    
    if not sample_files:
        raise FileNotFoundError(
            f"No daily summary files found in: {daily_summaries_path}\n"
            f"Expected files like: daily_summary_MM-DD.pkl"
        )
    
    return sorted(sample_files)[0]  # Return first file alphabetically


def check_station_metadata_consistency(daily_summaries_path: str, sample_file: str) -> None:
    """
    Check if station metadata is consistent across daily summary files.
    
    Loads 2-3 additional files and compares station coordinates to detect
    inconsistencies that might indicate data quality issues.
    
    Args:
        daily_summaries_path: Path to daily summaries directory
        sample_file: Path to the sample file already loaded
        
    Logs warnings if inconsistencies are detected.
    """
    # Find additional files to check (up to 2 more)
    pattern = os.path.join(daily_summaries_path, 'daily_summary_*.pkl')
    all_files = sorted(glob.glob(pattern))
    
    # Exclude the sample file we already loaded
    check_files = [f for f in all_files if f != sample_file][:2]
    
    if len(check_files) == 0:
        return  # Only one file available, can't check consistency
    
    try:
        # Load sample file metadata
        with open(sample_file, 'rb') as f:
            sample_data = pickle.load(f)
        sample_metadata = sample_data.get('station_metadata', {})
        
        inconsistencies = []
        
        for check_file in check_files:
            try:
                with open(check_file, 'rb') as f:
                    check_data = pickle.load(f)
                check_metadata = check_data.get('station_metadata', {})
                
                # Compare coordinates for common stations
                for station_id in sample_metadata.keys():
                    if station_id in check_metadata:
                        sample_lat = sample_metadata[station_id].get('latitude')
                        sample_lon = sample_metadata[station_id].get('longitude')
                        check_lat = check_metadata[station_id].get('latitude')
                        check_lon = check_metadata[station_id].get('longitude')
                        
                        if (sample_lat is not None and check_lat is not None and
                            sample_lon is not None and check_lon is not None):
                            # Check if coordinates differ significantly (> 0.001 degrees ≈ 100m)
                            lat_diff = abs(float(sample_lat) - float(check_lat))
                            lon_diff = abs(float(sample_lon) - float(check_lon))
                            
                            if lat_diff > 0.001 or lon_diff > 0.001:
                                inconsistencies.append({
                                    'station_id': station_id,
                                    'file1': os.path.basename(sample_file),
                                    'file2': os.path.basename(check_file),
                                    'lat_diff': lat_diff,
                                    'lon_diff': lon_diff
                                })
            except Exception as e:
                # Skip files that can't be loaded
                continue
        
        if inconsistencies:
            print(f"  WARNING: Detected {len(inconsistencies)} station coordinate inconsistencies:")
            for inc in inconsistencies[:5]:  # Show first 5
                print(f"    Station {inc['station_id']}: "
                      f"lat_diff={inc['lat_diff']:.6f}°, lon_diff={inc['lon_diff']:.6f}° "
                      f"between {inc['file1']} and {inc['file2']}")
            if len(inconsistencies) > 5:
                print(f"    ... and {len(inconsistencies) - 5} more")
            print(f"  Using metadata from: {os.path.basename(sample_file)}")
    
    except Exception as e:
        # Don't fail on consistency check errors, just log
        print(f"  Warning: Could not check station metadata consistency: {e}")


def select_stations_within_10km(lon: float, lat: float) -> pd.DataFrame:
    """
    Select stations within 10 km of target location.
    
    Reads station metadata from daily summary pickle files and filters
    to only stations within 10 km (hard cutoff).
    
    Args:
        lon: Target longitude (decimal degrees)
        lat: Target latitude (decimal degrees)
        
    Returns:
        DataFrame with columns:
        - station_id: station identifier
        - latitude: station latitude (decimal degrees)
        - longitude: station longitude (decimal degrees)
        - distance_km: distance from target location (kilometers)
        
        Stations are sorted by distance (closest first).
        
    Raises:
        ValueError: If no stations found within 10 km
        FileNotFoundError: If daily summary files not found
    """
    daily_summaries_path = PipelineConfig.DAILY_SUMMARIES_PATH
    
    if not os.path.exists(daily_summaries_path):
        raise FileNotFoundError(
            f"Daily summaries directory not found: {daily_summaries_path}\n"
            f"Please check FTMY_DAILY_SUMMARIES_PATH environment variable or config."
        )
    
    # Find a daily summary file to use (helper function for reuse)
    sample_file = find_daily_summary_file(daily_summaries_path, 'daily_summary_01-01.pkl')
    print(f"Loading stations from: {os.path.basename(sample_file)}")
    
    # Check station metadata consistency across files
    check_station_metadata_consistency(daily_summaries_path, sample_file)
    
    try:
        with open(sample_file, 'rb') as f:
            daily_data = pickle.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load daily summary file {sample_file}: {e}")
    
    # Extract and process stations
    stations_data = []
    seen_stations = set()
    
    # Extract station metadata from the pickle file structure
    # Structure: daily_data['station_metadata'] = {station_id: {latitude, longitude, ...}}
    station_metadata = daily_data.get('station_metadata', {})
    
    if not station_metadata:
        raise ValueError(
            f"No station_metadata found in {sample_file}.\n"
            f"File structure may be incorrect."
        )
    
    # Process each station
    for station_id, station_meta in station_metadata.items():
        if station_id in seen_stations:
            continue
        
        # Check if station has valid coordinates
        station_lat = station_meta.get('latitude')
        station_lon = station_meta.get('longitude')
        
        if station_lat is None or station_lon is None:
            continue
        
        # Calculate distance from target location using UTM coordinates
        # This is consistent with Module 3's distance calculation method
        try:
            # Determine UTM zone for target location
            utm_epsg = get_utm_zone_epsg(lon, lat)
            
            # Calculate UTM-based distance (more accurate for local distances)
            distance_km = calculate_utm_distance(
                lon, lat,
                float(station_lon), float(station_lat),
                utm_epsg
            )
        except Exception as e:
            print(f"Warning: Failed to calculate distance for station {station_id}: {e}")
            continue
        
        stations_data.append({
            'station_id': station_id,
            'latitude': float(station_lat),
            'longitude': float(station_lon),
            'distance_km': float(distance_km)
        })
        seen_stations.add(station_id)
    
    if len(stations_data) == 0:
        raise ValueError(
            f"No stations with valid coordinates found in {sample_file}.\n"
            f"Please check the daily summary file structure."
        )
    
    # Convert to DataFrame
    stations_df = pd.DataFrame(stations_data)
    
    print(f"  Found {len(stations_df)} total stations with coordinates")
    
    # HARD DISTANCE FILTER: Exclude any station with distance_km > 10.0
    stations_within_10km = stations_df[stations_df['distance_km'] <= 10.0].copy()
    
    if len(stations_within_10km) == 0:
        # Fail early with clear message
        closest_station = stations_df.loc[stations_df['distance_km'].idxmin()]
        closest_distance = closest_station['distance_km']
        
        raise ValueError(
            f"No stations found within 10 km of target location ({lat:.4f}°N, {lon:.4f}°E).\n"
            f"Closest station is {closest_distance:.2f} km away (station_id: {closest_station['station_id']}).\n"
            f"Pipeline cannot proceed without valid stations within 10 km."
        )
    
    # Sort by distance (closest first)
    stations_within_10km = stations_within_10km.sort_values('distance_km').reset_index(drop=True)
    
    print(f"  Filtered to {len(stations_within_10km)} stations within 10 km")
    print(f"  Distance range: {stations_within_10km['distance_km'].min():.2f} - "
          f"{stations_within_10km['distance_km'].max():.2f} km")
    
    # Save station distances to CSV
    results_path = PipelineConfig.get_results_path()
    stations_file = results_path / "station_distances.csv"
    
    # Add target location information as comments in the CSV
    utm_epsg = get_utm_zone_epsg(lon, lat)
    with open(stations_file, 'w', encoding='utf-8') as f:
        f.write(f"# Target Location: Longitude={lon}, Latitude={lat}\n")
        f.write(f"# Distance calculation: UTM-based ({utm_epsg})\n")
        f.write(f"# HARD DISTANCE LIMIT: Only stations within 10 km are included\n")
        f.write(f"# Total stations found: {len(stations_within_10km)}\n")
    
    # Append the actual station data
    stations_within_10km.to_csv(stations_file, index=False, mode='a')
    
    print(f"✓ Saved station distances to: {stations_file}")
    
    return stations_within_10km
