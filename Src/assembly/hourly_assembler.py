"""
Hourly Data Assembler

Given selected historical days and stations within 10 km, extracts hourly series
for all variables from 10-minute historical data files.

For each variable, searches stations within 10 km in order of distance until
data is found. Data must only come from the selected historical day.

Reference behavior: references/Synthetic_File_Creator_Updated.py (extraction logic)
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple
from pathlib import Path
import sys

# Add Src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import PipelineConfig


# Parameter ID mappings for historical data
# These map to parameter IDs in the historical JSONL files
VARIABLE_MAPPING = {
    'temp_dry_bulb_C': ['temp_dry'],
    'wind_speed_m_s': ['wind_speed'],
    'wind_direction_degrees': ['wind_dir'],
    'temp_dew_C': ['temp_dew'],
    'humidity_percent': ['humidity'],
    'pressure_hPa': ['pressure', 'pressure_at_sea'],
    'precip_past1h_mm': ['precip_past10min'],
    'radiation_W_m2': ['radia_glob'],
    'cloud_cover_percent': ['cloud_cover']
}


def normalize_station_id(station_id) -> str:
    """
    Convert station ID to historical file format (5-digit string with leading zeros).
    
    Args:
        station_id: Station ID (can be int, float, or string)
        
    Returns:
        5-digit station ID string (e.g., "06188")
    """
    if isinstance(station_id, float):
        clean_id = str(int(station_id))  # 6188.0 -> "6188"
    else:
        clean_id = str(station_id)
    
    # Add leading zeros to make it 5 digits
    if len(clean_id) == 4:
        return f"0{clean_id}"  # "6188" -> "06188"
    elif len(clean_id) == 3:
        return f"00{clean_id}"  # "188" -> "00188"
    elif len(clean_id) == 2:
        return f"000{clean_id}"  # "88" -> "00088"
    elif len(clean_id) == 1:
        return f"0000{clean_id}"  # "8" -> "00008"
    else:
        return clean_id  # Already 5+ digits


def circular_mean(angles_degrees: list) -> float:
    """
    Calculate circular mean for angular data (e.g., wind direction).
    
    Converts angles to unit vectors, computes mean, then converts back to angle.
    Handles 0-360 degree range correctly.
    
    Args:
        angles_degrees: List of angles in degrees (0-360)
        
    Returns:
        Circular mean angle in degrees (0-360)
    """
    if not angles_degrees:
        return np.nan
    
    # Convert to radians
    angles_rad = np.deg2rad(angles_degrees)
    
    # Convert to unit vectors
    x = np.mean(np.cos(angles_rad))
    y = np.mean(np.sin(angles_rad))
    
    # Convert back to angle
    mean_rad = np.arctan2(y, x)
    mean_deg = np.rad2deg(mean_rad)
    
    # Normalize to 0-360 range
    if mean_deg < 0:
        mean_deg += 360
    
    return mean_deg


def load_historical_file(historical_file: str, stations: pd.DataFrame) -> dict:
    """
    Load and parse historical JSONL file once, organizing data by station and hour.
    
    This function reads the file once and returns a structured dictionary that can
    be reused for all variables, avoiding repeated file I/O.
    
    Args:
        historical_file: Path to historical JSONL file (YYYY-MM-DD.txt format)
        stations: DataFrame with stations within 10 km
        
    Returns:
        Dictionary structure: {station_id: {hour: {param_id: [values]}}}
        where station_id is the original station ID (not normalized)
    """
    # Create set of normalized station IDs for fast lookup
    station_id_map = {}
    normalized_station_ids = set()
    for _, station_row in stations.iterrows():
        station_id = station_row['station_id']
        normalized_id = normalize_station_id(station_id)
        station_id_map[normalized_id] = station_id
        normalized_station_ids.add(normalized_id)
    
    # Structure: {station_id: {hour: {param_id: [values]}}}
    parsed_data = {}
    
    try:
        with open(historical_file, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                props = record['properties']
                
                station_id_normalized = props['stationId']
                if station_id_normalized not in normalized_station_ids:
                    continue  # Skip stations not in our 10 km list
                
                # Get original station ID
                station_id = station_id_map[station_id_normalized]
                
                # Initialize station entry if needed
                if station_id not in parsed_data:
                    parsed_data[station_id] = {hour: {} for hour in range(24)}
                
                # Parse datetime to get hour
                dt = datetime.strptime(props['observed'], '%Y-%m-%dT%H:%M:%SZ')
                hour = dt.hour
                
                param_id = props['parameterId']
                
                try:
                    value = float(props['value'])
                    
                    # Initialize param entry if needed
                    if param_id not in parsed_data[station_id][hour]:
                        parsed_data[station_id][hour][param_id] = []
                    
                    parsed_data[station_id][hour][param_id].append(value)
                except (ValueError, TypeError):
                    continue
                    
    except Exception as e:
        print(f"Warning: Error loading historical file {historical_file}: {e}")
        return {}
    
    return parsed_data


def extract_hourly_data_for_variable_from_parsed(
    parsed_data: dict,
    variable: str,
    stations: pd.DataFrame
) -> Tuple[pd.Series, pd.Series]:
    """
    Extract hourly data for a specific variable from pre-parsed historical data.
    
    Searches stations within 10 km in order of distance until data is found.
    Tracks source station at the hour level (different hours can use different stations).
    
    Args:
        parsed_data: Pre-parsed data from load_historical_file()
        variable: Variable name (e.g., 'temp_dry_bulb_C')
        stations: DataFrame with stations sorted by distance (closest first)
        
    Returns:
        Tuple of (hourly_series, source_station_series)
        hourly_series: Series with 24 values (one per hour, 0-23)
        source_station_series: Series with 24 station IDs (one per hour, 0-23)
    """
    if variable not in VARIABLE_MAPPING:
        return pd.Series(index=range(24), dtype=float), pd.Series(index=range(24), dtype=object)
    
    param_ids = set(VARIABLE_MAPPING[variable])
    
    # Initialize output series
    hourly_series = pd.Series(index=range(24), dtype=float)
    source_station_series = pd.Series(index=range(24), dtype=object)
    
    # For each hour, search stations in order of distance
    for hour in range(24):
        hourly_values = []
        source_station_id = None
        
        # Search stations in order of distance (closest first)
        for _, station_row in stations.iterrows():
            station_id = station_row['station_id']
            
            if station_id not in parsed_data:
                continue
            
            # Check if this station has data for this hour and variable
            hour_data = parsed_data[station_id].get(hour, {})
            
            # Collect values for this variable's parameter IDs
            for param_id in param_ids:
                if param_id in hour_data:
                    hourly_values.extend(hour_data[param_id])
            
            # If we found data, use this station and stop searching
            if hourly_values:
                source_station_id = station_id
                break
        
        # Aggregate values for this hour
        if hourly_values:
            if variable == 'precip_past1h_mm':
                # Precipitation: sum of 10-minute values
                hourly_series[hour] = sum(hourly_values)
            elif variable == 'wind_direction_degrees':
                # Wind direction: circular mean (correctly handles 0-360 range)
                hourly_series[hour] = circular_mean(hourly_values)
            else:
                # All other variables: mean of 10-minute values
                hourly_series[hour] = np.mean(hourly_values)
            
            source_station_series[hour] = source_station_id
        else:
            hourly_series[hour] = np.nan
            source_station_series[hour] = None
    
    return hourly_series, source_station_series


def assemble_hourly_data(
    matches: pd.DataFrame,
    stations: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assemble hourly data for all matched historical days.
    
    For each matched day, extracts hourly data for all variables from the
    selected historical date. Searches stations within 10 km in order of
    distance for each variable.
    
    Args:
        matches: DataFrame from daily_matcher with columns:
            - target_date: target date (YYYY-MM-DD)
            - selected_historical_date: historical date (YYYY-MM-DD)
            - selected_station_id: station ID used for matching
        stations: DataFrame with stations within 10 km, sorted by distance
        
    Returns:
        Tuple of (hourly_weather_df, hourly_sources_df):
        - hourly_weather_df: Hourly weather data (columns: datetime, temp_dry_bulb_C, ...)
        - hourly_sources_df: Source tracking (columns: datetime, <var>_source_station_id, ...)
    """
    base_dir = PipelineConfig.BASE_DIR
    
    # Ensure stations are sorted by distance
    stations_sorted = stations.sort_values('distance_km').reset_index(drop=True)
    
    hourly_weather_list = []
    hourly_sources_list = []
    
    # Process each matched day
    for idx, match_row in matches.iterrows():
        target_date_str = match_row['target_date']
        historical_date_str = match_row['selected_historical_date']
        
        # Parse dates
        target_date = pd.to_datetime(target_date_str)
        historical_date = pd.to_datetime(historical_date_str)
        
        # Construct historical file path
        # Format: BASE_DIR/YYYY/YYYY-MM-DD.txt
        historical_year = historical_date.year
        historical_file = os.path.join(
            base_dir,
            str(historical_year),
            f"{historical_date_str}.txt"
        )
        
        if not os.path.exists(historical_file):
            print(f"Warning: Historical file not found: {historical_file}")
            # Create empty row with NaN values
            hourly_row = {'datetime': target_date}
            source_row = {'datetime': target_date}
            for var in VARIABLE_MAPPING.keys():
                hourly_row[var] = np.nan
                source_row[f"{var}_source_station_id"] = None
                source_row[f"{var}_source_distance_km"] = np.nan
            source_row['historical_date_used'] = historical_date_str
            
            hourly_weather_list.append(hourly_row)
            hourly_sources_list.append(source_row)
            continue
        
        # Load historical file ONCE and parse all records
        # This parsed data will be reused for all variables
        parsed_data = load_historical_file(historical_file, stations_sorted)
        
        if not parsed_data:
            print(f"Warning: No data found in historical file: {historical_file}")
            # Create empty records with NaN values
            for hour in range(24):
                hour_datetime = target_date.replace(hour=hour, minute=0, second=0)
                hourly_weather_list.append({
                    'datetime': hour_datetime,
                    **{var: np.nan for var in VARIABLE_MAPPING.keys()}
                })
                hourly_sources_list.append({
                    'datetime': hour_datetime,
                    **{f"{var}_source_station_id": None for var in VARIABLE_MAPPING.keys()},
                    **{f"{var}_source_distance_km": np.nan for var in VARIABLE_MAPPING.keys()},
                    'historical_date_used': historical_date_str
                })
            continue
        
        # Create 24 hourly records for this day
        day_hourly_weather = []
        day_hourly_sources = []
        
        for hour in range(24):
            hour_datetime = target_date.replace(hour=hour, minute=0, second=0)
            day_hourly_weather.append({
                'datetime': hour_datetime,
                **{var: np.nan for var in VARIABLE_MAPPING.keys()}
            })
            day_hourly_sources.append({
                'datetime': hour_datetime,
                **{f"{var}_source_station_id": None for var in VARIABLE_MAPPING.keys()},
                **{f"{var}_source_distance_km": np.nan for var in VARIABLE_MAPPING.keys()},
                'historical_date_used': historical_date_str
            })
        
        # Extract hourly data for each variable from pre-parsed data
        # Source station is tracked at the hour level (different hours can use different stations)
        for variable in VARIABLE_MAPPING.keys():
            hourly_series, source_station_series = extract_hourly_data_for_variable_from_parsed(
                parsed_data, variable, stations_sorted
            )
            
            # Fill in values and source tracking for this variable across all 24 hours
            for hour in range(24):
                day_hourly_weather[hour][variable] = hourly_series[hour]
                
                source_station_id = source_station_series[hour]
                if source_station_id is not None:
                    # Get distance for source station at this hour
                    station_info = stations_sorted[stations_sorted['station_id'] == source_station_id]
                    if len(station_info) > 0:
                        distance_km = station_info.iloc[0]['distance_km']
                    else:
                        distance_km = np.nan
                    
                    day_hourly_sources[hour][f"{variable}_source_station_id"] = source_station_id
                    day_hourly_sources[hour][f"{variable}_source_distance_km"] = distance_km
        
        # Add day's records to main lists
        hourly_weather_list.extend(day_hourly_weather)
        hourly_sources_list.extend(day_hourly_sources)
    
    # Convert to DataFrames
    hourly_weather_df = pd.DataFrame(hourly_weather_list)
    hourly_sources_df = pd.DataFrame(hourly_sources_list)
    
    # Sort by datetime
    hourly_weather_df = hourly_weather_df.sort_values('datetime').reset_index(drop=True)
    hourly_sources_df = hourly_sources_df.sort_values('datetime').reset_index(drop=True)
    
    # Format datetime column
    hourly_weather_df['datetime'] = pd.to_datetime(hourly_weather_df['datetime'])
    hourly_sources_df['datetime'] = pd.to_datetime(hourly_sources_df['datetime'])
    
    # Format datetime as ISO string for CSV output
    hourly_weather_df['datetime'] = hourly_weather_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    hourly_sources_df['datetime'] = hourly_sources_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"\n✓ Assembled hourly data: {len(hourly_weather_df)} hourly records")
    print(f"  Date range: {hourly_weather_df['datetime'].min()} to {hourly_weather_df['datetime'].max()}")
    
    # Count missing values
    for var in VARIABLE_MAPPING.keys():
        if var in hourly_weather_df.columns:
            missing = hourly_weather_df[var].isna().sum()
            if missing > 0:
                print(f"  {var}: {missing} missing hours ({missing/len(hourly_weather_df)*100:.1f}%)")
    
    return hourly_weather_df, hourly_sources_df
