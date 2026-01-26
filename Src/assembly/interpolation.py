"""
Missing Value Interpolation

Applies the strict interpolation policy: interpolate only when exactly one
full day (24 hours) is missing for a variable, and both neighboring days
are complete for that variable.

Reference behavior: Replaces all previous interpolation logic (this is the only policy)
"""

import pandas as pd
import numpy as np
from typing import Tuple
from pathlib import Path
import sys

# Add Src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import PipelineConfig


def is_complete_day(series: pd.Series, day_start_idx: int) -> bool:
    """
    Check if a day (24 consecutive hours) is complete (no NaN values).
    
    Args:
        series: Series with hourly data
        day_start_idx: Starting index of the day (0-based)
        
    Returns:
        True if all 24 hours have valid (non-NaN) values, False otherwise
    """
    if day_start_idx + 24 > len(series):
        return False
    
    day_values = series.iloc[day_start_idx:day_start_idx + 24]
    return not day_values.isna().any()


def interpolate_missing_days(
    hourly_data: pd.DataFrame,
    hourly_sources: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply interpolation policy: interpolate only one missing day between two complete days.
    
    Policy:
    - Interpolate only when: exactly one full day (24 hours) is missing for a variable
    - AND both neighboring days are complete for that variable
    - Linear interpolation hour-by-hour between prev and next day
    - Everything else remains NaN
    
    Args:
        hourly_data: DataFrame with hourly weather data (columns: datetime, temp_dry_bulb_C, ...)
        hourly_sources: DataFrame with source tracking (columns: datetime, <var>_source_station_id, ...)
        
    Returns:
        Tuple of (updated_hourly_data, updated_hourly_sources, interpolation_log):
        - updated_hourly_data: Hourly data with interpolated values
        - updated_hourly_sources: Source tracking with interpolated values marked
        - interpolation_log: DataFrame with interpolation records
    """
    # Ensure datetime is datetime type
    hourly_data = hourly_data.copy()
    hourly_sources = hourly_sources.copy()
    
    hourly_data['datetime'] = pd.to_datetime(hourly_data['datetime'])
    hourly_sources['datetime'] = pd.to_datetime(hourly_sources['datetime'])
    
    # Sort by datetime to ensure chronological order
    hourly_data = hourly_data.sort_values('datetime').reset_index(drop=True)
    hourly_sources = hourly_sources.sort_values('datetime').reset_index(drop=True)
    
    # Verify both DataFrames have same length and datetime alignment
    if len(hourly_data) != len(hourly_sources):
        raise ValueError("hourly_data and hourly_sources must have same length")
    
    if not hourly_data['datetime'].equals(hourly_sources['datetime']):
        raise ValueError("hourly_data and hourly_sources must have aligned datetime columns")
    
    # Get list of climate variables (exclude datetime and source columns)
    climate_variables = [col for col in hourly_data.columns 
                        if col != 'datetime' and not col.endswith('_source')]
    
    interpolation_log = []
    
    # Group data by day to identify day boundaries
    # Each day should have exactly 24 hours
    hourly_data['date'] = pd.to_datetime(hourly_data['datetime']).dt.date
    unique_dates = sorted(hourly_data['date'].unique())
    
    # Process each variable independently
    for variable in climate_variables:
        if variable not in hourly_data.columns:
            continue
        
        series = hourly_data[variable].copy()
        
        # Check each day to see if it's missing (all 24 hours are NaN)
        for date_idx, date in enumerate(unique_dates):
            day_mask = hourly_data['date'] == date
            day_indices = hourly_data[day_mask].index.tolist()
            
            if len(day_indices) != 24:
                # Not exactly 24 hours for this day, skip (partial day)
                continue
            
            # Check if this day is completely missing (all 24 hours are NaN)
            day_values = series.iloc[day_indices]
            if not day_values.isna().all():
                continue  # Day is not completely missing, skip
            
            # Found a completely missing day
            missing_day_start_idx = day_indices[0]
            missing_day_end_idx = day_indices[-1]
            
            # Check if previous day exists and is complete
            prev_day_complete = False
            prev_day_start_idx = None
            if date_idx > 0:
                prev_date = unique_dates[date_idx - 1]
                prev_day_mask = hourly_data['date'] == prev_date
                prev_day_indices = hourly_data[prev_day_mask].index.tolist()
                
                if len(prev_day_indices) == 24:
                    prev_day_start_idx = prev_day_indices[0]
                    prev_day_complete = is_complete_day(series, prev_day_start_idx)
            
            # Check if next day exists and is complete
            next_day_complete = False
            next_day_start_idx = None
            if date_idx < len(unique_dates) - 1:
                next_date = unique_dates[date_idx + 1]
                next_day_mask = hourly_data['date'] == next_date
                next_day_indices = hourly_data[next_day_mask].index.tolist()
                
                if len(next_day_indices) == 24:
                    next_day_start_idx = next_day_indices[0]
                    next_day_complete = is_complete_day(series, next_day_start_idx)
            
            # Interpolate only if both neighboring days are complete
            if prev_day_complete and next_day_complete:
                # Get values from previous and next days
                prev_day_values = series.iloc[prev_day_start_idx:prev_day_start_idx + 24].values
                next_day_values = series.iloc[next_day_start_idx:next_day_start_idx + 24].values
                
                # Linear interpolation hour-by-hour
                # For each hour (0-23), interpolate between prev_day[hour] and next_day[hour]
                for hour_offset in range(24):
                    idx = missing_day_start_idx + hour_offset
                    prev_value = prev_day_values[hour_offset]
                    next_value = next_day_values[hour_offset]
                    
                    # Linear interpolation: value = prev + (next - prev) * 0.5
                    # (0.5 because we're halfway between the two days)
                    interpolated_value = prev_value + (next_value - prev_value) * 0.5
                    
                    hourly_data.loc[idx, variable] = interpolated_value
                    
                    # Mark source as "Interpolation"
                    source_col = f"{variable}_source_station_id"
                    distance_col = f"{variable}_source_distance_km"
                    if source_col in hourly_sources.columns:
                        hourly_sources.loc[idx, source_col] = "Interpolation"
                    if distance_col in hourly_sources.columns:
                        hourly_sources.loc[idx, distance_col] = np.nan
                
                # Record interpolation in log
                missing_date = hourly_data.loc[missing_day_start_idx, 'datetime']
                prev_date_obj = hourly_data.loc[prev_day_start_idx, 'datetime']
                next_date_obj = hourly_data.loc[next_day_start_idx, 'datetime']
                
                interpolation_log.append({
                    'missing_date': missing_date.strftime('%Y-%m-%d'),
                    'prev_date': prev_date_obj.strftime('%Y-%m-%d'),
                    'next_date': next_date_obj.strftime('%Y-%m-%d'),
                    'variables_interpolated': variable,
                    'method': 'linear',
                    'constraints_applied': 'one_missing_day_only'
                })
    
    # Remove temporary date column
    hourly_data = hourly_data.drop(columns=['date'])
    
    # Convert interpolation log to DataFrame
    if interpolation_log:
        interpolation_log_df = pd.DataFrame(interpolation_log)
    else:
        interpolation_log_df = pd.DataFrame(columns=[
            'missing_date', 'prev_date', 'next_date', 
            'variables_interpolated', 'method', 'constraints_applied'
        ])
    
    # Format datetime columns back to ISO string for CSV output
    hourly_data['datetime'] = hourly_data['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    hourly_sources['datetime'] = hourly_sources['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    if len(interpolation_log) > 0:
        print(f"\n✓ Interpolation applied: {len(interpolation_log)} missing days interpolated")
        print(f"  Variables interpolated: {len(interpolation_log_df['variables_interpolated'].unique())}")
    else:
        print(f"\n✓ No interpolation applied (no single-day gaps found)")
    
    return hourly_data, hourly_sources, interpolation_log_df
