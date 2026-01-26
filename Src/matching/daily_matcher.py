"""
Daily Historical Day Matcher

For each projected day, finds the best matching historical day using normalized
distance formula. Matches same calendar day only (month-day) across all historical years.

Reference behavior: references/Synthetic_File_Creator_Updated.py (matching logic patterns)
"""

import os
import pickle
import pandas as pd
import numpy as np
from typing import Tuple
from pathlib import Path
import sys

# Add Src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import PipelineConfig
from matching.station_selector import find_daily_summary_file


# Parameter ID mappings for historical data
# These map to parameter IDs in the daily summary pickle files
VARIABLE_MAPPING = {
    'temp_dry_bulb_C': ['temp_dry'],
    'wind_speed_m_s': ['wind_speed'],
    'radiation_W_m2': ['radia_glob'],
}

# Distance weights for normalized distance formula
DISTANCE_WEIGHTS = {
    'temp_dry_bulb_C': 0.40,
    'radiation_W_m2': 0.30,
    'wind_speed_m_s': 0.30,
}


def load_daily_summary(month_day: str) -> dict:
    """
    Load daily summary pickle file for a specific month-day.
    
    Args:
        month_day: Month-day string (e.g., "01-15")
        
    Returns:
        Daily summary dictionary or None if file doesn't exist
    """
    daily_summaries_path = PipelineConfig.DAILY_SUMMARIES_PATH
    daily_summary_file = os.path.join(
        daily_summaries_path,
        f"daily_summary_{month_day}.pkl"
    )
    
    if not os.path.exists(daily_summary_file):
        return None
    
    try:
        with open(daily_summary_file, 'rb') as f:
            daily_data = pickle.load(f)
        return daily_data
    except Exception as e:
        print(f"Warning: Error loading daily summary for {month_day}: {e}")
        return None


def aggregate_radiation_area(
    historical_year: int,
    month_day: str,
    stations: pd.DataFrame
) -> float:
    """
    Aggregate solar radiation from all stations within 10 km for a specific historical day.
    
    Uses distance-weighted mean (closer stations weighted higher).
    
    Args:
        historical_year: Year of the historical day
        month_day: Month-day string (e.g., "01-15")
        stations: DataFrame with stations within 10 km, including distance_km
        
    Returns:
        Aggregated daily radiation value (W/m²), or None if no valid data
    """
    daily_data = load_daily_summary(month_day)
    if daily_data is None:
        return None
    
    try:
        station_ids = set(stations['station_id'].tolist())
        station_distance_map = dict(zip(stations['station_id'], stations['distance_km']))
        rad_param_ids = set(VARIABLE_MAPPING.get('radiation_W_m2', []))
        
        # Find the specific year's data
        radiation_values = []
        distances = []
        
        for yearly_stat in daily_data.get('yearly_stats', []):
            if yearly_stat['year'] != historical_year:
                continue
            
            for stat in yearly_stat.get('stats', []):
                station_id = stat['station_id']
                if station_id not in station_ids:
                    continue
                
                param_id = stat['parameter_id']
                if param_id in rad_param_ids:
                    radiation_value = stat.get('mean')
                    if radiation_value is not None and not pd.isna(radiation_value):
                        radiation_values.append(radiation_value)
                        if station_id in station_distance_map:
                            distances.append(station_distance_map[station_id])
        
        if not radiation_values:
            return None
        
        # Use distance-weighted mean
        if len(distances) == len(radiation_values) and len(distances) > 1:
            weights = [1.0 / (d + 1.0) for d in distances]
            total_weight = sum(weights)
            weighted_sum = sum(r * w for r, w in zip(radiation_values, weights))
            aggregated_value = weighted_sum / total_weight if total_weight > 0 else sum(radiation_values) / len(radiation_values)
        else:
            # Simple mean if distances not available
            aggregated_value = sum(radiation_values) / len(radiation_values)
        
        return aggregated_value
        
    except Exception as e:
        print(f"Warning: Error aggregating radiation for {historical_year}-{month_day}: {e}")
        return None


def compute_normalized_distance(
    target_values: dict,
    candidate_values: dict,
    norm_denominators: dict
) -> Tuple[float, dict]:
    """
    Compute normalized weighted distance metric for ranking candidates.
    
    ΔT_norm = |T_hist − T_proj| / max(|T_proj|)
    ΔR_norm = |R_hist − R_proj| / max(R_proj)
    ΔW_norm = |W_hist − W_proj| / max(W_proj)
    ΔD = 0.4 * ΔT_norm + 0.3 * ΔR_norm + 0.3 * ΔW_norm
    
    Args:
        target_values: Dict with T_proj, R_proj, W_proj
        candidate_values: Dict with temp_dry_bulb_C, radiation_W_m2, wind_speed_m_s
        norm_denominators: Dict with max values for normalization
        
    Returns:
        Tuple of (delta_D, normalized_deltas_dict) or (None, None) if any required variable is missing
    """
    # Map projection variable names to internal variable names
    var_mapping = {
        'T_proj': 'temp_dry_bulb_C',
        'R_proj': 'radiation_W_m2',
        'W_proj': 'wind_speed_m_s',
    }
    
    normalized_deltas = {}
    distance = 0.0
    
    for proj_var, internal_var in var_mapping.items():
        weight = DISTANCE_WEIGHTS[internal_var]
        
        # Get target value (from projection data)
        target = target_values.get(proj_var)
        
        # Get candidate value (from historical data)
        candidate = candidate_values.get(internal_var)
        
        if target is None or pd.isna(target) or candidate is None or pd.isna(candidate):
            return None, None
        
        abs_diff = abs(candidate - target)
        norm_denom = norm_denominators.get(internal_var, 0.0)
        
        # Handle division-by-zero: fallback to raw difference if max = 0
        if norm_denom > 0:
            normalized_delta = abs_diff / norm_denom
        else:
            normalized_delta = abs_diff  # Fallback to raw difference
        
        normalized_deltas[internal_var] = normalized_delta
        distance += weight * normalized_delta
    
    return distance, normalized_deltas


def match_historical_days(
    filtered_data: pd.DataFrame,
    stations: pd.DataFrame,
    full_projection_data: pd.DataFrame = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Match projected days to historical days using normalized distance formula.
    
    For each projected day, finds the best matching historical day (same calendar day)
    from stations within 10 km using the normalized distance metric.
    
    IMPORTANT: Radiation (R_hist) is area-aggregated from all stations within 10 km
    for each historical day. This means all station candidates for a given historical
    day share the same R_hist value. Only temperature and wind speed vary by station.
    
    Args:
        filtered_data: DataFrame with projected days to match (columns: date, T_proj, R_proj, W_proj)
        stations: DataFrame with stations within 10 km (columns: station_id, latitude, longitude, distance_km)
        full_projection_data: Optional FS-selected projection dataset for normalization denominators.
                            If None, uses filtered_data. Should contain only FS-selected projection days
                            (~365 days) to ensure normalization statistics are computed efficiently.
        
    Returns:
        Tuple of (matches_df, diagnostics_df):
        - matches_df: Selected matches only (one row per target day)
        - diagnostics_df: All candidates ranked (many rows per target day)
    """
    # Ensure date is datetime
    filtered_data = filtered_data.copy()
    filtered_data['date'] = pd.to_datetime(filtered_data['date'])
    
    # Filter stations to only those within 10 km (hard limit)
    stations_within_10km = stations[stations['distance_km'] <= 10.0].copy()
    
    if len(stations_within_10km) == 0:
        raise ValueError("No stations found within 10 km. Cannot perform matching.")
    
    station_ids = set(stations_within_10km['station_id'].tolist())
    station_distance_map = dict(zip(stations_within_10km['station_id'], stations_within_10km['distance_km']))
    
    # Precompute parameter ID sets for fast lookup
    temp_param_ids = set(VARIABLE_MAPPING.get('temp_dry_bulb_C', []))
    wind_param_ids = set(VARIABLE_MAPPING.get('wind_speed_m_s', []))
    rad_param_ids = set(VARIABLE_MAPPING.get('radiation_W_m2', []))
    
    # Compute normalization denominators from FS-selected projection days only (~365 days)
    # This ensures normalization is based on the same dataset used for matching, avoiding
    # unnecessary computation on the full projection dataset (~9500 days)
    # den_T = max(|T_proj|), den_R = max(R_proj), den_W = max(W_proj)
    projection_data_for_norm = full_projection_data if full_projection_data is not None else filtered_data
    
    if full_projection_data is not None:
        print(f"  Using FS-selected projection dataset for normalization ({len(full_projection_data)} days)")
    else:
        print(f"  Using filtered dataset for normalization ({len(filtered_data)} days)")
    
    den_T = projection_data_for_norm['T_proj'].abs().max() if 'T_proj' in projection_data_for_norm.columns else 0.0
    den_R = projection_data_for_norm['R_proj'].max() if 'R_proj' in projection_data_for_norm.columns else 0.0
    den_W = projection_data_for_norm['W_proj'].max() if 'W_proj' in projection_data_for_norm.columns else 0.0
    
    norm_denominators = {
        'temp_dry_bulb_C': den_T if den_T > 0 else 0.0,
        'radiation_W_m2': den_R if den_R > 0 else 0.0,
        'wind_speed_m_s': den_W if den_W > 0 else 0.0,
    }
    
    if all(denom == 0.0 for denom in norm_denominators.values()):
        print("Warning: All normalization denominators are zero. Using raw differences.")
    
    # Process each target day
    matches_list = []
    diagnostics_list = []
    
    for idx, row in filtered_data.iterrows():
        target_date = row['date']
        month_day = target_date.strftime("%m-%d")
        target_date_str = target_date.strftime('%Y-%m-%d')
        
        # Extract target values
        target_values = {
            'T_proj': row.get('T_proj'),
            'R_proj': row.get('R_proj'),
            'W_proj': row.get('W_proj'),
        }
        
        # Check if target values are valid
        if any(pd.isna(val) or val is None for val in target_values.values()):
            print(f"Warning: Missing target values for {target_date_str}, skipping")
            continue
        
        # Load daily summary for this month-day
        daily_data = load_daily_summary(month_day)
        if daily_data is None:
            print(f"Warning: No daily summary found for {month_day}, skipping {target_date_str}")
            continue
        
        # Build candidate list
        raw_candidates = []
        
        # Cache for area-aggregated radiation: computed once per (historical_year, month_day)
        # All station candidates for the same historical day share the same R_hist value
        radiation_cache = {}
        
        for yearly_stat in daily_data.get('yearly_stats', []):
            historical_year = yearly_stat['year']
            historical_date = f"{historical_year}-{month_day}"
            
            # Compute area-aggregated radiation ONCE per (historical_year, month_day)
            # This value is shared by all station candidates for this historical day
            cache_key = (historical_year, month_day)
            if cache_key not in radiation_cache:
                aggregated_radiation = aggregate_radiation_area(
                    historical_year, month_day, stations_within_10km
                )
                radiation_cache[cache_key] = aggregated_radiation
            else:
                aggregated_radiation = radiation_cache[cache_key]
            
            # Collect temperature and wind data per station
            # Note: Temperature and wind are station-specific, but radiation is area-aggregated
            station_temp_wind = {}  # {station_id: {'temp': value, 'wind': value, 'distance': km}}
            
            for stat in yearly_stat.get('stats', []):
                station_id = stat['station_id']
                if station_id not in station_ids:
                    continue
                
                param_id = stat['parameter_id']
                mean_value = stat.get('mean')
                
                if mean_value is None or pd.isna(mean_value):
                    continue
                
                if station_id not in station_temp_wind:
                    if station_id in station_distance_map:
                        station_temp_wind[station_id] = {
                            'distance_km': station_distance_map[station_id]
                        }
                    else:
                        continue
                
                # Map parameter to variable
                if param_id in temp_param_ids:
                    station_temp_wind[station_id]['temp_dry_bulb_C'] = mean_value
                elif param_id in wind_param_ids:
                    station_temp_wind[station_id]['wind_speed_m_s'] = mean_value
            
            # Create one candidate per station (for temp/wind), but use shared aggregated radiation
            # IMPORTANT: All candidates for this (historical_year, month_day) share the same R_hist
            for station_id, station_data in station_temp_wind.items():
                if 'temp_dry_bulb_C' not in station_data:
                    continue  # Must have temperature
                
                candidate_values = {
                    'temp_dry_bulb_C': station_data['temp_dry_bulb_C'],
                    'wind_speed_m_s': station_data.get('wind_speed_m_s'),
                    'radiation_W_m2': aggregated_radiation  # Area-aggregated value
                }
                
                # Skip if any required variable is missing
                if any(
                    candidate_values[var] is None or pd.isna(candidate_values[var])
                    for var in ['temp_dry_bulb_C', 'radiation_W_m2', 'wind_speed_m_s']
                ):
                    continue
                
                raw_candidates.append({
                    'historical_year': historical_year,
                    'historical_date': historical_date,
                    'station_id': station_id,
                    'distance_km': station_data['distance_km'],
                    'candidate_values': candidate_values
                })
        
        if len(raw_candidates) == 0:
            print(f"Warning: No candidates found for {target_date_str}")
            continue
        
        # Compute normalized distances for all candidates
        all_candidates = []
        for raw_cand in raw_candidates:
            candidate_values = raw_cand['candidate_values']
            delta_D, normalized_deltas = compute_normalized_distance(
                target_values, candidate_values, norm_denominators
            )
            if delta_D is None:
                continue
            
            candidate_info = {
                'historical_year': raw_cand['historical_year'],
                'historical_date': raw_cand['historical_date'],
                'station_id': raw_cand['station_id'],
                'distance_km': raw_cand['distance_km'],
                'candidate_values': candidate_values,
                'normalized_deltas': normalized_deltas,
                'delta_D': delta_D
            }
            all_candidates.append(candidate_info)
        
        if len(all_candidates) == 0:
            print(f"Warning: No valid candidates after distance calculation for {target_date_str}")
            continue
        
        # Select best candidate (minimum delta_D, with tie-breaking)
        selected_candidate = min(
            all_candidates,
            key=lambda x: (x['delta_D'], x['distance_km'], -x['historical_year'])
        )
        
        # Sort all candidates by delta_D for ranking (diagnostic only)
        sorted_candidates = sorted(
            all_candidates,
            key=lambda x: (x['delta_D'], x['distance_km'], -x['historical_year'])
        )
        
        # Create match record (selected only)
        match_record = {
            'target_date': target_date_str,
            'selected_historical_year': selected_candidate['historical_year'],
            'selected_historical_date': selected_candidate['historical_date'],
            'selected_station_id': selected_candidate['station_id'],
            'T_proj': target_values['T_proj'],
            'R_proj': target_values['R_proj'],
            'W_proj': target_values['W_proj'],
            'T_hist': selected_candidate['candidate_values']['temp_dry_bulb_C'],
            'R_hist': selected_candidate['candidate_values']['radiation_W_m2'],
            'W_hist': selected_candidate['candidate_values']['wind_speed_m_s'],
            'dT_norm': round(selected_candidate['normalized_deltas'].get('temp_dry_bulb_C', 0.0), 6),
            'dR_norm': round(selected_candidate['normalized_deltas'].get('radiation_W_m2', 0.0), 6),
            'dW_norm': round(selected_candidate['normalized_deltas'].get('wind_speed_m_s', 0.0), 6),
            'delta_D': round(selected_candidate['delta_D'], 6),
            'rank': 1  # Always rank 1 for selected match
        }
        matches_list.append(match_record)
        
        # Create diagnostic records for ALL candidates
        for rank, candidate in enumerate(sorted_candidates, start=1):
            normalized_deltas = candidate.get('normalized_deltas', {})
            diagnostic_record = {
                'target_date': target_date_str,
                'candidate_date': candidate['historical_date'],
                'candidate_station_id': candidate['station_id'],
                'T_proj': target_values['T_proj'],
                'R_proj': target_values['R_proj'],
                'W_proj': target_values['W_proj'],
                'T_hist': candidate['candidate_values'].get('temp_dry_bulb_C'),
                'R_hist': candidate['candidate_values'].get('radiation_W_m2'),
                'W_hist': candidate['candidate_values'].get('wind_speed_m_s'),
                'delta_temperature_norm': round(normalized_deltas.get('temp_dry_bulb_C', 0.0), 6),
                'delta_solar_radiation_norm': round(normalized_deltas.get('radiation_W_m2', 0.0), 6),
                'delta_wind_speed_norm': round(normalized_deltas.get('wind_speed_m_s', 0.0), 6),
                'weighted_distance': round(candidate['delta_D'], 6),
                'rank': rank
            }
            diagnostics_list.append(diagnostic_record)
    
    # Convert to DataFrames
    matches_df = pd.DataFrame(matches_list)
    diagnostics_df = pd.DataFrame(diagnostics_list)
    
    print(f"\n✓ Matching completed: {len(matches_df)} days matched")
    if len(diagnostics_list) > 0:
        print(f"  Total candidate evaluations: {len(diagnostics_list)}")
        print(f"  Average candidates per day: {len(diagnostics_list) / len(matches_df):.1f}")
    
    return matches_df, diagnostics_df
