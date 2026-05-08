"""
FS (Finkelstein-Schafer) Month Selector

Selects representative months using FS distance on temperature only.
For each calendar month, chooses the year that minimizes FS distance
against the long-term distribution of that month.

Reference behavior: references/fs_selector.py
"""

import numpy as np
import pandas as pd
from typing import Optional


# Number of quantiles for FS distance calculation
# Standard TMY approach uses 12 quantiles (0, 1/12, 2/12, ..., 11/12, 1.0)
N_QUANTILES = 12


def calculate_cdf(values: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """
    Calculate cumulative distribution function (CDF) values at specified quantiles.
    
    Args:
        values: Array of daily values for a month
        quantiles: Array of quantile levels (e.g., [0, 1/12, 2/12, ..., 11/12, 1])
    
    Returns:
        Array of CDF values at each quantile level
    """
    if len(values) == 0:
        return np.full(len(quantiles), np.nan)
    
    # Remove NaN values
    values = values[~np.isnan(values)]
    
    if len(values) == 0:
        return np.full(len(quantiles), np.nan)
    
    # Calculate quantiles from the data
    cdf_values = np.quantile(values, quantiles, method='linear')
    
    return cdf_values


def calculate_fs_distance(cdf_observed: np.ndarray, cdf_target: np.ndarray) -> float:
    """
    Calculate Finkelstein-Schafer (FS) distance between two CDFs.
    
    FS distance = (1/n) * Σ |CDF_obs(i) - CDF_target(i)|
    
    Args:
        cdf_observed: CDF values from candidate month-year
        cdf_target: CDF values from long-term reference (target)
    
    Returns:
        FS distance (lower is better match)
    """
    if len(cdf_observed) != len(cdf_target):
        raise ValueError("CDFs must have same length")
    
    # Check for NaN values
    valid_mask = ~(np.isnan(cdf_observed) | np.isnan(cdf_target))
    if not np.any(valid_mask):
        return np.inf  # No valid data
    
    # Calculate FS distance only over valid quantiles
    valid_observed = cdf_observed[valid_mask]
    valid_target = cdf_target[valid_mask]
    
    # FS distance: mean absolute difference
    fs_distance = np.mean(np.abs(valid_observed - valid_target))
    
    return fs_distance


def select_months_fs(
    daily_projection_data: pd.DataFrame,
    start_year: int,
    end_year: int
) -> pd.DataFrame:
    """
    Select representative months using FS distance on temperature only.
    
    For each calendar month (1-12), selects the year within the period that
    minimizes FS distance against the long-term distribution of that month.
    
    Args:
        daily_projection_data: DataFrame from Module 3 with columns:
            - date: datetime
            - T_proj: daily mean temperature (°C)
            - R_proj: daily mean solar radiation (W/m²) [optional]
            - W_proj: daily mean wind speed (m/s) [optional]
            - max_temp, min_temp: optional temperature columns
        start_year: Start year of projection period
        end_year: End year of projection period
    
    Returns:
        DataFrame with columns:
        - month: calendar month (1-12)
        - selected_year: selected representative year
        - fs_score: FS distance score (lower is better)
        - avg_temp_mean: mean of T_proj for selected month (optional but recommended)
        - min_temp_mean: mean of min_temp if available (optional)
        - max_temp_mean: mean of max_temp if available (optional)
        - solar_radiation_mean: mean of R_proj if available (optional)
        - wind_speed_mean: mean of W_proj if available (optional)
    """
    # Verify required column exists
    if 'T_proj' not in daily_projection_data.columns:
        raise ValueError("daily_projection_data must contain 'T_proj' column")
    
    if 'date' not in daily_projection_data.columns:
        raise ValueError("daily_projection_data must contain 'date' column")
    
    # Ensure date is datetime
    df = daily_projection_data.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract year and month
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # Filter to period of interest
    df = df[(df['year'] >= start_year) & (df['year'] <= end_year)].copy()
    
    if len(df) == 0:
        raise ValueError(f"No data found for period {start_year}-{end_year}")
    
    # Standard quantiles: 0, 1/12, 2/12, ..., 11/12, 1.0
    quantiles = np.linspace(0, 1, N_QUANTILES + 1)
    
    # Process each calendar month
    results = []
    
    for month in range(1, 13):
        # Filter data for this calendar month
        month_data = df[df['month'] == month].copy()
        
        if len(month_data) == 0:
            print(f"⚠️ Warning: No data found for month {month}")
            continue
        
        # Group by year to get all candidate year-months
        candidate_groups = month_data.groupby('year')
        
        # Calculate long-term CDF for this month (pooled across all years)
        all_month_values = month_data['T_proj'].values
        long_term_cdf = calculate_cdf(all_month_values, quantiles)
        
        if np.any(np.isnan(long_term_cdf)):
            print(f"⚠️ Warning: Cannot calculate long-term CDF for month {month} - insufficient data")
            continue
        
        # Calculate FS distance for each candidate year
        fs_scores = []
        candidate_years = []
        
        for year, year_data in candidate_groups:
            year_values = year_data['T_proj'].values
            
            if len(year_values) == 0:
                continue
            
            # Calculate CDF for this candidate year-month
            candidate_cdf = calculate_cdf(year_values, quantiles)
            
            # Skip if CDF calculation failed
            if np.any(np.isnan(candidate_cdf)):
                continue
            
            # Calculate FS distance
            fs_score = calculate_fs_distance(candidate_cdf, long_term_cdf)
            
            fs_scores.append(fs_score)
            candidate_years.append(year)
        
        if len(fs_scores) == 0:
            print(f"⚠️ Warning: No valid candidates found for month {month}")
            continue
        
        # Select year with minimum FS distance (best match)
        min_idx = np.argmin(fs_scores)
        selected_year = candidate_years[min_idx]
        best_fs_score = fs_scores[min_idx]
        
        # Get statistics for selected year-month
        selected_data = month_data[month_data['year'] == selected_year]
        
        # Build result dictionary
        result = {
            'month': month,
            'selected_year': int(selected_year),
            'fs_score': float(best_fs_score),
        }
        
        # Add optional statistics if available
        if 'T_proj' in selected_data.columns:
            result['avg_temp_mean'] = float(selected_data['T_proj'].mean())
        
        if 'min_temp' in selected_data.columns:
            result['min_temp_mean'] = float(selected_data['min_temp'].mean())
        
        if 'max_temp' in selected_data.columns:
            result['max_temp_mean'] = float(selected_data['max_temp'].mean())
        
        if 'R_proj' in selected_data.columns:
            result['solar_radiation_mean'] = float(selected_data['R_proj'].mean())
        
        if 'W_proj' in selected_data.columns:
            result['wind_speed_mean'] = float(selected_data['W_proj'].mean())
        
        results.append(result)
    
    if len(results) == 0:
        raise ValueError("No months could be selected - insufficient data")
    
    # Convert to DataFrame
    result_df = pd.DataFrame(results)
    
    # Ensure month column is integer
    result_df['month'] = result_df['month'].astype(int)
    result_df['selected_year'] = result_df['selected_year'].astype(int)
    
    # Sort by month
    result_df = result_df.sort_values('month').reset_index(drop=True)
    
    print(f"\n✓ FS selection completed: {len(result_df)} months selected")
    print(f"  Year range: {result_df['selected_year'].min()}-{result_df['selected_year'].max()}")
    print(f"  Average FS score: {result_df['fs_score'].mean():.4f}")
    
    return result_df
