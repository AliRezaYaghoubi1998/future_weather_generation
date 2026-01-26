"""
Temperature Comparison Visualization

Compares projected daily temperature (from Module 3) with synthetic daily
average temperature (from final hourly data).

Reference behavior: references/visualization.py (but heavily pruned)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Add Src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import PipelineConfig


def create_temperature_comparison(
    daily_projection_data: pd.DataFrame,
    hourly_data: pd.DataFrame,
    results_path: Path
) -> None:
    """
    Create temperature comparison plot.
    
    Compares:
    - Projected daily temperature (from Module 3: T_proj)
    - Synthetic daily average temperature (from final hourly: temp_dry_bulb_C)
    
    Args:
        daily_projection_data: DataFrame with projected daily data (columns: date, T_proj, ...)
        hourly_data: DataFrame with hourly weather data (columns: datetime, temp_dry_bulb_C, ...)
        results_path: Path to results directory for saving output
    """
    # Ensure datetime/date columns are datetime type
    proj_df = daily_projection_data.copy()
    hourly_df = hourly_data.copy()
    
    proj_df['date'] = pd.to_datetime(proj_df['date'])
    hourly_df['datetime'] = pd.to_datetime(hourly_df['datetime'])
    
    # Calculate daily averages from hourly data
    hourly_df['date'] = hourly_df['datetime'].dt.date
    daily_avg_temp = hourly_df.groupby('date')['temp_dry_bulb_C'].mean().reset_index()
    daily_avg_temp['date'] = pd.to_datetime(daily_avg_temp['date'])
    
    # Merge projection and synthetic data on date
    comparison_df = proj_df[['date', 'T_proj']].copy()
    comparison_df = comparison_df.merge(
        daily_avg_temp[['date', 'temp_dry_bulb_C']],
        on='date',
        how='inner'
    )
    comparison_df = comparison_df.rename(columns={
        'T_proj': 'projected_daily_temp',
        'temp_dry_bulb_C': 'synthetic_daily_avg_temp'
    })
    
    if len(comparison_df) == 0:
        print("Warning: No overlapping dates found for temperature comparison")
        return
    
    # Sort by date
    comparison_df = comparison_df.sort_values('date').reset_index(drop=True)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Temperature Comparison: Projected vs Synthetic', fontsize=16, fontweight='bold')
    
    # Plot 1: Time series comparison
    ax1.plot(comparison_df['date'], comparison_df['projected_daily_temp'], 
            'b-', linewidth=2, label='Projected Daily (T_proj)', alpha=0.7)
    ax1.plot(comparison_df['date'], comparison_df['synthetic_daily_avg_temp'], 
            'r-', linewidth=2, label='Synthetic Daily Average', alpha=0.7)
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Temperature (°C)', fontsize=11)
    ax1.set_title('Daily Temperature Time Series Comparison', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Scatter plot with 1:1 line
    ax2.scatter(comparison_df['projected_daily_temp'], 
               comparison_df['synthetic_daily_avg_temp'],
               alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    
    # Add 1:1 line
    min_temp = min(comparison_df['projected_daily_temp'].min(), 
                  comparison_df['synthetic_daily_avg_temp'].min())
    max_temp = max(comparison_df['projected_daily_temp'].max(), 
                  comparison_df['synthetic_daily_avg_temp'].max())
    ax2.plot([min_temp, max_temp], [min_temp, max_temp], 
            'k--', linewidth=2, label='1:1 Line', alpha=0.5)
    
    ax2.set_xlabel('Projected Daily Temperature (°C)', fontsize=11)
    ax2.set_ylabel('Synthetic Daily Average Temperature (°C)', fontsize=11)
    ax2.set_title('Projected vs Synthetic Temperature (Scatter)', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Calculate and display statistics
    correlation = comparison_df['projected_daily_temp'].corr(comparison_df['synthetic_daily_avg_temp'])
    mean_diff = (comparison_df['synthetic_daily_avg_temp'] - comparison_df['projected_daily_temp']).mean()
    rmse = np.sqrt(((comparison_df['synthetic_daily_avg_temp'] - comparison_df['projected_daily_temp'])**2).mean())
    
    stats_text = f"Correlation: {correlation:.3f}\nMean Difference: {mean_diff:.2f}°C\nRMSE: {rmse:.2f}°C"
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    output_file = results_path / "temperature_comparison.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"✓ Created temperature comparison: {output_file}")
    print(f"  Correlation: {correlation:.3f}")
    print(f"  Mean difference: {mean_diff:.2f}°C")
    print(f"  RMSE: {rmse:.2f}°C")
