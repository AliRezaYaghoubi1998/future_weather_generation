"""
Climate Yearly Profiles Visualization

Plots time series yearly profile for key climate variables using hourly values.
Shows annual patterns for temperature, radiation, wind, and other available variables.

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


def create_yearly_profiles(hourly_data: pd.DataFrame, results_path: Path) -> None:
    """
    Create yearly profile plots for key climate variables.
    
    Plots time series of hourly values for at least temperature, radiation, and wind.
    Shows annual patterns with month labels.
    
    Args:
        hourly_data: DataFrame with hourly weather data (columns: datetime, temp_dry_bulb_C, ...)
        results_path: Path to results directory for saving output
    """
    # Ensure datetime is datetime type
    df = hourly_data.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Sort by datetime
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Identify climate variables (exclude datetime and source columns)
    climate_vars = [col for col in df.columns 
                   if col != 'datetime' and not col.endswith('_source') and not col.endswith('_distance_km')]
    
    # Prioritize key variables: temperature, radiation, wind
    key_vars = ['temp_dry_bulb_C', 'radiation_W_m2', 'wind_speed_m_s']
    other_vars = [v for v in climate_vars if v not in key_vars]
    
    # Order: key variables first, then others
    ordered_vars = [v for v in key_vars if v in climate_vars] + other_vars
    
    if len(ordered_vars) == 0:
        print("Warning: No climate variables found in hourly data")
        return
    
    # Calculate number of subplots
    n_vars = len(ordered_vars)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    # Create figure
    fig = plt.figure(figsize=(20, 6 * n_rows))
    fig.suptitle('Climate Yearly Profiles', fontsize=16, y=0.995)
    
    # Month labels
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Plot each variable
    for idx, var in enumerate(ordered_vars, 1):
        ax = plt.subplot(n_rows, n_cols, idx)
        
        if var not in df.columns:
            ax.text(0.5, 0.5, f'Variable {var} not found', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(var)
            continue
        
        # Get values (handle NaN)
        values = df[var].values
        
        # Plot hourly values
        ax.plot(range(len(df)), values, linewidth=0.5, alpha=0.6, color='blue', label='Hourly')
        
        # Calculate and plot daily moving average (24-hour window)
        if len(values) >= 24:
            daily_avg = pd.Series(values).rolling(window=24, center=True).mean()
            ax.plot(range(len(df)), daily_avg, 'r-', linewidth=2, alpha=0.8, label='Daily Average')
        
        # Set labels and title
        ax.set_title(var.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (Months)', fontsize=10)
        ax.set_ylabel(var.replace('_', ' ').title(), fontsize=10)
        
        # Set x-axis ticks to show months
        if len(df) > 0:
            # Calculate month positions
            df['month'] = df['datetime'].dt.month
            month_positions = []
            month_labels_actual = []
            
            for month in range(1, 13):
                month_mask = df['month'] == month
                if month_mask.any():
                    month_pos = month_mask.idxmax()
                    month_positions.append(month_pos)
                    month_labels_actual.append(month_labels[month - 1])
            
            if month_positions:
                ax.set_xticks(month_positions)
                ax.set_xticklabels(month_labels_actual, rotation=45, ha='right')
        
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    output_file = results_path / "climate_yearly_profiles_complete.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"✓ Created climate yearly profiles: {output_file}")
