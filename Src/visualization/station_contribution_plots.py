"""
Station Contribution Plots Visualization

Shows contribution per station using source tracking data.
Displays which stations contributed data and in what proportion.

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


def create_contribution_plots(
    hourly_sources: pd.DataFrame,
    stations: pd.DataFrame,
    results_path: Path
) -> None:
    """
    Create station contribution plots showing which stations provided data.
    
    Uses synthetic_hourly_sources.csv and station_distances.csv to show
    contribution per station (pie or bar chart).
    
    Args:
        hourly_sources: DataFrame with source tracking (columns: datetime, <var>_source_station_id, ...)
        stations: DataFrame with station information (columns: station_id, distance_km, ...)
        results_path: Path to results directory for saving output
    """
    # Ensure datetime is datetime type
    sources_df = hourly_sources.copy()
    sources_df['datetime'] = pd.to_datetime(sources_df['datetime'])
    
    # Find all source columns
    source_columns = [col for col in sources_df.columns 
                     if col.endswith('_source_station_id') and col != 'datetime']
    
    if len(source_columns) == 0:
        print("Warning: No source columns found in hourly_sources")
        return
    
    # Count contributions per station across all variables
    station_counts = {}
    station_distances = dict(zip(stations['station_id'], stations['distance_km']))
    
    for col in source_columns:
        # Count occurrences of each station (excluding None and "Interpolation")
        station_series = sources_df[col]
        valid_sources = station_series[
            (station_series.notna()) & 
            (station_series != "Interpolation") & 
            (station_series != "None")
        ]
        
        for station_id in valid_sources:
            if station_id not in station_counts:
                station_counts[station_id] = 0
            station_counts[station_id] += 1
    
    if len(station_counts) == 0:
        print("Warning: No station contributions found")
        return
    
    # Create DataFrame for plotting
    contribution_data = []
    for station_id, count in station_counts.items():
        distance_km = station_distances.get(station_id, np.nan)
        contribution_data.append({
            'station_id': station_id,
            'count': count,
            'distance_km': distance_km,
            'percentage': (count / len(sources_df) / len(source_columns)) * 100
        })
    
    contrib_df = pd.DataFrame(contribution_data)
    contrib_df = contrib_df.sort_values('count', ascending=False)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Station Contribution Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Pie chart of contributions
    # Show top stations (limit to top 10 for readability)
    top_n = min(10, len(contrib_df))
    top_contrib = contrib_df.head(top_n)
    other_count = contrib_df.iloc[top_n:]['count'].sum() if len(contrib_df) > top_n else 0
    
    # Prepare pie chart data
    pie_labels = [f"Station {int(sid)}\n({d:.2f} km)" for sid, d in 
                  zip(top_contrib['station_id'], top_contrib['distance_km'])]
    pie_values = top_contrib['count'].values
    
    if other_count > 0:
        pie_labels.append(f"Other\n({len(contrib_df) - top_n} stations)")
        pie_values = np.append(pie_values, other_count)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(pie_values)))
    ax1.pie(pie_values, labels=pie_labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax1.set_title('Station Contribution (Top Stations)', fontsize=12, fontweight='bold')
    
    # Plot 2: Bar chart sorted by contribution
    # Show top 15 stations
    top_n_bar = min(15, len(contrib_df))
    top_contrib_bar = contrib_df.head(top_n_bar)
    
    bar_labels = [f"Station {int(sid)}" for sid in top_contrib_bar['station_id']]
    bar_values = top_contrib_bar['count'].values
    bar_colors = plt.cm.viridis(top_contrib_bar['distance_km'] / top_contrib_bar['distance_km'].max())
    
    bars = ax2.barh(range(len(bar_labels)), bar_values, color=bar_colors)
    ax2.set_yticks(range(len(bar_labels)))
    ax2.set_yticklabels(bar_labels, fontsize=9)
    ax2.set_xlabel('Number of Hourly Records', fontsize=10)
    ax2.set_title('Station Contribution (Top 15, Colored by Distance)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add distance annotations
    for i, (idx, row) in enumerate(top_contrib_bar.iterrows()):
        ax2.text(row['count'], i, f" {row['distance_km']:.2f} km", 
                va='center', fontsize=8)
    
    # Invert y-axis to show highest contribution at top
    ax2.invert_yaxis()
    
    plt.tight_layout()
    
    # Save figure
    output_file = results_path / "station_contribution_plots_complete.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"✓ Created station contribution plots: {output_file}")
    print(f"  Total stations contributing: {len(contrib_df)}")
    print(f"  Top station: Station {int(contrib_df.iloc[0]['station_id'])} "
          f"({contrib_df.iloc[0]['count']} records, {contrib_df.iloc[0]['distance_km']:.2f} km)")
