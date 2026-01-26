"""
Pre-generated Daily Summary Creator

This script generates pre-processed daily summary files
(daily_summary_MM-DD.pkl) from raw historical meteorological data.

These files are REQUIRED INPUTS for the FTMY pipeline and must be
generated once prior to running FTMY. They are not generated
on-the-fly during FTMY execution.

The generation of these files is considered a preprocessing step
and is therefore kept separate from the main FTMY workflow.

Author: Ali Reza Yaghoubi
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import glob
from pathlib import Path
import pickle
import logging
from tqdm import tqdm

# Constants
DENMARK_BOUNDARIES = {
    "min_lon": 7.5,   # Western boundary
    "max_lon": 15.0,  # Eastern boundary
    "min_lat": 54.5,  # Southern boundary
    "max_lat": 58.0   # Northern boundary
}

VARIABLE_MAPPING = {
    'temp_dry_bulb_C': ['temp_mean_past1h', 'temp_dry'],
    'wind_speed_m_s': ['wind_speed_past1h', 'wind_speed'],
    'wind_direction_degrees': ['wind_dir_past1h', 'wind_dir'],
    'temp_dew_C': ['temp_dew'],
    'humidity_percent': ['humidity_past1h', 'humidity'],
    'pressure_hPa': ['pressure'],
    'precip_past1h_mm': ['precip_past1h'],
    'radiation_W_m2': ['radia_glob_past1h', 'radia_glob'],
    'sun_last1h_glob_min': ['sun_last1h_glob'],
    'cloud_cover_percent': ['cloud_cover'],
    'cloud_height_m': ['cloud_height']
}

class HistoricalDailySummaryCreator:
    def __init__(self, base_dir, output_dir):
        """
        Initialize the creator with source and output directories
        """
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging with more detailed format
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('historical_summary_creation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def is_within_denmark(self, lon, lat):
        try:
            return (DENMARK_BOUNDARIES['min_lon'] <= float(lon) <= DENMARK_BOUNDARIES['max_lon'] and
                    DENMARK_BOUNDARIES['min_lat'] <= float(lat) <= DENMARK_BOUNDARIES['max_lat'])
        except (TypeError, ValueError):
            return False

    def validate_record(self, record):
        """Validate if record has all required fields"""
        if not record:
            return False
        
        try:
            # Check if all required fields exist and are not None
            if not all(key in record for key in ['geometry', 'properties', 'type']):
                return False
            
            if not record['geometry'] or 'coordinates' not in record['geometry']:
                return False
            
            props = record['properties']
            required_props = ['observed', 'stationId', 'parameterId', 'value']
            if not all(prop in props for prop in required_props):
                return False
            
            # Validate coordinates
            coords = record['geometry']['coordinates']
            if not isinstance(coords, (list, tuple)) or len(coords) != 2:
                return False
                
            return True
        except Exception:
            return False

    def process_daily_file(self, file_path):
        """Process a single daily file and return structured data with improved error handling"""
        data = []
        invalid_records = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        line = line.strip()
                        if not line:  # Skip empty lines
                            continue
                            
                        record = json.loads(line)
                        
                        if not self.validate_record(record):
                            invalid_records += 1
                            continue

                        coords = record['geometry']['coordinates']
                        props = record['properties']
                        
                        # Filter locations outside Denmark
                        if not self.is_within_denmark(coords[0], coords[1]):
                            continue

                        # Create observation record
                        obs = {
                            'observed_time': props['observed'],
                            'station_id': props['stationId'],
                            'longitude': coords[0],
                            'latitude': coords[1],
                            'parameter_id': props['parameterId'],
                            'value': float(props['value'])  # Ensure value is numeric
                        }
                        data.append(obs)
                        
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        invalid_records += 1
                        self.logger.debug(f"Error in file {file_path}, line {line_num}: {str(e)}")
                        continue
                        
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            return pd.DataFrame()

        if invalid_records > 0:
            self.logger.warning(f"Found {invalid_records} invalid records in {file_path}")
            
        if not data:
            self.logger.warning(f"No valid data found in {file_path}")
            return pd.DataFrame()
            
        return pd.DataFrame(data)

    def aggregate_daily_data(self, df):
        """Aggregate data by station and parameter with error handling"""
        try:
            # Convert observed_time to datetime
            df['observed_time'] = pd.to_datetime(df['observed_time'])
            df['year'] = df['observed_time'].dt.year

            # Create station metadata
            station_metadata = df.groupby('station_id').agg({
                'longitude': 'first',
                'latitude': 'first'
            }).to_dict('index')

            # Aggregate values by station, parameter, and year
            daily_stats = df.groupby(['station_id', 'parameter_id', 'year'])['value'].agg([
                'mean', 'min', 'max', 'std', 'count'
            ]).reset_index()

            return station_metadata, daily_stats
            
        except Exception as e:
            self.logger.error(f"Error in data aggregation: {str(e)}")
            return {}, pd.DataFrame()

    def create_daily_summaries(self):
        """Create summary files for each day of the year with improved file handling"""
        total_processed = 0
        total_errors = 0
        files_found = 0
        
        # Process each year directory
        year_dirs = [d for d in self.base_dir.glob('[0-9]*') if d.is_dir()]
        self.logger.info(f"Found {len(year_dirs)} year directories")
        
        for year_dir in sorted(year_dirs):
            try:
                year = int(year_dir.name)
                
                # Get all day files for the year using the correct path pattern
                day_files = list(year_dir.glob(f'{year}-*.txt'))
                files_found += len(day_files)
                
                self.logger.info(f"Processing year {year} - Found {len(day_files)} daily files")
                
                if len(day_files) == 0:
                    self.logger.warning(f"No daily files found in directory: {year_dir}")
                    continue

                # Process each day file with progress bar
                for day_file in tqdm(sorted(day_files), desc=f"Processing {year}", unit="files"):
                    try:
                        # Verify file exists and is readable
                        if not day_file.is_file():
                            self.logger.error(f"File not found or not accessible: {day_file}")
                            total_errors += 1
                            continue

                        # Extract date from filename (e.g., "2018-12-21" from "2018-12-21.txt")
                        date_str = day_file.stem
                        try:
                            # Verify the date format
                            full_date = datetime.strptime(date_str, '%Y-%m-%d')
                            month_day = full_date.strftime('%m-%d')
                        except ValueError as e:
                            self.logger.error(f"Invalid date format in filename: {day_file}")
                            total_errors += 1
                            continue

                        # Process the daily file
                        df = self.process_daily_file(day_file)
                        if df.empty:
                            self.logger.debug(f"No valid data in file: {day_file}")
                            continue

                        # Aggregate daily data
                        station_metadata, daily_stats = self.aggregate_daily_data(df)
                        if daily_stats.empty:
                            continue

                        # Create or update the summary file
                        summary_file = self.output_dir / f"daily_summary_{month_day}.pkl"
                        
                        try:
                            if summary_file.exists():
                                with open(summary_file, 'rb') as f:
                                    summary_data = pickle.load(f)
                            else:
                                summary_data = {
                                    'station_metadata': {},
                                    'yearly_stats': []
                                }

                            # Update summary data
                            summary_data['station_metadata'].update(station_metadata)
                            
                            # Check if data for this year already exists
                            year_exists = False
                            for stats in summary_data['yearly_stats']:
                                if stats['year'] == year:
                                    stats['stats'] = daily_stats.to_dict('records')
                                    year_exists = True
                                    break
                            
                            if not year_exists:
                                summary_data['yearly_stats'].append({
                                    'year': year,
                                    'stats': daily_stats.to_dict('records')
                                })

                            # Save updated summary
                            with open(summary_file, 'wb') as f:
                                pickle.dump(summary_data, f)
                                
                            total_processed += 1
                            
                        except Exception as e:
                            self.logger.error(f"Error processing summary file {summary_file}: {str(e)}")
                            total_errors += 1
                            continue
                            
                    except Exception as e:
                        total_errors += 1
                        self.logger.error(f"Error processing file {day_file}: {str(e)}")
                        continue
                        
            except Exception as e:
                self.logger.error(f"Error processing year {year_dir}: {str(e)}")
                continue

        self.logger.info(f"\nFile Processing Statistics:")
        self.logger.info(f"Total files found: {files_found}")
        self.logger.info(f"Total files processed successfully: {total_processed}")
        self.logger.info(f"Total errors: {total_errors}")
        
        return total_processed, total_errors

    def run(self):
        """Run the entire process with summary statistics"""
        start_time = datetime.now()
        self.logger.info(f"Starting historical daily summary creation at {start_time}")
        
        try:
            total_processed, total_errors = self.create_daily_summaries()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.logger.info("\nProcessing Summary:")
            self.logger.info(f"Total files processed: {total_processed}")
            self.logger.info(f"Total errors encountered: {total_errors}")
            self.logger.info(f"Processing duration: {duration}")
            self.logger.info(f"Successfully completed summary creation at {end_time}")
            
        except Exception as e:
            self.logger.error(f"Critical error during processing: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    # The base_dir should modify based on the downloaded historical data files
    base_dir = "M:/SyntheticWeatherFile"
    # Output directory for pre-generated daily summary files
    # This directory must be referenced by FTMY_DAILY_SUMMARIES_PATH
    output_dir = "M:/SyntheticWeatherFile/historical_daily_climate_variables"

    
    creator = HistoricalDailySummaryCreator(base_dir, output_dir)
    creator.run()