"""
EPW Writer

Converts final hourly weather data to EnergyPlus Weather (EPW) format.
Generates EPW file with proper header and data rows.

Reference behavior: references/generate_epw.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Add Src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import PipelineConfig


def build_epw_metadata(
    latitude: float,
    longitude: float,
    scenario: str,
    gcm: str,
    rcm: str,
    year: int
) -> dict:
    """
    Build metadata dictionary for EPW header.
    
    Args:
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
        scenario: Climate scenario (e.g., "rcp85")
        gcm: Global Climate Model name (e.g., "ICHEC-EC-EARTH")
        rcm: Regional Climate Model name (e.g., "KNMI-RACMO22E")
        year: Target year
        
    Returns:
        Metadata dictionary for EPW header
    """
    # Always Denmark for this use case
    country = "Denmark"
    
    # Denmark timezone (UTC+1 for CET)
    timezone = 1.0
    
    # Use deterministic city name based on coordinates (no external API calls)
    city = f"Location_{latitude:.3f}_{longitude:.3f}"
    
    # Use deterministic elevation (default for Denmark, typical range 0-200m)
    # For Denmark, most locations are near sea level
    elevation_m = 50.0  # Default elevation for Denmark (deterministic, no API calls)
    
    # Generate comments
    comments1 = (
        f"Synthetic Weather | {city}, Denmark | "
        f"Lat: {latitude:.4f}, Lon: {longitude:.4f} | "
        f"Scenario: {scenario}, GCM: {gcm}, RCM: {rcm}, Year: {year}"
    )
    
    return {
        'city': city,
        'country': country,
        'latitude': latitude,
        'longitude': longitude,
        'timezone': timezone,
        'elevation_m': elevation_m,
        'comments1': comments1,
        'scenario': scenario,
        'gcm': gcm,
        'rcm': rcm,
        'year': year
    }


def parse_datetime_column(df: pd.DataFrame, datetime_column: str = "datetime") -> pd.Series:
    """
    Parse datetime column using strict ISO 8601 format.
    
    Args:
        df: DataFrame with datetime column
        datetime_column: Name of datetime column
        
    Returns:
        Parsed datetime Series
    """
    if datetime_column not in df.columns:
        raise ValueError(f"Datetime column '{datetime_column}' not found")
    
    datetime_series = df[datetime_column]
    try:
        return pd.to_datetime(datetime_series, format='%Y-%m-%d %H:%M:%S', errors='raise')
    except Exception as e:
        example = datetime_series.dropna().astype(str).head(1).tolist()
        example_value = example[0] if example else "N/A"
        raise ValueError(
            f"Datetime parsing failed. Expected format '%Y-%m-%d %H:%M:%S'. "
            f"Example value: {example_value}"
        ) from e


def _compute_dni_dhi_erbs(
    ghi: pd.Series,
    dt: pd.Series,
    latitude: float,
    longitude: float,
    timezone: float
) -> tuple:
    """
    Compute DNI and DHI from GHI using the Erbs diffuse fraction model.

    GHI is the reconstructed weather series input.
    DNI/DHI are derived estimates for EPW completeness.
    """
    n = len(ghi)
    lat_rad = np.radians(latitude)
    ghi_arr = ghi.values.astype(float)

    day_of_year = dt.dt.dayofyear.values.astype(float)
    hour_of_day = dt.dt.hour.values.astype(float) + dt.dt.minute.values.astype(float) / 60.0

    declination = np.radians(
        23.45 * np.sin(np.radians(360.0 / 365.0 * (284.0 + day_of_year)))
    )

    B_rad = 2.0 * np.pi * (day_of_year - 1.0) / 365.0
    eot = 229.18 * (
        0.000075
        + 0.001868 * np.cos(B_rad)
        - 0.032077 * np.sin(B_rad)
        - 0.014615 * np.cos(2.0 * B_rad)
        - 0.04089 * np.sin(2.0 * B_rad)
    )

    lstm = 15.0 * timezone
    time_correction = 4.0 * (longitude - lstm) + eot
    solar_time = hour_of_day + time_correction / 60.0

    hour_angle = np.radians(15.0 * (solar_time - 12.0))
    cos_zenith = (
        np.sin(lat_rad) * np.sin(declination)
        + np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle)
    )
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)

    i_sc = 1367.0
    eccentricity = 1.0 + 0.033 * np.cos(2.0 * np.pi * day_of_year / 365.0)
    i0h = i_sc * eccentricity * cos_zenith

    cos_zenith_min = 0.01
    sun_up = (cos_zenith > cos_zenith_min) & (i0h > 0)

    kt = np.where(sun_up, ghi_arr / np.maximum(i0h, 1e-6), 0.0)
    kt = np.clip(kt, 0.0, 1.0)

    kd = np.zeros(n)
    mask_low = sun_up & (kt <= 0.22)
    kd[mask_low] = 1.0 - 0.09 * kt[mask_low]

    mask_mid = sun_up & (kt > 0.22) & (kt <= 0.80)
    kt_mid = kt[mask_mid]
    kd[mask_mid] = (
        0.9511
        - 0.1604 * kt_mid
        + 4.388 * kt_mid ** 2
        - 16.638 * kt_mid ** 3
        + 12.336 * kt_mid ** 4
    )

    mask_high = sun_up & (kt > 0.80)
    kd[mask_high] = 0.165

    dhi_arr = np.where(sun_up, ghi_arr * kd, 0.0)
    dhi_arr = np.clip(dhi_arr, 0.0, None)
    dni_arr = np.where(
        sun_up & (cos_zenith > cos_zenith_min),
        (ghi_arr - dhi_arr) / cos_zenith,
        0.0,
    )
    dni_arr = np.clip(dni_arr, 0.0, None)

    night = (~sun_up) | (ghi_arr <= 0)
    dni_arr[night] = 0.0
    dhi_arr[night] = 0.0

    dni_arr = np.nan_to_num(dni_arr, nan=0.0)
    dhi_arr = np.nan_to_num(dhi_arr, nan=0.0)

    return pd.Series(dni_arr, index=ghi.index), pd.Series(dhi_arr, index=ghi.index)


def _collect_timestamp_diagnostics(dt: pd.Series, year: int) -> str:
    """Build detailed diagnostics for missing/duplicate hourly timestamps."""
    expected = pd.date_range(f"{year}-01-01 00:00:00", f"{year}-12-31 23:00:00", freq="h")
    dt_norm = dt.dt.floor("h")
    dup_count = int(dt_norm.duplicated().sum())
    missing = expected.difference(pd.DatetimeIndex(dt_norm))
    extra = pd.DatetimeIndex(dt_norm).difference(expected)
    parts = [
        f"duplicates_detected={dup_count}",
        f"missing_hours={len(missing)}",
        f"extra_hours={len(extra)}",
    ]
    if len(missing) > 0:
        parts.append(f"missing_first={missing[0]}")
        parts.append(f"missing_last={missing[-1]}")
    if len(extra) > 0:
        parts.append(f"extra_first={extra[0]}")
        parts.append(f"extra_last={extra[-1]}")
    return ", ".join(parts)


def csv_to_epw(
    hourly_data: pd.DataFrame,
    metadata: dict,
    output_epw: Path
) -> None:
    """
    Convert hourly weather DataFrame to EPW format.
    
    Args:
        hourly_data: DataFrame with hourly weather data
        metadata: Location metadata dictionary
        output_epw: Path to output EPW file

    EPW radiation fields:
    - Global Horizontal Radiation (GHI): directly from reconstructed `radiation_W_m2`
    - Direct Normal Radiation (DNI): Erbs-derived estimate when not provided
    - Diffuse Horizontal Radiation (DHI): Erbs-derived estimate when not provided
    """
    df = hourly_data.copy()
    
    # Parse datetime column
    dt = parse_datetime_column(df, "datetime")
    
    # Validate datetime parsing
    if dt.isna().all():
        raise ValueError("All datetime values are invalid. Please check the datetime column format.")
    
    # Structural EPW validation/enforcement: non-leap 8760 rows, consistent year.
    feb29_mask = (dt.dt.month == 2) & (dt.dt.day == 29)
    feb29_removed_count = int(feb29_mask.sum())
    df = df.loc[~feb29_mask].copy()
    dt = dt.loc[~feb29_mask].reset_index(drop=True)
    df = df.reset_index(drop=True)

    target_year = metadata["year"]
    df = df.sort_values("datetime").reset_index(drop=True)
    dt = parse_datetime_column(df, "datetime")

    if len(df) != 8760:
        diagnostics = _collect_timestamp_diagnostics(dt, target_year)
        raise ValueError(
            "EPW structural validation failed: expected 8760 rows. "
            f"row_count={len(df)}, feb29_removed={feb29_removed_count}, {diagnostics}"
        )

    # Date/time fields (EPW uses 1..24 hours; minute 60 for end-of-hour)
    year = pd.Series([target_year] * len(df))
    month = dt.dt.month
    day = dt.dt.day
    hour = (dt.dt.hour + 1).clip(1, 24)  # 1..24
    minute = 60
    data_src = 0  # Data Source and Uncertainty (0 default)
    
    # Map CSV columns to EPW core fields
    # Temperature and humidity
    dry_bulb = df.get("temp_dry_bulb_C", pd.Series([99.9] * len(df))).fillna(99.9)
    dew_point = df.get("temp_dew_C", pd.Series([99.9] * len(df))).fillna(99.9)
    rh = df.get("humidity_percent", pd.Series([999] * len(df))).fillna(999)
    
    # Pressure: convert hPa → Pa, EPW expects Pa
    # Sanitize: reject invalid/zero values, then fill with EPW missing placeholder
    if "pressure_hPa" in df.columns:
        p = df["pressure_hPa"].astype(float)
        # Convert to Pa
        p_pa = p * 100.0
        
        # Sanitize: reject invalid values (NaN, zero, negative, or unrealistic)
        # Typical sea-level pressure: 95000-105000 Pa (950-1050 hPa)
        # Reject values outside reasonable range (80000-110000 Pa)
        valid_mask = (
            p_pa.notna() & 
            (p_pa > 80000) & 
            (p_pa < 110000)
        )
        
        # Fill invalid values with EPW missing placeholder
        station_pressure = p_pa.where(valid_mask, 999999)
    else:
        station_pressure = pd.Series([999999] * len(df))
    
    # Radiation: GHI from reconstructed series; DNI/DHI derived estimates for EPW completeness.
    ghi = df.get("radiation_W_m2", pd.Series([0] * len(df))).fillna(0)  # 0 is valid at night
    has_valid_dni = (
        "DNI" in df.columns and df["DNI"].notna().any() and (df["DNI"] != 9999).any()
    )
    has_valid_dhi = (
        "DHI" in df.columns and df["DHI"].notna().any() and (df["DHI"] != 9999).any()
    )
    if has_valid_dni and has_valid_dhi:
        dni = df["DNI"].fillna(0).clip(lower=0)
        dhi = df["DHI"].fillna(0).clip(lower=0)
    else:
        dni, dhi = _compute_dni_dhi_erbs(
            ghi, dt, metadata["latitude"], metadata["longitude"], metadata["timezone"]
        )
        print("  ✓ DNI/DHI derived with Erbs model (GHI is from reconstructed series)")
    
    # Illum/longwave extras (missing by design)
    g_h_illum = pd.Series([999999] * len(df))
    dn_illum = pd.Series([999999] * len(df))
    dh_illum = pd.Series([999999] * len(df))
    zen_lum = pd.Series([9999] * len(df))
    ir_horiz = pd.Series([9999] * len(df))
    etr_horiz = pd.Series([0] * len(df))
    etr_norm = pd.Series([0] * len(df))
    
    # Wind
    wind_dir = df.get("wind_direction_degrees", pd.Series([999] * len(df))).fillna(999)
    wind_spd = df.get("wind_speed_m_s", pd.Series([999] * len(df))).fillna(999)
    
    # Sky cover: % → tenths [0..10]
    if "cloud_cover_percent" in df.columns:
        total_sky = (df["cloud_cover_percent"].astype(float).fillna(0) / 10.0).round().clip(0, 10).astype(int)
    else:
        total_sky = pd.Series([0] * len(df))
    opaque_sky = total_sky.copy()
    
    # Visibility/ceiling/present weather (missing)
    visibility = pd.Series([9999] * len(df))
    ceiling_h = pd.Series([99999] * len(df))
    present_obs = pd.Series([0] * len(df))
    present_codes = pd.Series([0] * len(df))
    
    # Water/optical/snow/albedo (missing)
    precipitable_water = pd.Series([999] * len(df))
    aerosol_optical_depth = pd.Series([999] * len(df))
    snow_depth = pd.Series([999] * len(df))
    days_since_snow = pd.Series([99] * len(df))
    albedo = pd.Series([999] * len(df))
    
    # Precipitation: depth (mm), quantity (hr)
    precip_mm = df.get("precip_past1h_mm", pd.Series([0] * len(df))).fillna(0).clip(lower=0)
    precip_hr = (precip_mm > 0).astype(int)
    
    # Create output directory if it doesn't exist
    output_epw.parent.mkdir(parents=True, exist_ok=True)
    
    # Write EPW file
    print(f"Writing EPW file: {output_epw}")
    with open(output_epw, "w", newline="", encoding='utf-8') as f:
        # Header section
        header = [
            # LOCATION,<city>,<state>,<country>,<source>,<WMO>,<lat>,<lon>,<tz>,<elev>
            f"LOCATION,{metadata['city']},,{metadata['country']},Synthetic-CSV,,{metadata['latitude']:.6f},{metadata['longitude']:.6f},{metadata['timezone']},{metadata['elevation_m']:.1f}",
            "DESIGN CONDITIONS,0",
            "TYPICAL/EXTREME PERIODS,0",
            "GROUND TEMPERATURES,0",
            "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0",
            f"COMMENTS 1,{metadata['comments1']}",
            "COMMENTS 2,CSV to EPW conversion for synthetic weather data (FSY pipeline)",
            (
                f"DATA PERIODS,1,1,Data,{datetime(target_year, 1, 1).strftime('%A')},"
                "1/1,12/31"
            ),
        ]
        
        for line in header:
            f.write(line + "\n")
        
        # Data rows
        for i in range(len(df)):
            row = [
                year.iat[i], month.iat[i], day.iat[i], hour.iat[i], minute, data_src,
                dry_bulb.iat[i], dew_point.iat[i], rh.iat[i], station_pressure.iat[i],
                etr_horiz.iat[i], etr_norm.iat[i], ir_horiz.iat[i], ghi.iat[i], dni.iat[i], dhi.iat[i],
                g_h_illum.iat[i], dn_illum.iat[i], dh_illum.iat[i], zen_lum.iat[i],
                wind_dir.iat[i], wind_spd.iat[i], total_sky.iat[i], opaque_sky.iat[i],
                visibility.iat[i], ceiling_h.iat[i], present_obs.iat[i], present_codes.iat[i],
                precipitable_water.iat[i], aerosol_optical_depth.iat[i], snow_depth.iat[i], days_since_snow.iat[i],
                albedo.iat[i], precip_mm.iat[i], precip_hr.iat[i]
            ]
            f.write(",".join(str(v) for v in row) + "\n")
    
    print(f"Successfully created EPW file: {output_epw}")
    print(f"Data period: {dt.iloc[0].strftime('%Y-%m-%d')} to {dt.iloc[-1].strftime('%Y-%m-%d')}")
    print(f"Total records: {len(df)}")
    print("✓ EPW structural validation passed (8760 rows, non-leap, consistent year)")


def generate_epw_file(
    hourly_data: pd.DataFrame,
    lon: float,
    lat: float,
    scenario: str,
    gcm: str,
    rcm: str,
    year: int,
    results_path: Path,
    legacy_ftmy_copy: bool = False
) -> Path:
    """
    Generate EPW file from hourly weather data.
    
    Args:
        hourly_data: DataFrame with hourly weather data (columns: datetime, temp_dry_bulb_C, ...)
        lon: Longitude (decimal degrees)
        lat: Latitude (decimal degrees)
        scenario: Climate scenario (e.g., "rcp85")
        gcm: Global Climate Model name (e.g., "ICHEC-EC-EARTH")
        rcm: Regional Climate Model name (e.g., "KNMI-RACMO22E")
        year: Target year
        results_path: Path to results directory
        
    Returns:
        Path to generated EPW file
    """
    # Build metadata
    metadata = build_epw_metadata(lat, lon, scenario, gcm, rcm, year)
    
    # Generate EPW filename: FSY_<lat>_<lon>_<scenario>_<gcm>_<rcm>_<year>.epw
    # Clean up scenario/GCM/RCM names for filename (remove special characters)
    scenario_clean = scenario.replace('/', '_').replace('\\', '_')
    gcm_clean = gcm.replace('/', '_').replace('\\', '_').replace(' ', '_')
    rcm_clean = rcm.replace('/', '_').replace('\\', '_').replace(' ', '_')
    
    epw_filename = f"FSY_{lat:.4f}_{lon:.4f}_{scenario_clean}_{gcm_clean}_{rcm_clean}_{year}.epw"
    output_epw = results_path / epw_filename
    
    # Convert to EPW
    csv_to_epw(hourly_data, metadata, output_epw)

    # Optional legacy filename copy for backward compatibility.
    if legacy_ftmy_copy:
        legacy_filename = f"FTMY_{lat:.4f}_{lon:.4f}_{scenario_clean}_{gcm_clean}_{rcm_clean}_{year}.epw"
        legacy_output = results_path / legacy_filename
        legacy_output.write_text(output_epw.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"✓ Legacy compatibility EPW copy created: {legacy_output}")
    
    return output_epw
