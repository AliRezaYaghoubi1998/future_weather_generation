"""
Projection Data Extractor

Reads NetCDF projection files and extracts daily projection features (T, R, W)
for the target location using nearest grid point selection.

Reference behavior: references/Prediction_weather_Creator.py (reading logic & unit conversion)
"""

import os
import glob
import pandas as pd
import xarray as xr
import numpy as np
from typing import Dict, List, Tuple, Optional
from pyproj import Transformer, CRS


def get_utm_zone_epsg(lon: float, lat: float) -> str:
    """
    Determine the appropriate UTM zone EPSG code based on longitude and latitude.
    
    UTM zones:
    - Northern hemisphere (lat >= 0): EPSG:326xx (xx = zone number, 01-60)
    - Southern hemisphere (lat < 0): EPSG:327xx (xx = zone number, 01-60)
    
    Zone number = floor((lon + 180) / 6) + 1
    
    Args:
        lon: Longitude in decimal degrees
        lat: Latitude in decimal degrees
        
    Returns:
        EPSG code string (e.g., "EPSG:32632" for UTM Zone 32N)
    """
    # Calculate UTM zone number (1-60)
    zone_number = int((lon + 180) / 6) + 1
    
    # Clamp to valid range
    zone_number = max(1, min(60, zone_number))
    
    # Determine hemisphere
    if lat >= 0:
        # Northern hemisphere
        epsg_code = f"EPSG:326{zone_number:02d}"
    else:
        # Southern hemisphere
        epsg_code = f"EPSG:327{zone_number:02d}"
    
    return epsg_code


def calculate_utm_distance(
    lon1: float, lat1: float,
    lon2: float, lat2: float,
    utm_epsg: str
) -> float:
    """
    Calculate distance between two points using UTM coordinates (in meters).
    
    Args:
        lon1, lat1: First point coordinates (decimal degrees)
        lon2, lat2: Second point coordinates (decimal degrees)
        utm_epsg: UTM EPSG code (e.g., "EPSG:32632")
        
    Returns:
        Distance in kilometers
    """
    transformer = Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)
    
    # Convert both points to UTM
    easting1, northing1 = transformer.transform(lon1, lat1)
    easting2, northing2 = transformer.transform(lon2, lat2)
    
    # Calculate Euclidean distance in meters
    distance_m = np.sqrt((easting2 - easting1)**2 + (northing2 - northing1)**2)
    
    # Convert to kilometers
    distance_km = distance_m / 1000.0
    
    return distance_km


def calculate_geographic_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate great-circle distance between two points using Haversine formula.
    
    Args:
        lon1, lat1: First point coordinates (decimal degrees)
        lon2, lat2: Second point coordinates (decimal degrees)
        
    Returns:
        Distance in kilometers
    """
    # Earth radius in kilometers
    R = 6371.0
    
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    # Haversine formula
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance_km = R * c
    
    return distance_km


def detect_grid_type(ds: xr.Dataset) -> Tuple[str, Optional[Dict]]:
    """
    Detect the grid type of a NetCDF dataset.
    
    Checks for:
    1. Rotated pole grids (rotated_latitude_longitude)
    2. Projected x/y coordinates (e.g., UTM, Lambert)
    3. Geographic lat/lon grids (fallback)
    
    Args:
        ds: xarray Dataset to inspect
        
    Returns:
        Tuple of (grid_type, grid_info)
        - grid_type: 'rotated_pole', 'projected', or 'geographic'
        - grid_info: Dictionary with grid-specific information
    """
    grid_info = {}
    
    # Check for rotated pole grid
    # Look for rotated_pole variable or grid_mapping attribute
    rotated_pole_var = None
    for var_name in ds.variables:
        var = ds.variables[var_name]
        if hasattr(var, 'grid_mapping'):
            grid_mapping_name = var.grid_mapping
            if grid_mapping_name in ds.variables:
                grid_mapping_var = ds.variables[grid_mapping_name]
                if hasattr(grid_mapping_var, 'grid_mapping_name'):
                    if grid_mapping_var.grid_mapping_name == 'rotated_latitude_longitude':
                        rotated_pole_var = grid_mapping_var
                        break
                # Also check for rotated_pole in variable name
                if 'rotated' in str(grid_mapping_name).lower() or 'pole' in str(grid_mapping_name).lower():
                    rotated_pole_var = grid_mapping_var
                    break
    
    # Also check for rlon/rlat coordinates (rotated longitude/latitude)
    has_rlon = 'rlon' in ds.coords or 'rlon' in ds.dims
    has_rlat = 'rlat' in ds.coords or 'rlat' in ds.dims
    
    if rotated_pole_var is not None or (has_rlon and has_rlat):
        print("  Detected rotated pole grid")
        if rotated_pole_var is not None:
            # Extract rotated pole parameters from grid_mapping variable
            if hasattr(rotated_pole_var, 'grid_north_pole_latitude'):
                grid_info['pole_lat'] = float(rotated_pole_var.grid_north_pole_latitude)
            elif hasattr(rotated_pole_var, 'attrs') and 'grid_north_pole_latitude' in rotated_pole_var.attrs:
                grid_info['pole_lat'] = float(rotated_pole_var.attrs['grid_north_pole_latitude'])
            
            if hasattr(rotated_pole_var, 'grid_north_pole_longitude'):
                grid_info['pole_lon'] = float(rotated_pole_var.grid_north_pole_longitude)
            elif hasattr(rotated_pole_var, 'attrs') and 'grid_north_pole_longitude' in rotated_pole_var.attrs:
                grid_info['pole_lon'] = float(rotated_pole_var.attrs['grid_north_pole_longitude'])
            
            if hasattr(rotated_pole_var, 'north_pole_grid_longitude'):
                grid_info['north_pole_grid_lon'] = float(rotated_pole_var.north_pole_grid_longitude)
            elif hasattr(rotated_pole_var, 'attrs') and 'north_pole_grid_longitude' in rotated_pole_var.attrs:
                grid_info['north_pole_grid_lon'] = float(rotated_pole_var.attrs['north_pole_grid_longitude'])
        
        # Also check dataset-level attributes as fallback
        if 'pole_lat' not in grid_info and hasattr(ds, 'attrs'):
            if 'grid_north_pole_latitude' in ds.attrs:
                grid_info['pole_lat'] = float(ds.attrs['grid_north_pole_latitude'])
            if 'grid_north_pole_longitude' in ds.attrs:
                grid_info['pole_lon'] = float(ds.attrs['grid_north_pole_longitude'])
            if 'north_pole_grid_longitude' in ds.attrs:
                grid_info['north_pole_grid_lon'] = float(ds.attrs['north_pole_grid_longitude'])
        
        # Set defaults if still missing (may not work correctly, but allows code to run)
        if 'pole_lat' not in grid_info:
            grid_info['pole_lat'] = 0.0
            print("  Warning: grid_north_pole_latitude not found, using 0.0")
        if 'pole_lon' not in grid_info:
            grid_info['pole_lon'] = 0.0
            print("  Warning: grid_north_pole_longitude not found, using 0.0")
        if 'north_pole_grid_lon' not in grid_info:
            grid_info['north_pole_grid_lon'] = 0.0
            print("  Warning: north_pole_grid_longitude not found, using 0.0")
        
        return 'rotated_pole', grid_info
    
    # Check for projected coordinates (x/y, projection_x_coordinate, etc.)
    has_x = 'x' in ds.coords or 'x' in ds.dims
    has_y = 'y' in ds.coords or 'y' in ds.dims
    has_proj_x = 'projection_x_coordinate' in ds.coords or 'projection_x_coordinate' in ds.dims
    has_proj_y = 'projection_y_coordinate' in ds.coords or 'projection_y_coordinate' in ds.dims
    
    # Check for grid_mapping with projection info
    grid_mapping_var = None
    for var_name in ds.variables:
        var = ds.variables[var_name]
        if hasattr(var, 'grid_mapping'):
            grid_mapping_name = var.grid_mapping
            if grid_mapping_name in ds.variables:
                grid_mapping_var = ds.variables[grid_mapping_name]
                # Check if it's a projection (not rotated pole)
                if hasattr(grid_mapping_var, 'grid_mapping_name'):
                    gm_name = grid_mapping_var.grid_mapping_name
                    if gm_name not in ['rotated_latitude_longitude']:
                        break
                else:
                    # Might still be a projection
                    break
    
    if (has_x and has_y) or (has_proj_x and has_proj_y) or grid_mapping_var is not None:
        print("  Detected projected coordinate grid")
        if grid_mapping_var is not None:
            # Try to extract CRS information
            grid_info['grid_mapping_var'] = grid_mapping_var
            # Try to build CRS from attributes
            crs_attrs = {}
            for attr in grid_mapping_var.attrs:
                crs_attrs[attr] = grid_mapping_var.attrs[attr]
            grid_info['crs_attrs'] = crs_attrs
        return 'projected', grid_info
    
    # Default: assume geographic lat/lon
    print("  Detected geographic lat/lon grid (fallback)")
    return 'geographic', grid_info


def find_nearest_grid_point_rotated_pole(
    ds: xr.Dataset,
    target_lon: float,
    target_lat: float,
    grid_info: Dict
) -> Tuple[float, float, float, float]:
    """
    Find nearest grid point for a rotated pole grid.
    
    Transforms target (lon, lat) to rotated coordinates and finds nearest grid point.
    
    Args:
        ds: xarray Dataset with rotated pole grid
        target_lon: Target longitude (decimal degrees, geographic)
        target_lat: Target latitude (decimal degrees, geographic)
        grid_info: Dictionary with rotated pole parameters
        
    Returns:
        Tuple of (nearest_rlon, nearest_rlat, grid_lon, grid_lat)
    """
    # Extract rotated pole parameters
    pole_lat = grid_info.get('pole_lat', 0.0)
    pole_lon = grid_info.get('pole_lon', 0.0)
    north_pole_grid_lon = grid_info.get('north_pole_grid_lon', 0.0)
    
    # Determine coordinate names (rlon/rlat or lon/lat in rotated space)
    rlon_name = None
    rlat_name = None
    
    if 'rlon' in ds.coords:
        rlon_name = 'rlon'
    elif 'lon' in ds.coords and 'rlat' in ds.coords:
        rlon_name = 'lon'
    
    if 'rlat' in ds.coords:
        rlat_name = 'rlat'
    elif 'lat' in ds.coords and rlon_name == 'lon':
        rlat_name = 'lat'
    
    if rlon_name is None or rlat_name is None:
        raise ValueError(
            "Cannot find rotated coordinate variables (rlon/rlat or lon/lat). "
            "Rotated pole grid detection may be incorrect."
        )
    
    # Transform geographic coordinates to rotated coordinates
    # Use spherical trigonometry to perform the rotation
    # Convert to radians
    target_lon_rad = np.radians(target_lon)
    target_lat_rad = np.radians(target_lat)
    pole_lon_rad = np.radians(pole_lon)
    pole_lat_rad = np.radians(pole_lat)
    
    # Spherical rotation transformation
    # This implements the standard rotated pole transformation
    # Reference: CF conventions for rotated_latitude_longitude
    
    # Step 1: Rotate so that the rotated pole becomes the north pole
    # First, convert to Cartesian coordinates
    x1 = np.cos(target_lat_rad) * np.cos(target_lon_rad)
    y1 = np.cos(target_lat_rad) * np.sin(target_lon_rad)
    z1 = np.sin(target_lat_rad)
    
    # Rotation matrix to move pole to (pole_lon, pole_lat)
    # Rotate around z-axis by -pole_lon
    cos_alpha = np.cos(-pole_lon_rad)
    sin_alpha = np.sin(-pole_lon_rad)
    x2 = x1 * cos_alpha - y1 * sin_alpha
    y2 = x1 * sin_alpha + y1 * cos_alpha
    z2 = z1
    
    # Rotate around y-axis by (90 - pole_lat)
    beta = np.pi/2 - pole_lat_rad
    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)
    x3 = x2 * cos_beta + z2 * sin_beta
    y3 = y2
    z3 = -x2 * sin_beta + z2 * cos_beta
    
    # Now rotate back around z-axis by north_pole_grid_lon
    gamma_rad = np.radians(north_pole_grid_lon)
    cos_gamma = np.cos(gamma_rad)
    sin_gamma = np.sin(gamma_rad)
    x4 = x3 * cos_gamma - y3 * sin_gamma
    y4 = x3 * sin_gamma + y3 * cos_gamma
    z4 = z3
    
    # Convert back to spherical coordinates (rotated space)
    rlat = np.arcsin(np.clip(z4, -1.0, 1.0))
    rlon = np.arctan2(y4, x4)
    
    # Convert to degrees
    rlat = np.degrees(rlat)
    rlon = np.degrees(rlon)
    
    # Find nearest grid point in rotated space
    rlon_coord = ds.coords[rlon_name]
    rlat_coord = ds.coords[rlat_name]
    
    nearest_rlon = float(rlon_coord.sel({rlon_name: rlon}, method="nearest").values)
    nearest_rlat = float(rlat_coord.sel({rlat_name: rlat}, method="nearest").values)
    
    # Get geographic coordinates of selected grid point for verification
    # For rotated pole grids, lat/lon are typically 2D arrays indexed by rlon/rlat
    if 'lat' in ds.coords and 'lon' in ds.coords:
        try:
            # Try to select using rotated coordinates
            # Check if lat/lon depend on rotated coordinates
            lat_dims = list(ds.coords['lat'].dims)
            lon_dims = list(ds.coords['lon'].dims)
            
            if rlon_name in lat_dims and rlat_name in lat_dims:
                # 2D lat/lon arrays
                grid_lat = float(ds.coords['lat'].sel({rlon_name: nearest_rlon, rlat_name: nearest_rlat}, method='nearest').values)
                grid_lon = float(ds.coords['lon'].sel({rlon_name: nearest_rlon, rlat_name: nearest_rlat}, method='nearest').values)
            else:
                # 1D coordinates or different dimension structure
                # Find indices for rotated coordinates
                rlon_idx = np.argmin(np.abs(rlon_coord.values - nearest_rlon))
                rlat_idx = np.argmin(np.abs(rlat_coord.values - nearest_rlat))
                
                # Try to get lat/lon by index
                if len(ds.coords['lat'].dims) == 1:
                    if rlat_name in ds.coords['lat'].dims:
                        grid_lat = float(ds.coords['lat'].isel({rlat_name: rlat_idx}).values)
                    else:
                        grid_lat = float(ds.coords['lat'].values[rlat_idx])
                else:
                    # Multi-dimensional - use isel with all dimensions
                    sel_dict = {}
                    for dim in ds.coords['lat'].dims:
                        if dim == rlat_name:
                            sel_dict[dim] = rlat_idx
                        elif dim == rlon_name:
                            sel_dict[dim] = rlon_idx
                        else:
                            sel_dict[dim] = 0  # Use first index for other dimensions
                    grid_lat = float(ds.coords['lat'].isel(sel_dict).values)
                
                if len(ds.coords['lon'].dims) == 1:
                    if rlon_name in ds.coords['lon'].dims:
                        grid_lon = float(ds.coords['lon'].isel({rlon_name: rlon_idx}).values)
                    else:
                        grid_lon = float(ds.coords['lon'].values[rlon_idx])
                else:
                    sel_dict = {}
                    for dim in ds.coords['lon'].dims:
                        if dim == rlat_name:
                            sel_dict[dim] = rlat_idx
                        elif dim == rlon_name:
                            sel_dict[dim] = rlon_idx
                        else:
                            sel_dict[dim] = 0
                    grid_lon = float(ds.coords['lon'].isel(sel_dict).values)
        except Exception as e:
            print(f"  Warning: Could not extract geographic coordinates: {e}")
            # Transform rotated coordinates back to geographic (inverse transformation)
            # This is approximate - use the transformation in reverse
            grid_lat = nearest_rlat  # Approximation
            grid_lon = nearest_rlon  # Approximation
    else:
        # No geographic coordinates available - transform back from rotated coordinates
        # Inverse transformation (approximate)
        grid_lat = nearest_rlat
        grid_lon = nearest_rlon
    
    return nearest_rlon, nearest_rlat, grid_lon, grid_lat


def find_nearest_grid_point_projected(
    ds: xr.Dataset,
    target_lon: float,
    target_lat: float,
    grid_info: Dict
) -> Tuple[float, float, float, float]:
    """
    Find nearest grid point for a projected coordinate grid.
    
    Transforms target (lon, lat) to dataset CRS and finds nearest grid point.
    
    Args:
        ds: xarray Dataset with projected grid
        target_lon: Target longitude (decimal degrees, geographic)
        target_lat: Target latitude (decimal degrees, geographic)
        grid_info: Dictionary with projection information
        
    Returns:
        Tuple of (nearest_x, nearest_y, grid_lon, grid_lat)
    """
    # Determine coordinate names
    x_name = None
    y_name = None
    
    if 'x' in ds.coords:
        x_name = 'x'
    elif 'projection_x_coordinate' in ds.coords:
        x_name = 'projection_x_coordinate'
    
    if 'y' in ds.coords:
        y_name = 'y'
    elif 'projection_y_coordinate' in ds.coords:
        y_name = 'projection_y_coordinate'
    
    if x_name is None or y_name is None:
        raise ValueError(
            "Cannot find projected coordinate variables (x/y or projection_x_coordinate/projection_y_coordinate). "
            "Projected grid detection may be incorrect."
        )
    
    # Try to determine CRS from grid_mapping
    crs = None
    if 'grid_mapping_var' in grid_info:
        grid_mapping_var = grid_info['grid_mapping_var']
        try:
            # Try to build CRS from attributes
            crs_attrs = grid_info.get('crs_attrs', {})
            
            # Check for EPSG code
            if 'epsg' in crs_attrs:
                epsg_code = crs_attrs['epsg']
                if isinstance(epsg_code, (int, str)):
                    crs = CRS.from_epsg(int(str(epsg_code).replace('EPSG:', '')))
            
            # Check for CRS string
            if crs is None and 'crs_wkt' in crs_attrs:
                crs = CRS.from_wkt(crs_attrs['crs_wkt'])
            
            # Check for proj4 string
            if crs is None and 'proj4' in crs_attrs:
                crs = CRS.from_string(crs_attrs['proj4'])
            
            # Try to build from grid_mapping_name and parameters
            if crs is None and hasattr(grid_mapping_var, 'grid_mapping_name'):
                gm_name = grid_mapping_var.grid_mapping_name
                # This would require more complex CRS building
                # For now, we'll try to infer from common projections
                pass
                
        except Exception as e:
            print(f"  Warning: Could not determine CRS from grid_mapping: {e}")
    
    if crs is None:
        # Try to infer CRS from coordinate values
        # Check if coordinates look like UTM (large values, typically 6-7 digits)
        x_coord = ds.coords[x_name]
        y_coord = ds.coords[y_name]
        x_vals = x_coord.values
        y_vals = y_coord.values
        
        x_mean = np.mean(x_vals)
        y_mean = np.mean(y_vals)
        
        # UTM coordinates are typically in the range 100000-1000000 for easting
        # and 5000000-10000000 for northing (northern hemisphere)
        if 100000 < abs(x_mean) < 1000000 and 5000000 < abs(y_mean) < 10000000:
            # Likely UTM - try to determine zone from target location
            utm_epsg = get_utm_zone_epsg(target_lon, target_lat)
            try:
                crs = CRS.from_string(utm_epsg)
            except:
                pass
    
    if crs is None:
        raise ValueError(
            "Cannot determine CRS for projected grid. "
            "Grid mapping information may be incomplete or unsupported."
        )
    
    # Transform target coordinates to dataset CRS
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    target_x, target_y = transformer.transform(target_lon, target_lat)
    
    # Find nearest grid point
    x_coord = ds.coords[x_name]
    y_coord = ds.coords[y_name]
    
    nearest_x = float(x_coord.sel({x_name: target_x}, method="nearest").values)
    nearest_y = float(y_coord.sel({y_name: target_y}, method="nearest").values)
    
    # Get geographic coordinates of selected grid point for verification
    if 'lat' in ds.coords and 'lon' in ds.coords:
        # Check if lat/lon are 2D
        if len(ds.coords['lat'].dims) == 2:
            grid_lat = float(ds.coords['lat'].sel({x_name: nearest_x, y_name: nearest_y}, method='nearest').values)
            grid_lon = float(ds.coords['lon'].sel({x_name: nearest_x, y_name: nearest_y}, method='nearest').values)
        else:
            # 1D coordinates - need to find by index
            x_idx = np.argmin(np.abs(x_coord.values - nearest_x))
            y_idx = np.argmin(np.abs(y_coord.values - nearest_y))
            grid_lat = float(ds.coords['lat'].isel({y_name: y_idx}).values)
            grid_lon = float(ds.coords['lon'].isel({x_name: x_idx}).values)
    else:
        # Transform back to geographic coordinates
        inv_transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        grid_lon, grid_lat = inv_transformer.transform(nearest_x, nearest_y)
    
    return nearest_x, nearest_y, grid_lon, grid_lat


def find_nearest_grid_point_geographic(
    ds: xr.Dataset,
    target_lon: float,
    target_lat: float
) -> Tuple[float, float, float, float, Optional[Dict]]:
    """
    Find nearest grid point for a geographic lat/lon grid.
    
    Handles both 1D and 2D coordinate arrays.
    
    Args:
        ds: xarray Dataset with geographic grid
        target_lon: Target longitude (decimal degrees)
        target_lat: Target latitude (decimal degrees)
        
    Returns:
        Tuple of (nearest_lon, nearest_lat, grid_lon, grid_lat, sel_dict)
        sel_dict is None for 1D coordinates, or a dict with dimension indices for 2D coordinates
    """
    # Find coordinate names
    lon_name = 'lon' if 'lon' in ds.coords else 'longitude'
    lat_name = 'lat' if 'lat' in ds.coords else 'latitude'
    
    if lon_name not in ds.coords or lat_name not in ds.coords:
        raise ValueError(
            "Cannot find geographic coordinate variables (lon/lat or longitude/latitude). "
            "Geographic grid detection may be incorrect."
        )
    
    lon_coord = ds.coords[lon_name]
    lat_coord = ds.coords[lat_name]
    
    # Check if coordinates are 1D or 2D
    lon_dims = list(lon_coord.dims)
    lat_dims = list(lat_coord.dims)
    
    if len(lon_dims) == 1 and len(lat_dims) == 1:
        # 1D coordinates - can use sel() directly
        try:
            nearest_lon = float(lon_coord.sel({lon_name: target_lon}, method="nearest").values)
            nearest_lat = float(lat_coord.sel({lat_name: target_lat}, method="nearest").values)
        except (KeyError, ValueError) as e:
            # If sel() fails (e.g., no index), use argmin
            lon_idx = np.argmin(np.abs(lon_coord.values - target_lon))
            lat_idx = np.argmin(np.abs(lat_coord.values - target_lat))
            nearest_lon = float(lon_coord.values[lon_idx])
            nearest_lat = float(lat_coord.values[lat_idx])
        
        return nearest_lon, nearest_lat, nearest_lon, nearest_lat, None
    else:
        # 2D coordinates - need to calculate distances to all points
        # Load coordinate arrays into memory
        lon_array = lon_coord.values
        lat_array = lat_coord.values
        
        # Calculate distances to all grid points using vectorized Haversine
        # Convert to radians
        target_lon_rad = np.radians(target_lon)
        target_lat_rad = np.radians(target_lat)
        lon_rad = np.radians(lon_array)
        lat_rad = np.radians(lat_array)
        
        # Haversine formula (vectorized)
        R = 6371.0  # Earth radius in km
        dlat = lat_rad - target_lat_rad
        dlon = lon_rad - target_lon_rad
        a = np.sin(dlat/2)**2 + np.cos(target_lat_rad) * np.cos(lat_rad) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distances = R * c
        
        # Find index of minimum distance
        min_idx = np.unravel_index(np.argmin(distances), distances.shape)
        
        # Get coordinates of nearest point
        nearest_lon = float(lon_array[min_idx])
        nearest_lat = float(lat_array[min_idx])
        
        # For 2D coordinates, create selection dictionary using dimension indices
        # The dimensions should match between lon and lat
        # Use indices directly for isel()
        sel_dict = {}
        if lon_dims == lat_dims and len(lon_dims) == 2:
            for i, dim_name in enumerate(lon_dims):
                sel_dict[dim_name] = int(min_idx[i])
        else:
            # Fallback: use indices from lon dimensions
            for i, dim_name in enumerate(lon_dims):
                if i < len(min_idx):
                    sel_dict[dim_name] = int(min_idx[i])
        
        return nearest_lon, nearest_lat, nearest_lon, nearest_lat, sel_dict


def inspect_and_convert_rsds_units(rsds_data: xr.DataArray, df_rsds: pd.DataFrame) -> pd.DataFrame:
    """
    Inspect rsds units from NetCDF metadata and convert if necessary.
    
    Common rsds units:
    - W/m² (watts per square meter) - correct, no conversion needed
    - J/m²/day (joules per square meter per day) - needs conversion
    - MJ/m²/day (megajoules per square meter per day) - needs conversion
    
    Conversion: 1 J/m²/day = 1 / 86400 W/m² (since 1 day = 86400 seconds)
               1 MJ/m²/day = 1e6 / 86400 W/m²
    
    Args:
        rsds_data: xarray DataArray with rsds variable
        df_rsds: DataFrame with rsds column (before renaming)
        
    Returns:
        DataFrame with converted rsds values (if conversion was needed)
    """
    # Check units attribute in NetCDF
    units = None
    if hasattr(rsds_data, 'units'):
        units = str(rsds_data.units).strip()
    elif 'rsds' in rsds_data.coords or 'rsds' in rsds_data.dims:
        # Try to get units from the variable itself
        if hasattr(rsds_data, 'attrs') and 'units' in rsds_data.attrs:
            units = str(rsds_data.attrs['units']).strip()
    
    print(f"  rsds units from NetCDF: {units if units else 'not specified'}")
    
    # Check if conversion is needed
    if units:
        units_lower = units.lower()
        
        # Check for J/m²/day or J m-2 day-1 (various formats)
        if 'j/m²/day' in units_lower or 'j m-2 day-1' in units_lower or 'j m-2 d-1' in units_lower:
            # Convert J/m²/day to W/m²
            # 1 J/m²/day = 1 / 86400 W/m²
            df_rsds['rsds'] = df_rsds['rsds'] / 86400.0
            print(f"  Converted rsds from J/m²/day to W/m² (divided by 86400)")
            return df_rsds
        
        # Check for MJ/m²/day or MJ m-2 day-1
        elif 'mj/m²/day' in units_lower or 'mj m-2 day-1' in units_lower or 'mj m-2 d-1' in units_lower:
            # Convert MJ/m²/day to W/m²
            # 1 MJ/m²/day = 1e6 / 86400 W/m²
            df_rsds['rsds'] = df_rsds['rsds'] * 1e6 / 86400.0
            print(f"  Converted rsds from MJ/m²/day to W/m² (multiplied by 1e6/86400)")
            return df_rsds
        
        # Check if already in W/m²
        elif 'w/m²' in units_lower or 'w m-2' in units_lower or 'watt' in units_lower:
            print(f"  rsds already in W/m², no conversion needed")
            return df_rsds
    
    # If units not specified or unrecognized, check value range
    # Typical W/m² values: 0-1000 (daily mean)
    # Typical J/m²/day values: 0-86400000 (daily total)
    if 'rsds' in df_rsds.columns:
        mean_value = df_rsds['rsds'].mean()
        max_value = df_rsds['rsds'].max()
        
        # If values are very large (> 10000), likely in J/m²/day
        if mean_value > 10000 or max_value > 100000:
            print(f"  WARNING: rsds values are very large (mean={mean_value:.1f}, max={max_value:.1f})")
            print(f"  Assuming J/m²/day and converting to W/m²")
            df_rsds['rsds'] = df_rsds['rsds'] / 86400.0
            return df_rsds
        else:
            print(f"  rsds values appear to be in W/m² (mean={mean_value:.1f}), no conversion applied")
    
    return df_rsds


def find_relevant_files(file_path: str, start_year: int, end_year: int) -> List[str]:
    """
    Find NetCDF files that cover the requested time period.
    
    Handles cases where data is split across multiple files with different
    date ranges (e.g., 2011-2020, 2021-2030, etc.).
    
    Args:
        file_path: Path to one example file (used to infer pattern)
        start_year: Start year of requested period
        end_year: End year of requested period
        
    Returns:
        List of file paths that cover any part of the requested period
    """
    directory = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    var_prefix = base_name.split('_')[0]  # tas, tasmax, rsds, etc.
    
    # Extract common parts of the filename to build pattern
    # Example: tas_KGDK-1_ICHEC-EC-EARTH_rcp85_r3i1p1_KNMI-RACMO22E_v1_day_20110101-20401231.nc
    parts = base_name.split('_')
    if len(parts) >= 7:
        pattern = f"{var_prefix}_KGDK-1_{parts[2]}_{parts[3]}_{parts[4]}_{parts[5]}_{parts[6]}_*.nc"
    else:
        # Fallback: simpler pattern
        pattern = f"{var_prefix}_*.nc"
    
    relevant_files = []
    for file in glob.glob(os.path.join(directory, pattern)):
        # Extract date range from filename (last part before .nc)
        date_range = os.path.basename(file).split('_')[-1].replace('.nc', '')
        
        # Parse date range (format: YYYYMMDD-YYYYMMDD)
        if '-' in date_range:
            try:
                file_start_str, file_end_str = date_range.split('-')
                file_start = int(file_start_str[:4])  # Extract year
                file_end = int(file_end_str[:4])
                
                # Check if file contains any part of the requested period
                if (file_start <= end_year) and (file_end >= start_year):
                    relevant_files.append(file)
            except (ValueError, IndexError):
                # If date parsing fails, include the file anyway (conservative)
                relevant_files.append(file)
        else:
            # No date range in filename, include it
            relevant_files.append(file)
    
    return sorted(relevant_files)


def extract_daily_projections(
    lon: float,
    lat: float,
    start_year: int,
    end_year: int,
    netcdf_files: Dict[str, str]
) -> pd.DataFrame:
    """
    Extract daily projection data from NetCDF files for target location.
    
    Args:
        lon: Target longitude (decimal degrees)
        lat: Target latitude (decimal degrees)
        start_year: Start year of projection period
        end_year: End year of projection period
        netcdf_files: Dictionary with keys: tas, tasmax, tasmin, rsds, sfcWind, pr, potevap
                     and values as file paths
        
    Returns:
        DataFrame with columns:
        - date: datetime (YYYY-MM-DD)
        - T_proj: daily mean temperature (°C)
        - R_proj: daily mean solar radiation (W/m²)
        - W_proj: daily mean wind speed (m/s)
        - Optional: max_temp, min_temp, precipitation, potential_evapotranspiration
    """
    print(f"Target location: {lat:.4f}°N, {lon:.4f}°E")
    
    # Find relevant files for each variable (handles multi-file datasets)
    print(f"\nFinding relevant files for period {start_year}-{end_year}...")
    
    tas_files = find_relevant_files(netcdf_files['tas'], start_year, end_year)
    tasmax_files = find_relevant_files(netcdf_files['tasmax'], start_year, end_year)
    tasmin_files = find_relevant_files(netcdf_files['tasmin'], start_year, end_year)
    rsds_files = find_relevant_files(netcdf_files['rsds'], start_year, end_year)
    sfcWind_files = find_relevant_files(netcdf_files['sfcWind'], start_year, end_year)
    pr_files = find_relevant_files(netcdf_files['pr'], start_year, end_year)
    potevap_files = find_relevant_files(netcdf_files['potevap'], start_year, end_year)
    
    if not tas_files:
        raise ValueError(f"No tas files found for period {start_year}-{end_year}")
    
    print(f"  tas: {len(tas_files)} file(s)")
    print(f"  rsds: {len(rsds_files)} file(s)")
    print(f"  sfcWind: {len(sfcWind_files)} file(s)")
    
    # Load and combine datasets
    print("\nLoading NetCDF datasets...")
    ds_tas = xr.open_mfdataset(tas_files, combine='by_coords', chunks={'time': 1000})
    ds_tasmax = xr.open_mfdataset(tasmax_files, combine='by_coords', chunks={'time': 1000})
    ds_tasmin = xr.open_mfdataset(tasmin_files, combine='by_coords', chunks={'time': 1000})
    ds_rsds = xr.open_mfdataset(rsds_files, combine='by_coords', chunks={'time': 1000})
    ds_sfcWind = xr.open_mfdataset(sfcWind_files, combine='by_coords', chunks={'time': 1000})
    ds_pr = xr.open_mfdataset(pr_files, combine='by_coords', chunks={'time': 1000}) if pr_files else None
    ds_potevap = xr.open_mfdataset(potevap_files, combine='by_coords', chunks={'time': 1000}) if potevap_files else None
    
    # Sort by time to ensure chronological order
    ds_tas = ds_tas.sortby('time')
    ds_tasmax = ds_tasmax.sortby('time')
    ds_tasmin = ds_tasmin.sortby('time')
    ds_rsds = ds_rsds.sortby('time')
    ds_sfcWind = ds_sfcWind.sortby('time')
    if ds_pr is not None:
        ds_pr = ds_pr.sortby('time')
    if ds_potevap is not None:
        ds_potevap = ds_potevap.sortby('time')
    
    # Detect grid type and find nearest grid point
    print("\nDetecting grid coordinate system...")
    grid_type, grid_info = detect_grid_type(ds_tas)
    print(f"  Grid type: {grid_type}")
    
    # Find nearest grid point based on grid type
    use_isel = False  # Default to sel() for coordinate-based selection
    if grid_type == 'rotated_pole':
        coord1, coord2, grid_lon, grid_lat = find_nearest_grid_point_rotated_pole(
            ds_tas, lon, lat, grid_info
        )
        coord1_name = 'rlon' if 'rlon' in ds_tas.coords else 'lon'
        coord2_name = 'rlat' if 'rlat' in ds_tas.coords else 'lat'
        print(f"  Selected rotated coordinates: {coord1_name}={coord1:.4f}, {coord2_name}={coord2:.4f}")
        sel_dict = {coord1_name: coord1, coord2_name: coord2}
        
    elif grid_type == 'projected':
        coord1, coord2, grid_lon, grid_lat = find_nearest_grid_point_projected(
            ds_tas, lon, lat, grid_info
        )
        coord1_name = 'x' if 'x' in ds_tas.coords else 'projection_x_coordinate'
        coord2_name = 'y' if 'y' in ds_tas.coords else 'projection_y_coordinate'
        print(f"  Selected projected coordinates: {coord1_name}={coord1:.2f}, {coord2_name}={coord2:.2f}")
        sel_dict = {coord1_name: coord1, coord2_name: coord2}
        
    else:  # geographic
        coord1, coord2, grid_lon, grid_lat, sel_dict_geo = find_nearest_grid_point_geographic(
            ds_tas, lon, lat
        )
        if sel_dict_geo is not None:
            # 2D coordinates - use the dimension-based selection with isel
            sel_dict = sel_dict_geo
            coord1_name = list(sel_dict.keys())[0] if sel_dict else 'lon'
            coord2_name = list(sel_dict.keys())[1] if len(sel_dict) > 1 else 'lat'
            print(f"  Selected geographic grid point using dimensions: {sel_dict}")
            use_isel = True  # Use isel() for index-based selection
        else:
            # 1D coordinates - use coordinate values
            coord1_name = 'lon' if 'lon' in ds_tas.coords else 'longitude'
            coord2_name = 'lat' if 'lat' in ds_tas.coords else 'latitude'
            print(f"  Selected geographic coordinates: {coord1_name}={coord1:.4f}, {coord2_name}={coord2:.4f}")
            sel_dict = {coord1_name: coord1, coord2_name: coord2}
            use_isel = False
    
    # Calculate distance from target to selected grid point
    # Use geographic distance (Haversine) for accuracy
    distance_km = calculate_geographic_distance(lon, lat, float(grid_lon), float(grid_lat))
    
    print(f"\nSelected grid point:")
    print(f"  Grid lat: {grid_lat:.4f}°N, Grid lon: {grid_lon:.4f}°E")
    print(f"  Distance from target: {distance_km:.2f} km")
    
    # MANDATORY SANITY CHECK: Fail if distance > 20 km
    if distance_km > 20:
        raise ValueError(
            f"Selected projection grid point is {distance_km:.1f} km from target. "
            "This indicates a coordinate system mismatch. "
            f"Target: ({lat:.4f}°N, {lon:.4f}°E), "
            f"Selected: ({grid_lat:.4f}°N, {grid_lon:.4f}°E), "
            f"Grid type: {grid_type}"
        )
    
    # Extract time period
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    
    print(f"\nExtracting data from {start_date} to {end_date}...")
    
    # Extract data at nearest grid point for the specified time period
    # Use isel for 2D coordinates (indices), sel for 1D coordinates (values)
    if use_isel:
        # 2D coordinates - use isel with indices
        tas = ds_tas.isel(**sel_dict).sel(time=slice(start_date, end_date))
        tasmax = ds_tasmax.isel(**sel_dict).sel(time=slice(start_date, end_date))
        tasmin = ds_tasmin.isel(**sel_dict).sel(time=slice(start_date, end_date))
        rsds = ds_rsds.isel(**sel_dict).sel(time=slice(start_date, end_date))
        sfcWind = ds_sfcWind.isel(**sel_dict).sel(time=slice(start_date, end_date))
    else:
        # 1D coordinates - use sel with coordinate values
        tas = ds_tas.sel(
            time=slice(start_date, end_date),
            **sel_dict,
            method='nearest'
        )
        tasmax = ds_tasmax.sel(
            time=slice(start_date, end_date),
            **sel_dict,
            method='nearest'
        )
        tasmin = ds_tasmin.sel(
            time=slice(start_date, end_date),
            **sel_dict,
            method='nearest'
        )
        rsds = ds_rsds.sel(
            time=slice(start_date, end_date),
            **sel_dict,
            method='nearest'
        )
        sfcWind = ds_sfcWind.sel(
            time=slice(start_date, end_date),
            **sel_dict,
            method='nearest'
        )
    
    # Convert to pandas DataFrames
    df_tas = tas.to_dataframe().reset_index().drop_duplicates(subset=['time'])
    df_tasmax = tasmax.to_dataframe().reset_index().drop_duplicates(subset=['time'])
    df_tasmin = tasmin.to_dataframe().reset_index().drop_duplicates(subset=['time'])
    df_rsds = rsds.to_dataframe().reset_index().drop_duplicates(subset=['time'])
    df_sfcWind = sfcWind.to_dataframe().reset_index().drop_duplicates(subset=['time'])
    
    # Inspect and convert rsds units if necessary
    print("\nInspecting rsds units...")
    df_rsds = inspect_and_convert_rsds_units(rsds, df_rsds)
    
    # Optional variables
    df_pr = None
    df_potevap = None
    if ds_pr is not None:
        if use_isel:
            pr = ds_pr.isel(**sel_dict).sel(time=slice(start_date, end_date))
        else:
            pr = ds_pr.sel(
                time=slice(start_date, end_date),
                **sel_dict,
                method='nearest'
            )
        df_pr = pr.to_dataframe().reset_index().drop_duplicates(subset=['time'])
    
    if ds_potevap is not None:
        if use_isel:
            potevap = ds_potevap.isel(**sel_dict).sel(time=slice(start_date, end_date))
        else:
            potevap = ds_potevap.sel(
                time=slice(start_date, end_date),
                **sel_dict,
                method='nearest'
            )
        df_potevap = potevap.to_dataframe().reset_index().drop_duplicates(subset=['time'])
    
    # Merge datasets on time
    df = df_tas[['time', 'tas']].copy()
    df = df.merge(df_tasmax[['time', 'tasmax']], on='time', how='left')
    df = df.merge(df_tasmin[['time', 'tasmin']], on='time', how='left')
    df = df.merge(df_rsds[['time', 'rsds']], on='time', how='left')
    df = df.merge(df_sfcWind[['time', 'sfcWind']], on='time', how='left')
    
    if df_pr is not None:
        df = df.merge(df_pr[['time', 'pr']], on='time', how='left')
    if df_potevap is not None:
        df = df.merge(df_potevap[['time', 'potevap']], on='time', how='left')
    
    # Convert units
    # Temperature: Kelvin to Celsius (if values > 100, assume Kelvin)
    print("\nConverting units...")
    temp_cols = ['tas', 'tasmax', 'tasmin']
    for col in temp_cols:
        if col in df.columns:
            if df[col].mean() > 100:  # Likely in Kelvin
                df[col] = df[col] - 273.15
                print(f"  Converted {col} from Kelvin to Celsius")
    
    # rsds units already handled by inspect_and_convert_rsds_units()
    # sfcWind: Should already be in m/s, verify units if possible
    # No conversion needed if units are correct
    
    # Rename columns to match guideline output format
    df.rename(columns={
        'time': 'date',
        'tas': 'T_proj',  # Daily mean temperature
        'rsds': 'R_proj',  # Daily mean solar radiation
        'sfcWind': 'W_proj',  # Daily mean wind speed
        'tasmax': 'max_temp',  # Optional
        'tasmin': 'min_temp',  # Optional
        'pr': 'precipitation',  # Optional
        'potevap': 'potential_evapotranspiration'  # Optional
    }, inplace=True)
    
    # Ensure date column is datetime type (required by interface_cli.py)
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Verify required columns exist
    required_cols = ['date', 'T_proj', 'R_proj', 'W_proj']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"\n✓ Extracted {len(df)} daily records")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Variables: T_proj, R_proj, W_proj")
    
    # Close datasets to free memory
    ds_tas.close()
    ds_tasmax.close()
    ds_tasmin.close()
    ds_rsds.close()
    ds_sfcWind.close()
    if ds_pr is not None:
        ds_pr.close()
    if ds_potevap is not None:
        ds_potevap.close()
    
    return df
