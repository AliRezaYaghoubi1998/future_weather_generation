"""
Projection File Resolver

Deterministically resolves projection NetCDF files and historical data paths
from directory-based inputs, without requiring manual file path specification.

This module provides:
1. Projection NetCDF Resolver: Scans directories for CORDEX-style files and
   resolves them by scenario, GCM, RCM, and temporal coverage.
2. Historical File Resolver: Constructs historical daily file paths from
   target dates.

Design principles:
- No GUI, no interactive prompts, no external APIs
- No silent fallback logic - fail fast with explicit errors
- Strict validation and deterministic behavior
- Full logging for transparency and reproducibility
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date, timedelta
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)

# Required variables for projection files
REQUIRED_VARIABLES = [
    "tas",
    "tasmax",
    "tasmin",
    "rsds",
    "sfcWind",
    "pr",
    "potevap",
]


@dataclass
class ParsedFilename:
    """Parsed components of a CORDEX NetCDF filename."""
    variable: str
    scenario: str
    gcm: str
    rcm: str
    start_date: date
    end_date: date
    file_path: Path


def _parse_projection_filename(file_path: Path) -> Optional[ParsedFilename]:
    """
    Parse a CORDEX-style NetCDF filename to extract metadata.
    
    Expected filename pattern:
    <variable>_<domain>_<GCM>_<scenario>_<ensemble>_<RCM>_<version>_<frequency>_<date_range>.nc
    
    Example:
    tas_KGDK-1_ICHEC-EC-EARTH_rcp45_r3i1p1_DMI-HIRHAM5_v2_day_20410101-20701231.nc
    
    Args:
        file_path: Path to the NetCDF file
        
    Returns:
        ParsedFilename object if parsing succeeds, None otherwise
        
    Raises:
        ValueError: If filename cannot be parsed (logged as warning, returns None)
    """
    filename = file_path.name
    
    # CORDEX filename pattern:
    # variable_domain_GCM_scenario_ensemble_RCM_version_frequency_YYYYMMDD-YYYYMMDD.nc
    pattern = (
        r"^([a-zA-Z]+)_"  # variable (tas, tasmax, etc.)
        r"([^_]+)_"  # domain (e.g., KGDK-1)
        r"([^_]+)_"  # GCM (e.g., ICHEC-EC-EARTH)
        r"([^_]+)_"  # scenario (e.g., rcp45, rcp85)
        r"([^_]+)_"  # ensemble (e.g., r3i1p1)
        r"([^_]+)_"  # RCM (e.g., DMI-HIRHAM5)
        r"([^_]+)_"  # version (e.g., v1, v2)
        r"([^_]+)_"  # frequency (e.g., day)
        r"(\d{8})-(\d{8})\.nc$"  # date range: YYYYMMDD-YYYYMMDD
    )
    
    match = re.match(pattern, filename)
    if not match:
        logger.warning(f"Could not parse filename: {filename}")
        return None
    
    try:
        variable = match.group(1)
        gcm = match.group(3)
        scenario = match.group(4)
        rcm = match.group(6)
        start_date_str = match.group(9)
        end_date_str = match.group(10)
        
        # Parse dates
        start_date = datetime.strptime(start_date_str, "%Y%m%d").date()
        end_date = datetime.strptime(end_date_str, "%Y%m%d").date()
        
        return ParsedFilename(
            variable=variable,
            scenario=scenario,
            gcm=gcm,
            rcm=rcm,
            start_date=start_date,
            end_date=end_date,
            file_path=file_path
        )
    except (ValueError, IndexError) as e:
        logger.warning(f"Error parsing filename {filename}: {e}")
        return None


def _validate_coverage(
    files: List[ParsedFilename],
    start_year: int,
    end_year: int
) -> Tuple[bool, Optional[str]]:
    """
    Validate that files collectively cover the requested time period without gaps.
    
    Args:
        files: List of parsed filenames (must be for same variable)
        start_year: Requested start year (inclusive)
        end_year: Requested end year (inclusive)
        
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if coverage is complete, False otherwise
        - error_message: None if valid, otherwise descriptive error message
    """
    if not files:
        return False, "No files provided"
    
    # Sort files by start date
    sorted_files = sorted(files, key=lambda f: f.start_date)
    
    # Calculate required date range
    required_start = date(start_year, 1, 1)
    required_end = date(end_year, 12, 31)
    
    # Check if first file starts before or at required start
    if sorted_files[0].start_date > required_start:
        return False, (
            f"Coverage gap: first file starts {sorted_files[0].start_date}, "
            f"but required start is {required_start}"
        )
    
    # Check if last file ends at or after required end
    if sorted_files[-1].end_date < required_end:
        return False, (
            f"Coverage gap: last file ends {sorted_files[-1].end_date}, "
            f"but required end is {required_end}"
        )
    
    # Check for gaps between files
    for i in range(len(sorted_files) - 1):
        current_end = sorted_files[i].end_date
        next_start = sorted_files[i + 1].start_date
        
        # Allow overlap but not gaps
        # Gap exists if next file starts more than 1 day after current ends
        if next_start > current_end + timedelta(days=1):
            return False, (
                f"Coverage gap: file {sorted_files[i].file_path.name} "
                f"ends {current_end}, but next file "
                f"{sorted_files[i + 1].file_path.name} starts {next_start}"
            )
    
    return True, None


def _select_minimal_file_set(
    candidates: List[ParsedFilename],
    start_year: int,
    end_year: int
) -> List[ParsedFilename]:
    """
    Select minimal set of files that cover the requested period.
    
    Uses greedy algorithm: sort by start date, select files that extend
    coverage furthest without gaps.
    
    Args:
        candidates: List of parsed filenames (same variable, scenario, GCM, RCM)
        start_year: Requested start year
        end_year: Requested end year
        
    Returns:
        List of selected files (sorted by start date)
    """
    required_start = date(start_year, 1, 1)
    required_end = date(end_year, 12, 31)
    
    # Sort by start date
    sorted_candidates = sorted(candidates, key=lambda f: f.start_date)
    
    selected = []
    current_end = required_start - timedelta(days=1)  # Start before required period
    
    for candidate in sorted_candidates:
        # Skip files that end before current coverage
        if candidate.end_date <= current_end:
            continue
        
        # If this file extends coverage, add it
        if candidate.start_date <= current_end + timedelta(days=1):
            selected.append(candidate)
            current_end = max(current_end, candidate.end_date)
            
            # If we've covered the required period, stop
            if current_end >= required_end:
                break
    
    return selected


def resolve_projection_files(
    projection_dir: Path,
    scenario: str,
    gcm: str,
    rcm: str,
    start_year: int,
    end_year: int
) -> Dict[str, List[Path]]:
    """
    Resolve projection NetCDF files from directory based on scenario, GCM, RCM, and temporal coverage.
    
    Scans the directory recursively for .nc files, parses filenames to extract metadata,
    filters by exact matches, and validates temporal coverage.
    
    Args:
        projection_dir: Root directory containing CORDEX-style NetCDF files
        scenario: Scenario name (e.g., "rcp45", "rcp85") - matched case-insensitively
        gcm: Global Climate Model name (e.g., "ICHEC-EC-EARTH")
        rcm: Regional Climate Model name (e.g., "DMI-HIRHAM5")
        start_year: Start year of requested period (inclusive)
        end_year: End year of requested period (inclusive)
        
    Returns:
        Dictionary mapping variable names to lists of Path objects:
        {
            "tas": [Path(...), Path(...), ...],
            "tasmax": [Path(...), ...],
            "tasmin": [Path(...), ...],
            "rsds": [Path(...), ...],
            "sfcWind": [Path(...), ...],
            "pr": [Path(...), ...],
            "potevap": [Path(...), ...]
        }
        
    Raises:
        ValueError: If projection_dir doesn't exist, required variables are missing,
                   coverage is incomplete, or filenames cannot be parsed
    """
    # Validate inputs
    projection_dir = Path(projection_dir)
    if not projection_dir.exists():
        raise ValueError(f"Projection directory does not exist: {projection_dir}")
    
    if not projection_dir.is_dir():
        raise ValueError(f"Projection path is not a directory: {projection_dir}")
    
    if start_year > end_year:
        raise ValueError(f"Start year ({start_year}) must be before or equal to end year ({end_year})")
    
    scenario_lower = scenario.lower()
    
    logger.info(f"Scanning projection directory: {projection_dir}")
    logger.info(f"Search criteria: scenario={scenario}, GCM={gcm}, RCM={rcm}")
    logger.info(f"Temporal coverage: {start_year}-{end_year}")
    
    # Scan directory recursively for .nc files
    all_nc_files = list(projection_dir.rglob("*.nc"))
    logger.info(f"Found {len(all_nc_files)} NetCDF file(s) in directory")
    
    # Parse all filenames
    parsed_files = []
    parse_failures = []
    
    for file_path in all_nc_files:
        parsed = _parse_projection_filename(file_path)
        if parsed:
            parsed_files.append(parsed)
        else:
            parse_failures.append(file_path.name)
    
    if parse_failures:
        logger.warning(f"Could not parse {len(parse_failures)} filename(s)")
        if len(parse_failures) <= 10:
            for failure in parse_failures:
                logger.warning(f"  - {failure}")
        else:
            logger.warning(f"  (showing first 10 of {len(parse_failures)} failures)")
            for failure in parse_failures[:10]:
                logger.warning(f"  - {failure}")
    
    # Filter by scenario (case-insensitive), GCM, RCM
    matching_files = []
    for parsed in parsed_files:
        if (parsed.scenario.lower() == scenario_lower and
            parsed.gcm == gcm and
            parsed.rcm == rcm):
            matching_files.append(parsed)
    
    logger.info(f"Found {len(matching_files)} file(s) matching scenario/GCM/RCM criteria")
    
    # Group by variable
    files_by_variable: Dict[str, List[ParsedFilename]] = {}
    for parsed in matching_files:
        var = parsed.variable
        if var not in files_by_variable:
            files_by_variable[var] = []
        files_by_variable[var].append(parsed)
    
    # Log matching candidates per variable
    logger.info("Matching candidates per variable:")
    for var in REQUIRED_VARIABLES:
        count = len(files_by_variable.get(var, []))
        logger.info(f"  {var}: {count} candidate(s)")
    
    # Check for missing required variables
    missing_variables = []
    for var in REQUIRED_VARIABLES:
        if var not in files_by_variable or len(files_by_variable[var]) == 0:
            missing_variables.append(var)
    
    if missing_variables:
        raise ValueError(
            f"Missing required variables: {', '.join(missing_variables)}\n"
            f"Found variables: {', '.join(sorted(files_by_variable.keys()))}\n"
            f"Search criteria: scenario={scenario}, GCM={gcm}, RCM={rcm}"
        )
    
    # Resolve files for each variable
    resolved_files: Dict[str, List[Path]] = {}
    
    for var in REQUIRED_VARIABLES:
        candidates = files_by_variable[var]
        
        # Select minimal file set
        selected = _select_minimal_file_set(candidates, start_year, end_year)
        
        if not selected:
            raise ValueError(
                f"No files selected for variable {var} covering period "
                f"{start_year}-{end_year}"
            )
        
        # Validate coverage
        is_valid, error_msg = _validate_coverage(selected, start_year, end_year)
        if not is_valid:
            raise ValueError(
                f"Incomplete temporal coverage for variable {var}: {error_msg}\n"
                f"Selected files: {[f.file_path.name for f in selected]}"
            )
        
        # Return all selected files as a list
        resolved_files[var] = [f.file_path for f in selected]
        
        # Log selection
        logger.info(f"Selected file(s) for {var}:")
        for sel_file in selected:
            logger.info(
                f"  {sel_file.file_path.name} "
                f"({sel_file.start_date} to {sel_file.end_date})"
            )
        
        # Log coverage
        coverage_start = min(f.start_date for f in selected)
        coverage_end = max(f.end_date for f in selected)
        logger.info(
            f"  Coverage: {coverage_start} to {coverage_end} "
            f"(requested: {date(start_year, 1, 1)} to {date(end_year, 12, 31)})"
        )
    
    return resolved_files


def resolve_historical_day_file(
    historical_root: Path,
    target_date: date
) -> Path:
    """
    Resolve historical daily file path from target date.
    
    Constructs path as: <historical_root>/<year>/<YYYY-MM-DD>.txt
    
    Args:
        historical_root: Root directory containing year subdirectories
        target_date: Target date for historical file
        
    Returns:
        Path to the historical daily file
        
    Raises:
        ValueError: If year directory or daily file does not exist
    """
    historical_root = Path(historical_root)
    
    if not historical_root.exists():
        raise ValueError(f"Historical root directory does not exist: {historical_root}")
    
    if not historical_root.is_dir():
        raise ValueError(f"Historical root path is not a directory: {historical_root}")
    
    # Construct expected path: <root>/<year>/<YYYY-MM-DD>.txt
    year_dir = historical_root / str(target_date.year)
    daily_file = year_dir / f"{target_date.strftime('%Y-%m-%d')}.txt"
    
    # Validate year directory exists
    if not year_dir.exists():
        raise ValueError(
            f"Year directory does not exist: {year_dir}\n"
            f"Target date: {target_date}"
        )
    
    if not year_dir.is_dir():
        raise ValueError(
            f"Year path is not a directory: {year_dir}\n"
            f"Target date: {target_date}"
        )
    
    # Validate daily file exists
    if not daily_file.exists():
        raise ValueError(
            f"Historical daily file does not exist: {daily_file}\n"
            f"Target date: {target_date}\n"
            f"Expected format: <root>/{target_date.year}/{target_date.strftime('%Y-%m-%d')}.txt"
        )
    
    logger.debug(f"Resolved historical file: {daily_file} for date {target_date}")
    
    return daily_file


if __name__ == "__main__":
    """
    Command-line interface for testing the resolver.
    
    Usage:
        python projection_resolver.py --projection-dir <dir> --scenario <scenario> \\
            --gcm <gcm> --rcm <rcm> --start-year <year> --end-year <year>
        
        python projection_resolver.py --historical-root <dir> --date <YYYY-MM-DD>
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test projection and historical file resolvers"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Projection resolver subcommand
    proj_parser = subparsers.add_parser("projection", help="Resolve projection files")
    proj_parser.add_argument("--projection-dir", type=Path, required=True,
                            help="Directory containing NetCDF files")
    proj_parser.add_argument("--scenario", required=True,
                            help="Scenario (e.g., rcp45, rcp85)")
    proj_parser.add_argument("--gcm", required=True,
                            help="Global Climate Model name")
    proj_parser.add_argument("--rcm", required=True,
                            help="Regional Climate Model name")
    proj_parser.add_argument("--start-year", type=int, required=True,
                            help="Start year")
    proj_parser.add_argument("--end-year", type=int, required=True,
                            help="End year")
    
    # Historical resolver subcommand
    hist_parser = subparsers.add_parser("historical", help="Resolve historical file")
    hist_parser.add_argument("--historical-root", type=Path, required=True,
                            help="Root directory containing year subdirectories")
    hist_parser.add_argument("--date", type=str, required=True,
                            help="Target date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )
    
    if args.command == "projection":
        try:
            resolved = resolve_projection_files(
                args.projection_dir,
                args.scenario,
                args.gcm,
                args.rcm,
                args.start_year,
                args.end_year
            )
            print("\n✓ Successfully resolved projection files:")
            for var, paths in resolved.items():
                if len(paths) == 1:
                    print(f"  {var}: {paths[0]}")
                else:
                    print(f"  {var}: {len(paths)} file(s)")
                    for path in paths:
                        print(f"    - {path}")
        except ValueError as e:
            print(f"\n✗ Error: {e}")
            exit(1)
    
    elif args.command == "historical":
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
            resolved = resolve_historical_day_file(args.historical_root, target_date)
            print(f"\n✓ Successfully resolved historical file:")
            print(f"  {resolved}")
        except ValueError as e:
            print(f"\n✗ Error: {e}")
            exit(1)
    
    else:
        parser.print_help()
