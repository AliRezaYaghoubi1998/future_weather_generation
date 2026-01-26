"""
FTMY Pipeline Configuration Module

Single source of truth for execution mode, paths, and pipeline configuration.
All modules must use this module for path resolution and mode detection.

Reference behavior: references/config.py (path management and mode switching)

Author: Ali Reza Yaghoubi
"""

import os
from pathlib import Path
from typing import Optional


class PipelineConfig:
    """
    Central configuration class for FTMY pipeline.
    
    Manages:
    - Execution mode (test/full) via FTMY_MODE environment variable
    - Results path resolution (Output/results vs Test_Output/results)
    - Input data paths (historical, projection, daily summaries)
    - Test mode configuration
    """
    
    # ============================================================================
    # Execution Mode Configuration
    # ============================================================================
    
    @staticmethod
    def get_execution_mode() -> str:
        """
        Get current execution mode from environment variable.
        
        Returns:
            "test" or "full" (defaults to "full" if not set)
        """
        return os.getenv("FTMY_MODE", "full").lower()
    
    @staticmethod
    def is_test_mode() -> bool:
        """Check if pipeline is running in test mode"""
        return PipelineConfig.get_execution_mode() == "test"
    
    @staticmethod
    def is_full_mode() -> bool:
        """Check if pipeline is running in full mode"""
        return PipelineConfig.get_execution_mode() == "full"
    
    # ============================================================================
    # Input Data Paths (Infrastructure - Centralized)
    # ============================================================================
    # These paths MUST be set via environment variables. No machine-specific defaults.
    # External datasets are large (>20 GB) and must be downloaded manually by the user.
    
    # Historical data base directory (contains year subdirectories with .txt files)
    BASE_DIR = os.getenv("FTMY_BASE_DIR", None)
    
    # Climate analysis path (legacy, may not be used in new pipeline)
    CLIMATE_ANALYSIS_PATH = os.getenv("FTMY_CLIMATE_ANALYSIS_PATH", None)
    
    # Prediction/projection NetCDF files directory
    PREDICTION_PATH = os.getenv("FTMY_PREDICTION_PATH", None)
    
    # Daily summaries directory (contains daily_summary_MM-DD.pkl files)
    DAILY_SUMMARIES_PATH = os.getenv("FTMY_DAILY_SUMMARIES_PATH", None)
    
    # ============================================================================
    # Output Paths (Mode-Dependent)
    # ============================================================================
    
    @staticmethod
    def get_output_base() -> Path:
        """
        Get base output directory based on execution mode.
        
        Workspace root is determined by:
        1. FTMY_WORKSPACE_ROOT environment variable (if set)
        2. Repository root (auto-detected from config.py location)
        
        Returns:
            Path to Output/ or Test_Output/ directory
        """
        # Option 1: Environment variable (highest priority)
        workspace_root_env = os.getenv("FTMY_WORKSPACE_ROOT")
        if workspace_root_env:
            workspace_root = Path(workspace_root_env)
        else:
            # Option 2: Auto-detect repository root from config.py location
            # config.py is at: Src/pipeline/config.py
            # Repo root is: config.py -> parent -> parent -> parent
            config_file = Path(__file__)
            workspace_root = config_file.parent.parent.parent
        
        mode = PipelineConfig.get_execution_mode()
        
        if mode == "test":
            return workspace_root / "Test_Output"
        else:
            return workspace_root / "Output"
    
    @staticmethod
    def get_results_path() -> Path:
        """
        Get results directory path, creating it if necessary.
        
        Returns:
            Path to results/ subdirectory (Output/results or Test_Output/results)
        """
        results_path = PipelineConfig.get_output_base() / "results"
        results_path.mkdir(parents=True, exist_ok=True)
        return results_path
    
    # ============================================================================
    # Test Mode Configuration
    # ============================================================================
    
    # Test days: (month, day) tuples for engineering benchmarking only
    # These are the ONLY days processed in test mode.
    # FS-based month selection is still executed for alignment with full mode,
    # but must NOT restrict or filter these test days.
    TEST_DAYS = [
        # 5 winter days
        (1, 15),   # Jan 15
        (2, 1),    # Feb 1
        (12, 20),  # Dec 20
        (1, 5),    # Jan 5
        (2, 28),   # Feb 28
        # 5 summer days
        (6, 15),   # Jun 15
        (7, 20),   # Jul 20
        (8, 10),   # Aug 10
        (6, 1),    # Jun 1
        (7, 31),   # Jul 31
    ]
    
    # Test mode disclaimer (must be included in all test mode outputs)
    TEST_MODE_DISCLAIMER = (
        "WARNING: This output was generated in TEST MODE for engineering "
        "performance benchmarking only. Test mode preserves the full pipeline "
        "structure, including FS-based month selection and multi-variable "
        "historical matching. Processing is limited to a predefined subset of "
        "test days for engineering and debugging purposes only. Results are "
        "not climate-representative and must not be used for scientific analysis."
    )
    
    # ============================================================================
    # Path Resolution and Logging
    # ============================================================================
    
    @staticmethod
    def print_configuration():
        """
        Print resolved configuration at runtime.
        Called by interface_cli.py at pipeline start.
        """
        mode = PipelineConfig.get_execution_mode()
        results_path = PipelineConfig.get_results_path()
        
        print("=" * 80)
        print("FTMY PIPELINE CONFIGURATION")
        print("=" * 80)
        print(f"Execution Mode: {mode.upper()}")
        print(f"Results Path: {results_path}")
        print()
        print("Input Data Paths:")
        print(f"  Historical Base: {PipelineConfig.BASE_DIR}")
        print(f"  Prediction/Projection: {PipelineConfig.PREDICTION_PATH}")
        print(f"  Daily Summaries: {PipelineConfig.DAILY_SUMMARIES_PATH}")
        print()
        
        if mode == "test":
            print("TEST MODE ACTIVE:")
            print(f"  Test Days: {PipelineConfig.TEST_DAYS}")
            print(f"  {PipelineConfig.TEST_MODE_DISCLAIMER}")
            print()
        
        print("=" * 80)
    
    @staticmethod
    def validate_paths() -> bool:
        """
        Validate that all required input paths are configured and exist.
        
        Returns:
            True if all paths exist, False otherwise
            
        Raises:
            RuntimeError: If required environment variables are not set.
            
        Note:
            Module 2 (interface_cli.py) must treat a False return as fatal
            and abort execution immediately. This validation is non-negotiable
            for pipeline integrity.
        """
        # Check that required environment variables are set
        required_env_vars = {
            "FTMY_BASE_DIR": PipelineConfig.BASE_DIR,
            "FTMY_PREDICTION_PATH": PipelineConfig.PREDICTION_PATH,
            "FTMY_DAILY_SUMMARIES_PATH": PipelineConfig.DAILY_SUMMARIES_PATH,
        }
        
        missing_env_vars = []
        for env_var, path_value in required_env_vars.items():
            if path_value is None:
                missing_env_vars.append(env_var)
        
        if missing_env_vars:
            error_msg = (
                "ERROR: Required environment variables are not set:\n"
            )
            for env_var in missing_env_vars:
                error_msg += f"  - {env_var}\n"
            error_msg += (
                "\nThese environment variables must be set before running the pipeline.\n"
                "Large external datasets (>20 GB) must be downloaded manually and placed\n"
                "in the directories specified by these environment variables.\n"
                "\nExample setup (Windows PowerShell):\n"
                "  $env:FTMY_BASE_DIR = \"D:\\Data\\SyntheticWeatherFile\"\n"
                "  $env:FTMY_PREDICTION_PATH = \"D:\\Data\\SynthticWeatherFile\\prediction\"\n"
                "  $env:FTMY_DAILY_SUMMARIES_PATH = \"D:\\Data\\SyntheticWeatherFile\\historical_daily_climate_variables\"\n"
            )
            print(error_msg)
            raise RuntimeError("Required environment variables are not set. See error message above.")
        
        # Check that paths exist
        required_paths = {
            "Historical Base": PipelineConfig.BASE_DIR,
            "Prediction Path": PipelineConfig.PREDICTION_PATH,
            "Daily Summaries": PipelineConfig.DAILY_SUMMARIES_PATH,
        }
        
        missing = []
        for name, path in required_paths.items():
            if not os.path.exists(path):
                missing.append(f"{name}: {path}")
        
        if missing:
            print("ERROR: Missing required input directories:")
            for path in missing:
                print(f"  - {path}")
            return False
        
        return True


# ============================================================================
# Convenience Accessors
# ============================================================================

# Global results path accessor (for backward compatibility)
# Note: Use PipelineConfig.get_results_path() for dynamic resolution
RESULTS_PATH = PipelineConfig.get_results_path()
