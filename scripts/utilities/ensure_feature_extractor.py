#!/usr/bin/env python3
"""
Ensure Feature Extractor Setup

This script verifies that the feature extractor module is properly installed
and sets up any necessary components if they're missing.
"""

import os
import sys
import importlib
import shutil
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def ensure_feature_extractor_module():
    """Ensure the feature extractor module exists and is properly set up."""
    feature_extractor_path = project_root / "src" / "data" / "processors" / "feature_extractor.py"
    
    if feature_extractor_path.exists():
        logger.info(f"Feature extractor module found at {feature_extractor_path}")
        
        # Verify it can be imported
        try:
            from src.data.processors.feature_extractor import FeatureExtractor
            logger.info("Successfully imported FeatureExtractor class")
            return True
        except ImportError as e:
            logger.error(f"Error importing feature extractor: {e}")
            return False
    else:
        logger.warning(f"Feature extractor module not found at {feature_extractor_path}")
        return False

def ensure_prepare_datasets_script():
    """Ensure the prepare_datasets_for_training.py script exists."""
    script_path = project_root / "scripts" / "prepare_datasets_for_training.py"
    
    if script_path.exists():
        logger.info(f"Dataset preparation script found at {script_path}")
        # Make it executable
        os.chmod(script_path, 0o755)
        return True
    else:
        logger.warning(f"Dataset preparation script not found at {script_path}")
        return False

def ensure_prepare_drive_datasets_script():
    """Ensure the prepare_drive_datasets.sh script exists."""
    script_path = project_root / "scripts" / "prepare_drive_datasets.sh"
    
    if script_path.exists():
        logger.info(f"Drive datasets script found at {script_path}")
        # Make it executable
        os.chmod(script_path, 0o755)
        return True
    else:
        logger.warning(f"Drive datasets script not found at {script_path}")
        return False

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        "transformers",
        "datasets",
        "torch",
        "tqdm"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Missing required packages: {', '.join(missing_packages)}")
        logger.warning("Please install them with: pip install " + " ".join(missing_packages))
        return False
    
    logger.info("All required dependencies are installed")
    return True

def create_output_dirs():
    """Create necessary output directories."""
    dirs_to_create = [
        project_root / "data" / "processed" / "features",
        project_root / "data" / "processed" / "features" / "deepseek_coder"
    ]
    
    for directory in dirs_to_create:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def main():
    """Main function to verify and set up feature extractor components."""
    logger.info("Checking feature extractor setup...")
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Missing dependencies. Please install them before proceeding.")
        return False
    
    # Check feature extractor module
    feature_extractor_ok = ensure_feature_extractor_module()
    
    # Check scripts
    prepare_datasets_ok = ensure_prepare_datasets_script()
    prepare_drive_datasets_ok = ensure_prepare_drive_datasets_script()
    
    # Create output directories
    create_output_dirs()
    
    # Summary
    if feature_extractor_ok and prepare_datasets_ok and prepare_drive_datasets_ok:
        logger.info("✅ Feature extractor setup is complete and ready to use")
        return True
    else:
        logger.warning("⚠️ Feature extractor setup is incomplete")
        logger.warning("Please check the logs for details on what components are missing")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 