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
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent  # Go up two levels from utilities to project root
sys.path.append(str(project_root))

# Also add notebooks directory to path if we're in Paperspace
notebook_path = Path("/notebooks")
if notebook_path.exists():
    sys.path.append(str(notebook_path))
    # Try to create symlinks if needed
    src_dir = notebook_path / "src"
    scripts_dir = notebook_path / "scripts"
    if not src_dir.exists():
        try:
            os.makedirs(src_dir, exist_ok=True)
            if (project_root / "src").exists():
                for item in (project_root / "src").iterdir():
                    if not (src_dir / item.name).exists():
                        logger.info(f"Creating symlink for {item.name} in /notebooks/src")
                        os.symlink(item, src_dir / item.name)
        except Exception as e:
            logger.warning(f"Failed to create src symlinks: {e}")
    
    if not scripts_dir.exists():
        try:
            os.makedirs(scripts_dir, exist_ok=True)
            if (project_root / "scripts").exists():
                for item in (project_root / "scripts").iterdir():
                    if not (scripts_dir / item.name).exists():
                        logger.info(f"Creating symlink for {item.name} in /notebooks/scripts")
                        os.symlink(item, scripts_dir / item.name)
        except Exception as e:
            logger.warning(f"Failed to create scripts symlinks: {e}")

def ensure_feature_extractor_module():
    """Ensure the feature extractor module exists and is properly set up."""
    feature_extractor_paths = [
        project_root / "src" / "data" / "processors" / "feature_extractor.py",
        Path("/notebooks/src/data/processors/feature_extractor.py") if Path("/notebooks").exists() else None
    ]
    
    for feature_extractor_path in feature_extractor_paths:
        if feature_extractor_path and feature_extractor_path.exists():
            logger.info(f"Feature extractor module found at {feature_extractor_path}")
            
            # Verify it can be imported
            try:
                # Add the src directory to the path
                src_path = str(feature_extractor_path.parent.parent.parent)
                if src_path not in sys.path:
                    sys.path.append(src_path)

                # Now try to import
                try:
                    from data.processors.feature_extractor import FeatureExtractor
                    logger.info("Successfully imported FeatureExtractor class")
                    return True
                except ImportError:
                    # Try another import path
                    sys.path.append(str(project_root))
                    from src.data.processors.feature_extractor import FeatureExtractor
                    logger.info("Successfully imported FeatureExtractor class from src.data.processors")
                    return True
            except ImportError as e:
                logger.warning(f"Error importing feature extractor from {feature_extractor_path}: {e}")
                # Continue to try other paths
    
    logger.error("Feature extractor module not found in any expected location")
    return False

def ensure_prepare_datasets_script():
    """Ensure the prepare_datasets_for_training.py script exists."""
    script_paths = [
        project_root / "scripts" / "prepare_datasets_for_training.py",
        Path("/notebooks/scripts/prepare_datasets_for_training.py") if Path("/notebooks").exists() else None
    ]
    
    for script_path in script_paths:
        if script_path and script_path.exists():
            logger.info(f"Dataset preparation script found at {script_path}")
            # Make it executable
            os.chmod(script_path, 0o755)
            return True
    
    logger.warning(f"Dataset preparation script not found in any expected location")
    return False

def ensure_prepare_drive_datasets_script():
    """Ensure the prepare_drive_datasets.sh script exists."""
    script_paths = [
        project_root / "scripts" / "prepare_drive_datasets.sh",
        Path("/notebooks/scripts/prepare_drive_datasets.sh") if Path("/notebooks").exists() else None
    ]
    
    for script_path in script_paths:
        if script_path and script_path.exists():
            logger.info(f"Drive datasets script found at {script_path}")
            # Make it executable
            os.chmod(script_path, 0o755)
            return True
    
    logger.warning(f"Drive datasets script not found in any expected location")
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
    
    # Add notebooks paths if in Paperspace
    if Path("/notebooks").exists():
        dirs_to_create.extend([
            Path("/notebooks/data/processed/features"),
            Path("/notebooks/data/processed/features/deepseek_coder")
        ])
    
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