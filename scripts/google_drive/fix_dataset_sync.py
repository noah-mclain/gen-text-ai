#!/usr/bin/env python3
"""
Fix Dataset Sync Script

This script finds and syncs datasets that were previously processed but not successfully 
synced to Google Drive. It can also scan temporary directories for processed datasets.
"""

import os
import sys
import logging
import argparse
import glob
import tempfile
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add project root to Python path
current_file = os.path.abspath(__file__)
scripts_dir = os.path.dirname(os.path.dirname(current_file))
project_root = os.path.dirname(scripts_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    logger.info(f"Added project root to Python path: {project_root}")

# Try different import paths for Google Drive manager
drive_utils_imported = False

# Try direct import from src.utils first (main implementation)
try:
    from src.utils.google_drive_manager import sync_to_drive, configure_sync_method
    logger.info("Successfully imported Drive manager from src.utils.google_drive_manager")
    drive_utils_imported = True
except ImportError:
    pass

# If not imported yet, try scripts.google_drive path 
if not drive_utils_imported:
    try:
        from scripts.google_drive.google_drive_manager import sync_to_drive, configure_sync_method
        logger.info("Successfully imported Drive manager from scripts.google_drive.google_drive_manager")
        drive_utils_imported = True
    except ImportError:
        pass

# If still not imported, try direct import from utils dir
if not drive_utils_imported:
    try:
        utils_path = os.path.join(project_root, 'src', 'utils')
        if utils_path not in sys.path:
            sys.path.append(utils_path)
        from google_drive_manager import sync_to_drive, configure_sync_method
        logger.info("Successfully imported Drive manager from utils path")
        drive_utils_imported = True
    except ImportError as e:
        logger.error(f"Failed to import Google Drive manager: {e}")
        logger.error("Please make sure google_drive_manager.py exists and is properly configured")
        sys.exit(1)

def find_processed_datasets(paths_to_check):
    """Find processed datasets in the provided paths."""
    dataset_dirs = {}
    
    for base_path in paths_to_check:
        if not os.path.exists(base_path):
            logger.warning(f"Path does not exist: {base_path}")
            continue
            
        logger.info(f"Searching for datasets in: {base_path}")
        
        # Look for directories with the "_processed" suffix
        processed_dirs = glob.glob(os.path.join(base_path, "*_processed"))
        logger.debug(f"Found {len(processed_dirs)} processed directories")
        
        # Also look for dataset directories with known names
        # Check using dataset_config.json to find dataset names
        config_path = os.path.join(project_root, "config", "dataset_config.json")
        dataset_names = []
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    dataset_config = json.load(f)
                    dataset_names = list(dataset_config.keys())
                    logger.info(f"Found {len(dataset_names)} dataset names in config")
            except Exception as e:
                logger.warning(f"Error loading dataset config: {e}")
        
        # Look for directories matching dataset names
        for name in dataset_names:
            dataset_dir = os.path.join(base_path, name)
            if os.path.isdir(dataset_dir) and dataset_dir not in processed_dirs:
                processed_dirs.append(dataset_dir)
        
        # Add them to the dataset_dirs dictionary
        for dir_path in processed_dirs:
            name = os.path.basename(dir_path)
            if name.endswith("_processed"):
                name = name[:-10]  # Remove "_processed" suffix
            dataset_dirs[name] = dir_path
    
    return dataset_dirs

def sync_datasets_to_drive(dataset_dirs, drive_folder, delete_source=False):
    """Sync dataset directories to Google Drive."""
    results = {"success": [], "failure": []}
    
    if not dataset_dirs:
        logger.warning("No datasets found to sync")
        return results
    
    logger.info(f"Found {len(dataset_dirs)} datasets to sync to Google Drive folder '{drive_folder}'")
    
    # Sync each dataset to Drive
    for name, path in dataset_dirs.items():
        try:
            # Calculate stats for logging
            try:
                num_files = sum([len(files) for _, _, files in os.walk(path)])
                size_mb = sum(os.path.getsize(os.path.join(root, file)) 
                             for root, _, files in os.walk(path) 
                             for file in files) / (1024 * 1024)
                logger.info(f"Syncing dataset {name} ({num_files} files, {size_mb:.2f} MB)")
            except Exception as e:
                logger.warning(f"Error calculating directory stats: {e}")
                logger.info(f"Syncing dataset {name}")
            
            # Sync to Drive
            success = sync_to_drive(
                path,
                drive_folder,
                delete_source=delete_source,
                update_only=False
            )
            
            if success:
                logger.info(f"Successfully synced dataset {name} to Drive")
                results["success"].append(name)
            else:
                logger.error(f"Failed to sync dataset {name} to Drive")
                results["failure"].append(name)
        except Exception as e:
            logger.error(f"Error syncing dataset {name}: {e}")
            results["failure"].append(name)
    
    return results

def find_temp_directories():
    """Find temporary directories that may contain processed datasets."""
    temp_dirs = []
    
    # Check common temp directories for dataset_processing folders
    temp_base_dirs = [tempfile.gettempdir(), "/tmp"]
    for base_dir in temp_base_dirs:
        if not os.path.exists(base_dir):
            continue
            
        # Find directories with dataset_processing prefix
        temp_dirs.extend(glob.glob(os.path.join(base_dir, "dataset_processing*")))
    
    return temp_dirs

def main():
    parser = argparse.ArgumentParser(
        description="Fix Dataset Sync - Find and sync datasets to Google Drive"
    )
    parser.add_argument(
        "--scan_paths", nargs="+", default=["data/processed", "temp_datasets*"],
        help="Paths to scan for processed datasets"
    )
    parser.add_argument(
        "--drive_folder", type=str, default="datasets",
        help="Google Drive folder to sync datasets to"
    )
    parser.add_argument(
        "--scan_temp", action="store_true",
        help="Scan temporary directories for processed datasets"
    )
    parser.add_argument(
        "--delete_source", action="store_true",
        help="Delete source files after successful sync"
    )
    parser.add_argument(
        "--use_rclone", action="store_true",
        help="Use rclone for syncing (if available)"
    )
    
    args = parser.parse_args()
    
    # Configure sync method
    configure_sync_method(
        use_rclone=args.use_rclone,
        base_dir=args.drive_folder
    )
    
    # Collect all paths to check
    paths_to_check = []
    
    # Add all scan paths
    for path_pattern in args.scan_paths:
        # Expand glob patterns
        if "*" in path_pattern or "?" in path_pattern:
            # Handle relative paths from project root
            if not os.path.isabs(path_pattern):
                path_pattern = os.path.join(project_root, path_pattern)
            paths_to_check.extend(glob.glob(path_pattern))
        else:
            # Handle relative paths from project root
            if not os.path.isabs(path_pattern):
                path = os.path.join(project_root, path_pattern)
            else:
                path = path_pattern
            paths_to_check.append(path)
    
    # Add temporary directories if requested
    if args.scan_temp:
        temp_dirs = find_temp_directories()
        logger.info(f"Found {len(temp_dirs)} temporary directories that may contain datasets")
        
        for temp_dir in temp_dirs:
            # Check for processed subdirectory
            processed_dir = os.path.join(temp_dir, "processed")
            if os.path.exists(processed_dir):
                paths_to_check.append(processed_dir)
            else:
                # The temp dir itself might contain the datasets
                paths_to_check.append(temp_dir)
    
    # Find processed datasets
    dataset_dirs = find_processed_datasets(paths_to_check)
    
    # Sync datasets to Google Drive
    results = sync_datasets_to_drive(
        dataset_dirs,
        args.drive_folder,
        delete_source=args.delete_source
    )
    
    # Print summary
    logger.info("\n--- Sync Summary ---")
    logger.info(f"Drive Folder: {args.drive_folder}")
    logger.info(f"Successfully synced {len(results['success'])} datasets")
    logger.info(f"Failed to sync {len(results['failure'])} datasets")
    
    if results['success']:
        logger.info("Successful syncs:")
        for name in results['success']:
            logger.info(f"  - {name}")
    
    if results['failure']:
        logger.warning("Failed syncs:")
        for name in results['failure']:
            logger.warning(f"  - {name}")
    
    return 0 if not results['failure'] else 1

if __name__ == "__main__":
    sys.exit(main()) 