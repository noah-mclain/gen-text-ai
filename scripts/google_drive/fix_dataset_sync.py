#!/usr/bin/env python3
"""
Fix Dataset Sync Script

This script finds and syncs datasets that were previously processed but not successfully 
synced to Google Drive. It can also scan temporary directories for processed datasets.
Specifically looks for HuggingFace datasets saved in Arrow format with dataset_info.json
and .arrow files.
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
        
        # Log directory contents for debugging
        try:
            contents = os.listdir(base_path)
            logger.info(f"Directory contents of {base_path}: {contents}")
        except Exception as e:
            logger.warning(f"Could not list directory contents: {e}")
        
        # Look for directories with the "_processed" suffix (first pattern)
        processed_dirs = glob.glob(os.path.join(base_path, "*_processed"))
        logger.info(f"Found {len(processed_dirs)} *_processed directories")
        
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
        
        # Look for directories matching dataset names directly
        dataset_dirs_found = 0
        for name in dataset_names:
            dataset_dir = os.path.join(base_path, name)
            if os.path.isdir(dataset_dir) and dataset_dir not in processed_dirs:
                processed_dirs.append(dataset_dir)
                dataset_dirs_found += 1
                
        logger.info(f"Found {dataset_dirs_found} additional directories matching dataset names")
        
        # Check one level deeper for processed directories (e.g. temp_dir/processed/dataset)
        subdirs = []
        try:
            for item in os.listdir(base_path):
                full_path = os.path.join(base_path, item)
                if os.path.isdir(full_path):
                    subdirs.append(full_path)
        except Exception as e:
            logger.warning(f"Error listing subdirectories: {e}")
            
        # Look in subdirectories
        for subdir in subdirs:
            try:
                logger.info(f"Checking subdirectory: {subdir}")
                subdir_contents = os.listdir(subdir)
                logger.info(f"Subdirectory contents: {subdir_contents}")
                
                # Check for dataset directories
                for name in dataset_names:
                    dataset_dir = os.path.join(subdir, name)
                    if os.path.isdir(dataset_dir) and dataset_dir not in processed_dirs:
                        processed_dirs.append(dataset_dir)
                        logger.info(f"Found dataset {name} in subdirectory {subdir}")
            except Exception as e:
                logger.warning(f"Error checking subdirectory {subdir}: {e}")
                
        # Even more aggressive search - look for any files that look like they might be dataset files
        # This might find false positives but better to find too many than miss real datasets
        if not processed_dirs:
            logger.info("No datasets found yet, trying more aggressive search...")
            try:
                # Search for dataset_info.json files and .arrow files which are indicators of HuggingFace datasets
                for root, dirs, files in os.walk(base_path):
                    has_dataset_info = "dataset_info.json" in files
                    has_arrow_files = any(f.endswith('.arrow') for f in files)
                    
                    if has_dataset_info or has_arrow_files:
                        # This is likely a dataset directory
                        logger.info(f"Found potential dataset directory: {root}")
                        if has_dataset_info:
                            logger.info(f"  - Contains dataset_info.json")
                        if has_arrow_files:
                            arrow_files = [f for f in files if f.endswith('.arrow')]
                            logger.info(f"  - Contains {len(arrow_files)} Arrow files: {arrow_files}")
                        processed_dirs.append(root)
                        
                    for d in dirs:
                        # Check if directory name contains dataset name
                        for dataset_name in dataset_names:
                            if dataset_name.lower() in d.lower():
                                full_path = os.path.join(root, d)
                                logger.info(f"Found potential dataset directory by name: {full_path}")
                                processed_dirs.append(full_path)
            except Exception as e:
                logger.warning(f"Error during aggressive search: {e}")
        
        # Add them to the dataset_dirs dictionary
        for dir_path in processed_dirs:
            name = os.path.basename(dir_path)
            if name.endswith("_processed"):
                name = name[:-10]  # Remove "_processed" suffix
            dataset_dirs[name] = dir_path
            
            # Log what we found in this directory
            try:
                arrow_files = glob.glob(os.path.join(dir_path, "*.arrow")) + \
                             glob.glob(os.path.join(dir_path, "**/*.arrow"))
                has_dataset_info = os.path.exists(os.path.join(dir_path, "dataset_info.json"))
                
                logger.info(f"Added dataset: {name} at {dir_path}")
                if has_dataset_info:
                    logger.info(f"  - Contains dataset_info.json")
                if arrow_files:
                    logger.info(f"  - Contains {len(arrow_files)} Arrow files")
            except Exception as e:
                logger.warning(f"Error checking dataset contents: {e}")
    
    logger.info(f"Total datasets found: {len(dataset_dirs)}")
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
                total_files = 0
                arrow_files = 0
                other_files = 0
                size_mb = 0
                for root, _, files in os.walk(path):
                    for file in files:
                        total_files += 1
                        if file.endswith('.arrow'):
                            arrow_files += 1
                        else:
                            other_files += 1
                        try:
                            size_mb += os.path.getsize(os.path.join(root, file)) / (1024 * 1024)
                        except Exception:
                            pass
                
                logger.info(f"Syncing dataset {name} ({total_files} files, {arrow_files} Arrow files, {size_mb:.2f} MB)")
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
    temp_base_dirs = [tempfile.gettempdir(), "/tmp", "/var/tmp", "/notebooks/tmp"]
    for base_dir in temp_base_dirs:
        if not os.path.exists(base_dir):
            continue
            
        # Find directories with dataset_processing prefix
        temp_dirs.extend(glob.glob(os.path.join(base_dir, "dataset_processing*")))
        
        # Also look for temp_datasets directories
        temp_dirs.extend(glob.glob(os.path.join(base_dir, "temp_dataset*")))
    
    # Also check for temp directories in the project root
    project_temp_dirs = glob.glob(os.path.join(project_root, "temp_dataset*"))
    temp_dirs.extend(project_temp_dirs)
    
    # Add current and parent directories of /tmp/dataset_processing* (sometimes datasets end up here)
    for base_dir in ["/tmp", tempfile.gettempdir()]:
        if os.path.exists(base_dir):
            temp_dirs.append(base_dir)
    
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
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")
    
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