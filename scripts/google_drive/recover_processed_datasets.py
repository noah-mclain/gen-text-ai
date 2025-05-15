#!/usr/bin/env python3
"""
Recover Processed Datasets Script

This script helps recover processed datasets that might be lost in temporary directories
by directly searching for arrow files and dataset_info.json files that indicate HuggingFace
datasets saved with save_to_disk(), then copying them to a recovery directory and 
syncing them to Google Drive.
"""

import os
import sys
import glob
import logging
import argparse
import shutil
import json
from pathlib import Path
import subprocess
import tempfile
import re

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

def find_dataset_files():
    """Find dataset files in temporary directories."""
    # Get dataset names from config
    dataset_names = []
    config_path = os.path.join(project_root, "config", "dataset_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                dataset_config = json.load(f)
                dataset_names = list(dataset_config.keys())
        except Exception as e:
            logger.warning(f"Error loading dataset config: {e}")
            
    if not dataset_names:
        # Default dataset names if config not found
        dataset_names = [
            "codesearchnet_python", "codesearchnet_java", "codesearchnet_javascript",
            "codesearchnet_php", "codesearchnet_ruby", "codesearchnet_go",
            "codesearchnet_all", "code_alpaca", "mbpp", "humaneval", "codeparrot",
            "instruct_code", "the_stack_filtered"
        ]
        
    logger.info(f"Looking for {len(dataset_names)} dataset types")
    
    # Places to search
    temp_dirs = [
        "/tmp",
        tempfile.gettempdir(),
        "/var/tmp",
        "/notebooks/tmp",
        os.path.join(project_root, "temp_datasets_*"),
        os.path.join(project_root, "data", "processed")
    ]
    
    # Find all processed dataset directories and arrow files
    dataset_dirs = {}
    
    # Look for processed dataset directories
    for temp_dir in temp_dirs:
        if "*" in temp_dir:
            # Handle glob patterns
            matching_dirs = glob.glob(temp_dir)
            for match_dir in matching_dirs:
                if not os.path.exists(match_dir):
                    continue
                
                logger.info(f"Searching in directory: {match_dir}")
                
                # Look for _processed directories or dataset_name directories
                for dataset_name in dataset_names:
                    # Try both naming patterns
                    processed_dir_patterns = [
                        os.path.join(match_dir, f"{dataset_name}_processed"),
                        os.path.join(match_dir, dataset_name),
                        os.path.join(match_dir, "processed", f"{dataset_name}_processed"),
                        os.path.join(match_dir, "processed", dataset_name)
                    ]
                    
                    for dir_pattern in processed_dir_patterns:
                        if os.path.isdir(dir_pattern):
                            # Check if this looks like a dataset directory
                            # Datasets saved with .save_to_disk() typically have dataset_info.json and .arrow files
                            has_dataset_info = os.path.exists(os.path.join(dir_pattern, "dataset_info.json"))
                            has_arrow_files = bool(glob.glob(os.path.join(dir_pattern, "*.arrow")) or 
                                                  glob.glob(os.path.join(dir_pattern, "**/*.arrow")))
                            
                            if has_dataset_info or has_arrow_files:
                                if dataset_name not in dataset_dirs:
                                    dataset_dirs[dataset_name] = []
                                if dir_pattern not in dataset_dirs[dataset_name]:
                                    dataset_dirs[dataset_name].append(dir_pattern)
                                    logger.info(f"Found dataset directory for {dataset_name}: {dir_pattern}")
                                    if has_dataset_info:
                                        logger.info(f"  - Contains dataset_info.json")
                                    if has_arrow_files:
                                        arrow_files = glob.glob(os.path.join(dir_pattern, "*.arrow")) + \
                                                     glob.glob(os.path.join(dir_pattern, "**/*.arrow"))
                                        logger.info(f"  - Contains {len(arrow_files)} Arrow files")
        elif os.path.exists(temp_dir):
            logger.info(f"Searching in directory: {temp_dir}")
            
            # Use find command for deeper search of processed directories
            try:
                # Look for dataset_info.json which indicates a dataset directory
                cmd = ["find", temp_dir, "-name", "dataset_info.json", "-type", "f"]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    info_files = result.stdout.strip().split("\n")
                    for file in info_files:
                        if not file:  # Skip empty lines
                            continue
                        
                        # Get the directory containing the dataset_info.json
                        dir_path = os.path.dirname(file)
                        
                        # Try to determine dataset name from directory path
                        dataset_name = None
                        for name in dataset_names:
                            if name in dir_path:
                                dataset_name = name
                                break
                        
                        if dataset_name:
                            if dataset_name not in dataset_dirs:
                                dataset_dirs[dataset_name] = []
                            if dir_path not in dataset_dirs[dataset_name]:
                                dataset_dirs[dataset_name].append(dir_path)
                                logger.info(f"Found dataset directory for {dataset_name}: {dir_path}")
                                logger.info(f"  - Contains dataset_info.json")
                
                # Also look for .arrow files
                cmd = ["find", temp_dir, "-name", "*.arrow", "-type", "f"]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    arrow_files = result.stdout.strip().split("\n")
                    for file in arrow_files:
                        if not file:  # Skip empty lines
                            continue
                        
                        # Get the directory containing the arrow file
                        dir_path = os.path.dirname(file)
                        
                        # Try to determine dataset name from file path
                        dataset_name = None
                        for name in dataset_names:
                            if name in dir_path or name in file:
                                dataset_name = name
                                break
                        
                        if dataset_name:
                            if dataset_name not in dataset_dirs:
                                dataset_dirs[dataset_name] = []
                            if dir_path not in dataset_dirs[dataset_name]:
                                dataset_dirs[dataset_name].append(dir_path)
                                logger.info(f"Found dataset directory for {dataset_name}: {dir_path}")
                                logger.info(f"  - Contains arrow file: {os.path.basename(file)}")
            except Exception as e:
                logger.warning(f"Error running find command: {e}")
    
    # Count directories found
    total_dirs = sum(len(dirs) for dirs in dataset_dirs.values())
    logger.info(f"Found {total_dirs} dataset directories for {len(dataset_dirs)} datasets")
    
    # Print found datasets
    for dataset_name, dirs in dataset_dirs.items():
        if dirs:
            logger.info(f"Found {len(dirs)} directories for dataset {dataset_name}: {dirs}")
    
    return dataset_dirs

def recover_datasets(dataset_dirs, recovery_dir, drive_folder):
    """Recover datasets by copying files to a recovery directory and syncing to Drive."""
    if not dataset_dirs:
        logger.warning("No dataset directories found to recover")
        return {}
    
    # Create recovery directory
    os.makedirs(recovery_dir, exist_ok=True)
    
    # Dictionary to track recovered datasets
    recovered_datasets = {}
    
    # Copy files to recovery directory
    for dataset_name, dirs in dataset_dirs.items():
        if not dirs:
            continue
            
        logger.info(f"Recovering dataset {dataset_name}...")
        
        # Create dataset directory
        dataset_dir = os.path.join(recovery_dir, f"{dataset_name}_processed")
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Copy each found directory
        for dir_path in dirs:
            try:
                # Calculate stats for logging
                file_count = 0
                size_mb = 0
                arrow_count = 0
                for root, _, files in os.walk(dir_path):
                    for file in files:
                        file_count += 1
                        if file.endswith('.arrow'):
                            arrow_count += 1
                        try:
                            size_mb += os.path.getsize(os.path.join(root, file)) / (1024 * 1024)
                        except:
                            pass
                
                logger.info(f"Copying {file_count} files ({arrow_count} Arrow files, {size_mb:.2f} MB) from {dir_path} to {dataset_dir}")
                
                # Copy directory contents
                for item in os.listdir(dir_path):
                    source = os.path.join(dir_path, item)
                    dest = os.path.join(dataset_dir, item)
                    
                    if os.path.isdir(source):
                        # Copy directory and contents
                        shutil.copytree(source, dest, dirs_exist_ok=True)
                        logger.info(f"Copied directory: {item}")
                    else:
                        # Copy file
                        shutil.copy2(source, dest)
                        if item == 'dataset_info.json':
                            logger.info(f"Copied dataset_info.json")
                        elif item.endswith('.arrow'):
                            logger.info(f"Copied Arrow file: {item}")
                
                # Track recovered datasets
                if dataset_name not in recovered_datasets:
                    recovered_datasets[dataset_name] = []
                recovered_datasets[dataset_name].append(dir_path)
                logger.info(f"Successfully copied dataset {dataset_name} from {dir_path}")
            except Exception as e:
                logger.error(f"Error copying from {dir_path}: {e}")
    
    # Sync recovered datasets to Google Drive
    if drive_utils_imported:
        for dataset_name, dirs in recovered_datasets.items():
            if dirs:
                dataset_dir = os.path.join(recovery_dir, f"{dataset_name}_processed")
                
                # Check if the dataset directory exists and has content
                if os.path.exists(dataset_dir) and os.listdir(dataset_dir):
                    logger.info(f"Syncing recovered dataset {dataset_name} to Google Drive folder '{drive_folder}'")
                    
                    success = sync_to_drive(
                        dataset_dir,
                        drive_folder,
                        delete_source=False,
                        update_only=False
                    )
                    
                    if success:
                        logger.info(f"Successfully synced recovered dataset {dataset_name} to Drive")
                    else:
                        logger.error(f"Failed to sync recovered dataset {dataset_name} to Drive")
    else:
        logger.warning("Google Drive sync not available, datasets only recovered locally")
    
    return recovered_datasets

def main():
    parser = argparse.ArgumentParser(
        description="Recover processed datasets from temporary directories"
    )
    parser.add_argument(
        "--recovery_dir", type=str, default="recovered_datasets",
        help="Directory to store recovered datasets"
    )
    parser.add_argument(
        "--drive_folder", type=str, default="datasets",
        help="Google Drive folder to sync recovered datasets to"
    )
    parser.add_argument(
        "--use_rclone", action="store_true",
        help="Use rclone for syncing (if available)"
    )
    
    args = parser.parse_args()
    
    # Configure sync method
    if drive_utils_imported:
        configure_sync_method(
            use_rclone=args.use_rclone,
            base_dir=args.drive_folder
        )
    
    # Find dataset files
    dataset_dirs = find_dataset_files()
    
    # Recover datasets
    recovered_datasets = recover_datasets(dataset_dirs, args.recovery_dir, args.drive_folder)
    
    # Print summary
    logger.info("\n--- Recovery Summary ---")
    logger.info(f"Recovery Directory: {os.path.abspath(args.recovery_dir)}")
    logger.info(f"Google Drive Folder: {args.drive_folder}")
    logger.info(f"Recovered {len(recovered_datasets)} datasets")
    
    for dataset_name, dirs in recovered_datasets.items():
        logger.info(f"  - {dataset_name}: {len(dirs)} directories")
    
    if recovered_datasets:
        logger.info("\nRecovery completed successfully!")
    else:
        logger.warning("\nNo datasets were recovered. Your datasets might be lost or in a different location.")
        logger.warning("Try manually finding arrow files or dataset directories with:")
        logger.warning("  find /tmp -name \"dataset_info.json\" -type f")
        logger.warning("  find /tmp -name \"*.arrow\" -type f")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 