#!/usr/bin/env python3
"""
Recover Processed Datasets Script

This script helps recover processed datasets that might be lost in temporary directories
by directly searching for .jsonl files that match dataset patterns and copying them
to a recovery directory, then syncing them to Google Drive.
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
    
    # Find all jsonl files
    dataset_files = {}
    
    # Add dataset names to search patterns
    patterns = []
    for name in dataset_names:
        patterns.append(f"{name}*.jsonl")
        patterns.append(f"*/{name}*.jsonl")
        patterns.append(f"*/{name}_processed/*.jsonl")
        patterns.append(f"*/{name}/*.jsonl")
        patterns.append(f"*/processed/{name}*.jsonl")
    
    # Search for dataset files
    for temp_dir in temp_dirs:
        if "*" in temp_dir:
            # Handle glob patterns
            matching_dirs = glob.glob(temp_dir)
            for match_dir in matching_dirs:
                if os.path.exists(match_dir):
                    for pattern in patterns:
                        files = glob.glob(os.path.join(match_dir, pattern), recursive=True)
                        for file in files:
                            # Try to determine dataset name from file path
                            dataset_name = None
                            for name in dataset_names:
                                if name in file:
                                    dataset_name = name
                                    break
                            
                            if dataset_name:
                                if dataset_name not in dataset_files:
                                    dataset_files[dataset_name] = []
                                dataset_files[dataset_name].append(file)
        elif os.path.exists(temp_dir):
            # Use find command for deeper search (more reliable than glob)
            try:
                cmd = ["find", temp_dir, "-name", "*.jsonl", "-type", "f"]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    files = result.stdout.strip().split("\n")
                    for file in files:
                        if not file:  # Skip empty lines
                            continue
                            
                        # Try to determine dataset name from file path
                        dataset_name = None
                        for name in dataset_names:
                            if name in file:
                                dataset_name = name
                                break
                            
                        if dataset_name:
                            if dataset_name not in dataset_files:
                                dataset_files[dataset_name] = []
                            dataset_files[dataset_name].append(file)
            except Exception as e:
                logger.warning(f"Error running find command: {e}")
                # Fall back to glob
                for pattern in patterns:
                    files = glob.glob(os.path.join(temp_dir, pattern), recursive=True)
                    for file in files:
                        # Try to determine dataset name from file path
                        dataset_name = None
                        for name in dataset_names:
                            if name in file:
                                dataset_name = name
                                break
                            
                        if dataset_name:
                            if dataset_name not in dataset_files:
                                dataset_files[dataset_name] = []
                            dataset_files[dataset_name].append(file)
    
    # Count files found
    total_files = sum(len(files) for files in dataset_files.values())
    logger.info(f"Found {total_files} dataset files for {len(dataset_files)} datasets")
    
    # Print found datasets
    for dataset_name, files in dataset_files.items():
        logger.info(f"Found {len(files)} files for dataset {dataset_name}")
    
    return dataset_files

def recover_datasets(dataset_files, recovery_dir, drive_folder):
    """Recover datasets by copying files to a recovery directory and syncing to Drive."""
    if not dataset_files:
        logger.warning("No dataset files found to recover")
        return {}
    
    # Create recovery directory
    os.makedirs(recovery_dir, exist_ok=True)
    
    # Dictionary to track recovered datasets
    recovered_datasets = {}
    
    # Copy files to recovery directory
    for dataset_name, files in dataset_files.items():
        logger.info(f"Recovering dataset {dataset_name}...")
        
        # Create dataset directory
        dataset_dir = os.path.join(recovery_dir, f"{dataset_name}_processed")
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Copy files
        for file in files:
            try:
                # Get base filename
                filename = os.path.basename(file)
                target_file = os.path.join(dataset_dir, filename)
                
                # Copy file
                shutil.copy2(file, target_file)
                logger.info(f"Copied {file} to {target_file}")
                
                # Track recovered datasets
                if dataset_name not in recovered_datasets:
                    recovered_datasets[dataset_name] = []
                recovered_datasets[dataset_name].append(target_file)
            except Exception as e:
                logger.error(f"Error copying file {file}: {e}")
    
    # Sync recovered datasets to Google Drive
    if drive_utils_imported:
        logger.info(f"Syncing recovered datasets to Google Drive folder '{drive_folder}'")
        for dataset_name, files in recovered_datasets.items():
            if files:
                dataset_dir = os.path.join(recovery_dir, f"{dataset_name}_processed")
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
    dataset_files = find_dataset_files()
    
    # Recover datasets
    recovered_datasets = recover_datasets(dataset_files, args.recovery_dir, args.drive_folder)
    
    # Print summary
    logger.info("\n--- Recovery Summary ---")
    logger.info(f"Recovery Directory: {os.path.abspath(args.recovery_dir)}")
    logger.info(f"Google Drive Folder: {args.drive_folder}")
    logger.info(f"Recovered {len(recovered_datasets)} datasets")
    
    for dataset_name, files in recovered_datasets.items():
        logger.info(f"  - {dataset_name}: {len(files)} files")
    
    if recovered_datasets:
        logger.info("\nRecovery completed successfully!")
    else:
        logger.warning("\nNo datasets were recovered. Your datasets might be lost or in a different location.")
        logger.warning("Try manually finding .jsonl files with 'find /tmp -name \"*.jsonl\"' command")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 