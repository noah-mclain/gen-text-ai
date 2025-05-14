#!/usr/bin/env python3
"""
Manual Dataset Sync Script

This script manually syncs all processed datasets to Google Drive.
Use this when automatic sync fails during preprocessing.
"""

import os
import sys
import argparse
import logging
import glob
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(scripts_dir)
sys.path.append(project_root)

# Try to import Drive functions from all possible locations
try:
    # First try direct import from reorganized path
    from scripts.google_drive.google_drive_manager import sync_to_drive, test_authentication, configure_sync_method
    logger.info("Successfully imported Drive manager from scripts.google_drive")
except ImportError:
    try:
        # Next try the src.utils path
        from src.utils.google_drive_manager import sync_to_drive, test_authentication, configure_sync_method
        logger.info("Successfully imported Drive manager from src.utils")
    except ImportError:
        # Lastly try direct import from utils dir
        try:
            utils_path = os.path.join(project_root, 'src', 'utils')
            if utils_path not in sys.path:
                sys.path.append(utils_path)
            from google_drive_manager import sync_to_drive, test_authentication, configure_sync_method
            logger.info("Successfully imported Drive manager from utils path")
        except ImportError:
            logger.error("CRITICAL: Could not import Google Drive functions. Syncing will not work.")
            sys.exit(1)

def sync_all_datasets(data_dir, drive_dir="preprocessed", dry_run=False):
    """
    Sync all processed datasets to Google Drive.
    
    Args:
        data_dir: Directory containing processed datasets
        drive_dir: Target directory in Google Drive
        dry_run: If True, only print what would be synced without transferring
    """
    logger.info(f"Checking for processed datasets in {data_dir}")
    
    # Test authentication first
    if not test_authentication():
        logger.error("Google Drive authentication failed. Please run setup_google_drive.py first.")
        return False
    
    # Configure sync method (rclone preferred if available)
    try:
        import subprocess
        result = subprocess.run(
            ["rclone", "version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        use_rclone = result.returncode == 0
    except Exception:
        use_rclone = False
    
    configure_sync_method(use_rclone=use_rclone, base_dir="DeepseekCoder")
    logger.info(f"Configured Drive sync using {'rclone' if use_rclone else 'Google Drive API'}")
    
    # Find all processed datasets
    processed_dirs = []
    
    # Look for *_processed directories
    for entry in os.listdir(data_dir):
        full_path = os.path.join(data_dir, entry)
        if os.path.isdir(full_path) and "_processed" in entry:
            processed_dirs.append(full_path)
    
    # Look for interim directories too
    for entry in os.listdir(data_dir):
        full_path = os.path.join(data_dir, entry)
        if os.path.isdir(full_path) and "_interim_" in entry:
            processed_dirs.append(full_path)
    
    logger.info(f"Found {len(processed_dirs)} processed dataset directories")
    
    # Check if paperspace notebooks directory has more datasets
    paperspace_dir = "/notebooks/data/processed"
    if os.path.exists(paperspace_dir) and os.path.isdir(paperspace_dir):
        logger.info(f"Checking Paperspace directory: {paperspace_dir}")
        for entry in os.listdir(paperspace_dir):
            full_path = os.path.join(paperspace_dir, entry)
            if os.path.isdir(full_path) and ("_processed" in entry or "_interim_" in entry):
                processed_dirs.append(full_path)
        logger.info(f"Found total of {len(processed_dirs)} processed dataset directories")
    
    # Sort dirs for consistent output
    processed_dirs = sorted(processed_dirs)
    
    # Print what we found
    logger.info("Datasets to sync:")
    for i, path in enumerate(processed_dirs, 1):
        logger.info(f"{i}. {os.path.basename(path)}")
    
    # Confirm before proceeding if not dry run
    if not dry_run:
        if len(processed_dirs) == 0:
            logger.error("No processed datasets found to sync")
            return False
        
        print("\nReady to sync these datasets to Google Drive.")
        confirm = input("Proceed with sync? [y/N]: ").strip().lower()
        if confirm != 'y':
            logger.info("Sync cancelled by user")
            return False
    
    # Sync each dataset
    success_count = 0
    for dataset_path in processed_dirs:
        dataset_name = os.path.basename(dataset_path)
        logger.info(f"Syncing {dataset_name} to Google Drive...")
        
        if dry_run:
            logger.info(f"DRY RUN: Would sync {dataset_path} to {drive_dir}")
            success_count += 1
            continue
        
        success = sync_to_drive(
            dataset_path,
            drive_dir,
            delete_source=False,
            update_only=False
        )
        
        if success:
            logger.info(f"Successfully synced {dataset_name} to Google Drive")
            success_count += 1
        else:
            logger.error(f"Failed to sync {dataset_name} to Google Drive")
    
    # Report results
    if not dry_run:
        logger.info(f"Sync complete: {success_count}/{len(processed_dirs)} datasets synced successfully")
    else:
        logger.info(f"DRY RUN: Would sync {success_count} datasets to Google Drive")
    
    return success_count > 0

def main():
    parser = argparse.ArgumentParser(description="Manually sync processed datasets to Google Drive")
    parser.add_argument("--data-dir", default="data/processed", 
                       help="Directory containing processed datasets (default: data/processed)")
    parser.add_argument("--drive-dir", default="preprocessed",
                       help="Target directory in Google Drive (default: preprocessed)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Only print what would be synced without transferring")
    
    args = parser.parse_args()
    
    # Check if local directory exists
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        return 1
    
    success = sync_all_datasets(args.data_dir, args.drive_dir, args.dry_run)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 