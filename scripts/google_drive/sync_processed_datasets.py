#!/usr/bin/env python3
"""
Sync Processed Datasets to Google Drive

This script finds all processed datasets in the local data directory
and syncs them to the Google Drive DeepseekCoder/data/processed/ folder.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Try to import drive sync functionality
try:
    from src.utils.google_drive_manager import (
        sync_to_drive,
        configure_sync_method,
        test_authentication
    )
except ImportError:
    logger.error("Failed to import Google Drive manager. Make sure the correct modules are installed.")
    sys.exit(1)

def sync_processed_datasets(data_dir="data/processed", notebooks_dir="/notebooks/data/processed",
                         drive_folder="data/processed", drive_base_dir="DeepseekCoder",
                         force=False, skip_auth_check=False):
    """
    Find all processed datasets and sync them to Google Drive.
    
    Args:
        data_dir: Local data directory to check for processed datasets
        notebooks_dir: Paperspace notebooks data directory to check
        drive_folder: Google Drive folder to sync to (relative to drive_base_dir)
        drive_base_dir: Base directory in Google Drive
        force: Skip confirmation prompt if True
        skip_auth_check: Skip authentication check if True
        
    Returns:
        Success count (number of datasets successfully synced)
    """
    # Check authentication first
    if not skip_auth_check:
        logger.info("Testing Google Drive authentication...")
        if not test_authentication():
            logger.error("Google Drive authentication failed. Please run setup_google_drive.py first.")
            return 0
    
    # Configure Drive sync with the specified base directory
    configure_sync_method(use_rclone=False, base_dir=drive_base_dir)
    logger.info(f"Configured Drive sync with base directory: {drive_base_dir}")
    
    # Find all processed datasets
    processed_dirs = []
    
    # Look in project data directory
    project_data_dir = os.path.join(project_root, data_dir)
    if os.path.exists(project_data_dir):
        logger.info(f"Checking for datasets in: {project_data_dir}")
        for entry in os.listdir(project_data_dir):
            full_path = os.path.join(project_data_dir, entry)
            if os.path.isdir(full_path) and entry.endswith("_processed"):
                processed_dirs.append(full_path)
                
    # Check if Paperspace notebooks directory exists and look there too
    if os.path.exists(notebooks_dir):
        logger.info(f"Checking for datasets in Paperspace directory: {notebooks_dir}")
        for entry in os.listdir(notebooks_dir):
            full_path = os.path.join(notebooks_dir, entry)
            if os.path.isdir(full_path) and entry.endswith("_processed"):
                processed_dirs.append(full_path)
    
    # Sort for consistent output
    processed_dirs = sorted(processed_dirs)
    
    # Print what we found
    logger.info(f"Found {len(processed_dirs)} processed datasets:")
    for i, path in enumerate(processed_dirs, 1):
        logger.info(f"{i}. {os.path.basename(path)}")
    
    # Confirm before proceeding if not forced
    if not force and processed_dirs:
        confirm = input("\nReady to sync these datasets to Google Drive. Proceed? [y/N]: ").strip().lower()
        if confirm != 'y':
            logger.info("Sync cancelled by user")
            return 0
    
    # Sync each dataset
    success_count = 0
    for dataset_path in processed_dirs:
        dataset_name = os.path.basename(dataset_path)
        logger.info(f"Syncing {dataset_name} to Google Drive...")
        
        try:
            success = sync_to_drive(
                dataset_path,
                drive_folder,
                delete_source=False,
                update_only=False
            )
            
            if success:
                logger.info(f"Successfully synced {dataset_name} to Google Drive")
                success_count += 1
            else:
                logger.error(f"Failed to sync {dataset_name} to Google Drive")
        except Exception as e:
            logger.error(f"Error syncing {dataset_name}: {e}")
    
    logger.info(f"Sync complete. Successfully synced {success_count}/{len(processed_dirs)} datasets to Google Drive.")
    return success_count

def main():
    parser = argparse.ArgumentParser(description="Sync processed datasets to Google Drive")
    parser.add_argument("--data-dir", default="data/processed",
                      help="Local data directory (default: data/processed)")
    parser.add_argument("--drive-folder", default="processed_data",
                      help="Folder in Google Drive to sync to (default: processed_data). Use 'preprocessed' for compatibility with training scripts.")
    parser.add_argument("--drive-base-dir", default="DeepseekCoder",
                      help="Base directory in Google Drive (default: DeepseekCoder)")
    parser.add_argument("--force", action="store_true",
                      help="Skip confirmation prompt")
    parser.add_argument("--skip-auth-check", action="store_true",
                      help="Skip authentication check")
    parser.add_argument("--use-preprocessed", action="store_true",
                      help="Use 'preprocessed' folder instead of 'processed_data' for compatibility with training scripts")
    args = parser.parse_args()
    
    # If --use-preprocessed is specified, override drive_folder
    drive_folder = "preprocessed" if args.use_preprocessed else args.drive_folder
    
    success_count = sync_processed_datasets(
        data_dir=args.data_dir,
        drive_folder=drive_folder,
        drive_base_dir=args.drive_base_dir,
        force=args.force,
        skip_auth_check=args.skip_auth_check
    )
    
    return 0 if success_count > 0 else 1

if __name__ == "__main__":
    sys.exit(main()) 