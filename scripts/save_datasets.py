#!/usr/bin/env python3
"""
Dataset Sync and Save Script

This script ensures that datasets are properly saved locally and synced with Google Drive.
It checks for existing datasets on Drive, downloads them if needed, and ensures all
processed datasets are available for training.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from datasets import load_from_disk, Dataset
from src.utils.google_drive_manager import sync_to_drive, sync_from_drive, configure_sync_method
from src.utils.drive_dataset_checker import prepare_datasets

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dataset_exists(dataset_path):
    """Check if a dataset exists locally."""
    try:
        if os.path.exists(dataset_path):
            # Try to load it to make sure it's valid
            dataset = load_from_disk(dataset_path)
            return True
        return False
    except Exception as e:
        logger.error(f"Error checking dataset at {dataset_path}: {e}")
        return False

def sync_datasets_to_drive(datasets_dir, drive_folder="preprocessed"):
    """Sync all locally processed datasets to Google Drive."""
    logger.info(f"Syncing datasets from {datasets_dir} to Drive folder '{drive_folder}'")
    
    success_count = 0
    failure_count = 0
    
    # Find all processed dataset directories
    for entry in os.listdir(datasets_dir):
        path = os.path.join(datasets_dir, entry)
        if os.path.isdir(path) and entry.endswith("_processed"):
            dataset_name = entry
            logger.info(f"Syncing dataset {dataset_name} to Drive")
            
            try:
                # Sync to Drive
                success = sync_to_drive(
                    path,
                    os.path.join(drive_folder, dataset_name) if drive_folder else dataset_name,
                    delete_source=False,
                    update_only=False
                )
                
                if success:
                    logger.info(f"Successfully synced {dataset_name} to Drive")
                    success_count += 1
                else:
                    logger.error(f"Failed to sync {dataset_name} to Drive")
                    failure_count += 1
            except Exception as e:
                logger.error(f"Error syncing {dataset_name} to Drive: {e}")
                failure_count += 1
    
    logger.info(f"Sync complete. Success: {success_count}, Failures: {failure_count}")
    return success_count, failure_count

def sync_datasets_from_drive(datasets_dir, drive_folder="preprocessed"):
    """Download all datasets from Google Drive to local storage."""
    logger.info(f"Syncing datasets from Drive folder '{drive_folder}' to {datasets_dir}")
    
    # Ensure local directory exists
    os.makedirs(datasets_dir, exist_ok=True)
    
    # Get dataset config
    config_path = os.path.join(project_root, "config", "dataset_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get list of datasets from config
    dataset_names = list(config.keys())
    
    # Prepare datasets using the drive_dataset_checker
    available, needed, download_time = prepare_datasets(
        config_path=config_path,
        output_dir=datasets_dir,
        drive_folder=drive_folder
    )
    
    logger.info(f"Downloaded datasets: {', '.join(available)}")
    logger.info(f"Missing datasets that need processing: {', '.join(needed)}")
    logger.info(f"Download time: {download_time:.2f} seconds")
    
    return available, needed

def process_missing_datasets(needed_datasets, config_path, output_dir):
    """Process any missing datasets that couldn't be downloaded from Drive."""
    if not needed_datasets:
        logger.info("No missing datasets to process.")
        return True
    
    logger.info(f"Processing missing datasets: {', '.join(needed_datasets)}")
    
    # Import the processing module
    from src.data.process_datasets import process_datasets
    
    try:
        # Process only the needed datasets
        process_datasets(
            config_path=config_path,
            datasets=needed_datasets,
            streaming=True,
            no_cache=True
        )
        logger.info(f"Successfully processed missing datasets: {', '.join(needed_datasets)}")
        return True
    except Exception as e:
        logger.error(f"Error processing missing datasets: {e}")
        return False

def verify_datasets(datasets_dir, config_path):
    """Verify all required datasets exist and are valid."""
    # Load dataset config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    dataset_names = list(config.keys())
    missing_datasets = []
    
    for dataset_name in dataset_names:
        dataset_path = os.path.join(datasets_dir, f"{dataset_name}_processed")
        if not check_dataset_exists(dataset_path):
            missing_datasets.append(dataset_name)
    
    logger.info(f"Found {len(dataset_names) - len(missing_datasets)} valid datasets")
    if missing_datasets:
        logger.warning(f"Missing datasets: {', '.join(missing_datasets)}")
    
    return missing_datasets

def main():
    parser = argparse.ArgumentParser(description="Ensure datasets are properly saved and synced")
    parser.add_argument("--data_dir", type=str, default="data/processed", 
                       help="Directory containing processed datasets")
    parser.add_argument("--drive_folder", type=str, default="preprocessed",
                       help="Folder name on Google Drive")
    parser.add_argument("--download", action="store_true",
                       help="Download datasets from Drive to local storage")
    parser.add_argument("--upload", action="store_true",
                       help="Upload local datasets to Drive")
    parser.add_argument("--verify", action="store_true",
                       help="Verify all required datasets exist locally")
    parser.add_argument("--process_missing", action="store_true",
                       help="Process any missing datasets")
    parser.add_argument("--credentials_path", type=str, default="credentials.json",
                       help="Path to Google Drive API credentials JSON file")
    parser.add_argument("--drive_base_dir", type=str, default="DeepseekCoder",
                       help="Base directory on Google Drive")
    
    args = parser.parse_args()
    
    # Ensure paths are absolute
    data_dir = os.path.join(project_root, args.data_dir) if not os.path.isabs(args.data_dir) else args.data_dir
    config_path = os.path.join(project_root, "config", "dataset_config.json")
    
    # Configure Drive sync
    configure_sync_method(base_dir=args.drive_base_dir)
    
    if args.download:
        # Download datasets from Drive
        available, needed = sync_datasets_from_drive(data_dir, args.drive_folder)
        
        # Process missing datasets if requested
        if args.process_missing and needed:
            process_missing_datasets(needed, config_path, data_dir)
    
    if args.upload:
        # Upload datasets to Drive
        sync_datasets_to_drive(data_dir, args.drive_folder)
    
    if args.verify:
        # Verify datasets exist
        missing = verify_datasets(data_dir, config_path)
        
        # Process missing datasets if requested
        if args.process_missing and missing:
            process_missing_datasets(missing, config_path, data_dir)
    
    if not (args.download or args.upload or args.verify):
        # Default behavior: download, verify, and process missing
        logger.info("No specific action requested. Performing full sync...")
        available, needed = sync_datasets_from_drive(data_dir, args.drive_folder)
        
        if args.process_missing and needed:
            process_missing_datasets(needed, config_path, data_dir)
            
        # Verify again after processing
        missing = verify_datasets(data_dir, config_path)
        
        if not missing:
            logger.info("All datasets are now available!")
        else:
            logger.warning(f"Still missing datasets: {', '.join(missing)}")

if __name__ == "__main__":
    main() 