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
import time
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
        setup_drive_directories,
        test_authentication
    )
    logger.info("Successfully imported Google Drive manager from src.utils")
except ImportError:
    try:
        from scripts.google_drive.google_drive_manager import (
            sync_to_drive,
            configure_sync_method,
            setup_drive_directories,
            test_authentication
        )
        logger.info("Successfully imported Google Drive manager from scripts.google_drive")
    except ImportError as e:
        logger.error(f"Failed to import Google Drive manager: {e}")
        logger.error("Make sure the correct modules are installed")
        sys.exit(1)

def find_processed_datasets(data_dir="data/processed", notebooks_dir="/notebooks/data/processed", 
                           verbose=False):
    """
    Find all processed datasets in the specified directories.
    
    Args:
        data_dir: Local data directory to check for processed datasets
        notebooks_dir: Paperspace notebooks data directory to check
        verbose: Whether to print additional debug info
        
    Returns:
        List of paths to processed datasets
    """
    processed_dirs = []
    
    # Define directories to search
    search_dirs = []
    
    # Add project data directory if it exists
    project_data_dir = os.path.join(project_root, data_dir)
    if os.path.exists(project_data_dir):
        search_dirs.append(project_data_dir)
        
    # Add absolute path if different from project path
    if os.path.isabs(data_dir) and os.path.exists(data_dir) and data_dir != project_data_dir:
        search_dirs.append(data_dir)
        
    # Add Paperspace notebooks directory if it exists
    if os.path.exists(notebooks_dir):
        search_dirs.append(notebooks_dir)
    
    # Search each directory for processed datasets
    for search_dir in search_dirs:
        logger.info(f"Checking for datasets in: {search_dir}")
        
        try:
            for entry in os.listdir(search_dir):
                full_path = os.path.join(search_dir, entry)
                
                # Skip non-directories
                if not os.path.isdir(full_path):
                    continue
                
                # Match various dataset naming patterns
                if (entry.endswith("_processed") or 
                    "_processed_" in entry or
                    "processed_interim" in entry or
                    entry.startswith("codesearchnet_") or
                    entry.startswith("code_alpaca_") or
                    entry.startswith("instruct_code_") or
                    entry.startswith("mbpp_") or
                    entry.startswith("humaneval_")):
                    
                    # Make sure we don't add duplicates (same folder name from different locations)
                    if entry not in [os.path.basename(p) for p in processed_dirs]:
                        processed_dirs.append(full_path)
                        if verbose:
                            logger.debug(f"Found dataset: {entry} at {full_path}")
        except Exception as e:
            logger.error(f"Error reading directory {search_dir}: {e}")
    
    # If no specific datasets found, include all directories as fallback
    if not processed_dirs:
        logger.warning("No datasets matched specific patterns. Checking for any directories...")
        
        for search_dir in search_dirs:
            try:
                for entry in os.listdir(search_dir):
                    full_path = os.path.join(search_dir, entry)
                    if os.path.isdir(full_path):
                        # Make sure we don't add duplicates
                        if entry not in [os.path.basename(p) for p in processed_dirs]:
                            processed_dirs.append(full_path)
                            logger.info(f"Found directory: {entry}")
            except Exception as e:
                logger.error(f"Error reading directory {search_dir}: {e}")
    
    # Sort for consistent output
    processed_dirs = sorted(processed_dirs)
    return processed_dirs

def sync_processed_datasets(data_dir="data/processed", notebooks_dir="/notebooks/data/processed",
                         drive_folder="data/processed", drive_base_dir="DeepseekCoder",
                         force=False, skip_auth_check=False, verbose=False):
    """
    Find all processed datasets and sync them to Google Drive.
    
    Args:
        data_dir: Local data directory to check for processed datasets
        notebooks_dir: Paperspace notebooks data directory to check
        drive_folder: Google Drive folder to sync to (relative to drive_base_dir)
        drive_base_dir: Base directory in Google Drive
        force: Skip confirmation prompt if True
        skip_auth_check: Skip authentication check if True
        verbose: Enable verbose logging
        
    Returns:
        Success count (number of datasets successfully synced)
    """
    # Check authentication first
    if not skip_auth_check:
        logger.info("Testing Google Drive authentication...")
        if not test_authentication():
            logger.error("Google Drive authentication failed. Please run setup_google_drive.py first.")
            return 0
    
    # Set up drive directories to ensure they exist
    logger.info(f"Setting up Google Drive with base directory: {drive_base_dir}")
    drive_folders = setup_drive_directories(base_dir=drive_base_dir)
    if not drive_folders:
        logger.error("Failed to set up Google Drive directories")
        return 0
    logger.info(f"Successfully set up Google Drive directories: {list(drive_folders.keys())}")
    
    # Configure Drive sync with the specified base directory
    configure_sync_method(use_rclone=False, base_dir=drive_base_dir)
    logger.info(f"Configured Drive sync with base directory: {drive_base_dir}")
    
    # Find all processed datasets
    processed_dirs = find_processed_datasets(data_dir, notebooks_dir, verbose)
    
    # Print what we found
    if processed_dirs:
        logger.info(f"Found {len(processed_dirs)} datasets to sync:")
        for i, path in enumerate(processed_dirs, 1):
            logger.info(f"{i}. {os.path.basename(path)} ({path})")
    else:
        logger.error("No datasets found to sync. Check that your datasets exist in the specified directories.")
        return 0
    
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
        logger.info(f"Syncing {dataset_name} to Google Drive folder '{drive_folder}'...")
        
        try:
            # Add a small delay to avoid rate limiting
            time.sleep(1)
            
            # Record start time for performance tracking
            start_time = time.time()
            
            success = sync_to_drive(
                dataset_path,
                drive_folder,
                delete_source=False,
                update_only=False
            )
            
            # Calculate sync time
            sync_time = time.time() - start_time
            
            if success:
                logger.info(f"✅ Successfully synced {dataset_name} to Google Drive in {sync_time:.2f} seconds")
                success_count += 1
            else:
                logger.error(f"❌ Failed to sync {dataset_name} to Google Drive")
        except Exception as e:
            logger.error(f"Error syncing {dataset_name}: {e}")
    
    # Final summary
    if success_count > 0:
        logger.info(f"Sync complete! Successfully synced {success_count}/{len(processed_dirs)} datasets to Google Drive.")
    else:
        logger.error(f"Sync failed. No datasets were successfully synced to Google Drive.")
    
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
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose logging")
    parser.add_argument("--list-only", action="store_true",
                      help="Only list datasets, don't sync")
    parser.add_argument("--specific-dataset", 
                      help="Sync only a specific dataset (by name)")
    args = parser.parse_args()
    
    # If --use-preprocessed is specified, override drive_folder
    drive_folder = "preprocessed" if args.use_preprocessed else args.drive_folder
    
    # Find processed datasets
    processed_dirs = find_processed_datasets(
        data_dir=args.data_dir,
        notebooks_dir="/notebooks/data/processed",
        verbose=args.verbose
    )
    
    # If no datasets found, exit
    if not processed_dirs:
        logger.error("No datasets found. Make sure the data directory exists and contains processed datasets.")
        return 1
    
    # Filter specific dataset if requested
    if args.specific_dataset:
        specific_name = args.specific_dataset
        filtered_dirs = []
        for dataset in processed_dirs:
            if os.path.basename(dataset) == specific_name or os.path.basename(dataset).startswith(specific_name):
                filtered_dirs.append(dataset)
        
        if not filtered_dirs:
            logger.error(f"No datasets found matching '{specific_name}'")
            # List available datasets
            logger.info("Available datasets:")
            for dataset in processed_dirs:
                logger.info(f"  - {os.path.basename(dataset)}")
            return 1
            
        processed_dirs = filtered_dirs
        logger.info(f"Filtered to {len(processed_dirs)} datasets matching '{specific_name}'")
    
    # List only mode
    if args.list_only:
        logger.info("Dataset listing complete. Use --verbose for more details.")
        return 0
    
    # If we're continuing, sync the datasets
    success_count = sync_processed_datasets(
        data_dir=args.data_dir,
        drive_folder=drive_folder,
        drive_base_dir=args.drive_base_dir,
        force=args.force,
        skip_auth_check=args.skip_auth_check,
        verbose=args.verbose
    )
    
    return 0 if success_count > 0 else 1

if __name__ == "__main__":
    sys.exit(main()) 