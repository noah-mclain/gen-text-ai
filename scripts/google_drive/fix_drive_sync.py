#!/usr/bin/env python3
"""
Fix Google Drive Sync Issues

This script fixes Google Drive synchronization issues by:
1. Ensuring the proper Drive Manager implementation is used
2. Fixing import paths in Python scripts
3. Directly syncing processed datasets to Google Drive

Usage:
    python scripts/google_drive/fix_drive_sync.py
"""

import os
import sys
import logging
import glob
import json
import argparse
import importlib
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(scripts_dir)
sys.path.append(str(project_root))

def fix_import_paths():
    """Fix Python import paths to ensure Google Drive Manager is properly loaded."""
    # Ensure src directory is in path
    src_path = os.path.join(project_root, 'src')
    if os.path.exists(src_path) and src_path not in sys.path:
        sys.path.insert(0, src_path)
        logger.info(f"Added src directory to path: {src_path}")
    
    # For Paperspace, also add /notebooks path
    notebooks_path = "/notebooks"
    if os.path.exists(notebooks_path) and notebooks_path not in sys.path:
        sys.path.insert(0, notebooks_path)
        logger.info(f"Added notebooks directory to path: {notebooks_path}")
    
    return True

def copy_implementation_files():
    """
    Copy Google Drive implementation files to ensure they're in all necessary locations.
    This ensures that even if import paths are wrong, the files will be found.
    """
    # Define source and target paths
    source_paths = [
        os.path.join(project_root, 'src', 'utils', 'google_drive_manager.py'),
        os.path.join(project_root, 'scripts', 'google_drive', 'google_drive_manager.py'),
        os.path.join(project_root, 'scripts', 'src', 'utils', 'google_drive_manager.py'),
        os.path.join('/notebooks', 'src', 'utils', 'google_drive_manager.py')
    ]
    
    # Find the first valid source
    valid_source = None
    for path in source_paths:
        if os.path.exists(path):
            # Check if it's a valid implementation
            try:
                with open(path, 'r') as f:
                    content = f.read()
                    if 'DriveManager' in content and 'GOOGLE_API_AVAILABLE' in content and 'minimal' not in content:
                        valid_source = path
                        logger.info(f"Found valid implementation at: {path}")
                        break
            except Exception as e:
                logger.warning(f"Error reading file {path}: {e}")
    
    if not valid_source:
        logger.error("No valid implementation found to copy")
        return False
    
    # Define target directories to copy to
    target_dirs = [
        os.path.join(project_root, 'src', 'utils'),
        os.path.join(project_root, 'scripts', 'src', 'utils'),
        os.path.join('/notebooks/src/utils') if os.path.exists('/notebooks') else None
    ]
    
    # Remove None values
    target_dirs = [d for d in target_dirs if d]
    
    # Copy to each target directory
    success = False
    for target_dir in target_dirs:
        if not os.path.exists(target_dir):
            try:
                os.makedirs(target_dir, exist_ok=True)
                logger.info(f"Created directory: {target_dir}")
            except Exception as e:
                logger.warning(f"Failed to create directory {target_dir}: {e}")
                continue
        
        target_path = os.path.join(target_dir, 'google_drive_manager.py')
        if target_path != valid_source:  # Don't copy to itself
            try:
                shutil.copy2(valid_source, target_path)
                logger.info(f"Copied implementation from {valid_source} to {target_path}")
                success = True
            except Exception as e:
                logger.warning(f"Failed to copy to {target_path}: {e}")
    
    return success

def clean_cache_files():
    """
    Clean Python cache files to ensure fresh imports.
    This prevents Python from using old cached imports.
    """
    cache_patterns = [
        os.path.join(project_root, '**', '__pycache__', '*.pyc'),
        os.path.join('/notebooks', '**', '__pycache__', '*.pyc') if os.path.exists('/notebooks') else None
    ]
    
    # Remove None values
    cache_patterns = [p for p in cache_patterns if p]
    
    # Delete cache files
    for pattern in cache_patterns:
        try:
            for cache_file in glob.glob(pattern, recursive=True):
                if 'google_drive_manager' in cache_file:
                    os.remove(cache_file)
                    logger.info(f"Removed cache file: {cache_file}")
        except Exception as e:
            logger.warning(f"Error cleaning cache files for pattern {pattern}: {e}")
    
    return True

def sync_processed_datasets(drive_base_dir="DeepseekCoder"):
    """
    Manually sync processed datasets to Google Drive.
    
    Args:
        drive_base_dir: Base directory in Google Drive
    
    Returns:
        True if successful, False otherwise
    """
    # Fix import paths first
    fix_import_paths()
    
    # Try to import the Drive Manager
    try:
        # First try from src.utils (preferred)
        from src.utils.google_drive_manager import drive_manager, sync_to_drive, configure_sync_method
        logger.info("Successfully imported Drive Manager from src.utils")
    except ImportError:
        try:
            # Next try from scripts.google_drive
            from scripts.google_drive.google_drive_manager import drive_manager, sync_to_drive, configure_sync_method
            logger.info("Successfully imported Drive Manager from scripts.google_drive")
        except ImportError:
            logger.error("Failed to import Drive Manager after fixing paths. Cannot continue.")
            return False
    
    # Configure the Drive Manager
    configure_sync_method(use_rclone=False, base_dir=drive_base_dir)
    logger.info(f"Configured Drive Manager with base directory: {drive_base_dir}")
    
    # Authenticate
    if not drive_manager.authenticate():
        logger.error("Failed to authenticate with Google Drive. Cannot continue.")
        return False
    
    logger.info("Successfully authenticated with Google Drive")
    
    # Find processed datasets
    processed_dirs = []
    data_dir_options = [
        os.path.join(project_root, 'data', 'processed'),
        os.path.join('/notebooks', 'data', 'processed') if os.path.exists('/notebooks') else None
    ]
    
    # Remove None values
    data_dir_options = [d for d in data_dir_options if d and os.path.exists(d)]
    
    if not data_dir_options:
        logger.error("No data directories found. Cannot continue.")
        return False
    
    # Find processed datasets in all data directories
    for data_dir in data_dir_options:
        logger.info(f"Searching for processed datasets in {data_dir}")
        for entry in os.listdir(data_dir):
            full_path = os.path.join(data_dir, entry)
            if os.path.isdir(full_path) and "_processed" in entry:
                processed_dirs.append(full_path)
    
    if not processed_dirs:
        logger.error("No processed datasets found.")
        return False
    
    logger.info(f"Found {len(processed_dirs)} processed datasets:")
    for dir_path in processed_dirs:
        logger.info(f"  - {os.path.basename(dir_path)}")
    
    # Sync each dataset
    success_count = 0
    for dataset_path in processed_dirs:
        dataset_name = os.path.basename(dataset_path)
        logger.info(f"Syncing {dataset_name} to Google Drive...")
        
        success = sync_to_drive(
            dataset_path,
            "processed_data",  # Drive folder key
            delete_source=False,
            update_only=False
        )
        
        if success:
            logger.info(f"Successfully synced {dataset_name} to Google Drive")
            success_count += 1
        else:
            logger.error(f"Failed to sync {dataset_name} to Google Drive")
    
    logger.info(f"Sync complete: {success_count}/{len(processed_dirs)} datasets synced successfully")
    
    return success_count > 0

def main():
    parser = argparse.ArgumentParser(description="Fix Google Drive sync issues")
    parser.add_argument("--drive-base-dir", type=str, default="DeepseekCoder",
                        help="Base directory in Google Drive")
    parser.add_argument("--skip-implementation-copy", action="store_true",
                        help="Skip copying implementation files")
    parser.add_argument("--skip-cache-clean", action="store_true",
                        help="Skip cleaning cache files")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("Google Drive Sync Fix Tool".center(80))
    print("="*80 + "\n")
    
    # 1. Fix import paths
    logger.info("Step 1: Fixing import paths...")
    fix_import_paths()
    
    # 2. Copy implementation files
    if not args.skip_implementation_copy:
        logger.info("Step 2: Copying implementation files...")
        if not copy_implementation_files():
            logger.error("Failed to copy implementation files")
            return 1
    else:
        logger.info("Skipping implementation copy as requested")
    
    # 3. Clean cache files
    if not args.skip_cache_clean:
        logger.info("Step 3: Cleaning cache files...")
        clean_cache_files()
    else:
        logger.info("Skipping cache cleaning as requested")
    
    # 4. Sync datasets
    logger.info("Step 4: Syncing processed datasets...")
    success = sync_processed_datasets(args.drive_base_dir)
    
    if success:
        print("\n" + "="*80)
        print("✅ Google Drive Sync Fix Complete".center(80))
        print("="*80 + "\n")
        print("Successfully synced datasets to Google Drive")
        return 0
    else:
        print("\n" + "="*80)
        print("❌ Google Drive Sync Fix Failed".center(80))
        print("="*80 + "\n")
        print("Failed to sync datasets to Google Drive")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 