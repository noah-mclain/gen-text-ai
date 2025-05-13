#!/usr/bin/env python3
"""
Cleanup Additional Drive Scripts

This script cleans up redundant drive-related scripts after the new
Google Drive manager integration.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Files to be removed
REDUNDANT_FILES = [
    "scripts/preprocess_with_drive.sh",
    "scripts/fix_drive_auth.py",
    "scripts/archived/setup_drive.py"
]

def cleanup_files(dry_run: bool = False):
    """
    Remove redundant drive-related scripts.
    
    Args:
        dry_run: If True, only print actions without deleting files
    """
    for file_path in REDUNDANT_FILES:
        if os.path.exists(file_path):
            if dry_run:
                logger.info(f"Would remove: {file_path}")
            else:
                try:
                    os.remove(file_path)
                    logger.info(f"Removed: {file_path}")
                except Exception as e:
                    logger.error(f"Error removing {file_path}: {e}")
        else:
            logger.info(f"File not found: {file_path}")

def main():
    parser = argparse.ArgumentParser(description="Clean up redundant drive-related scripts")
    
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be removed without actually deleting files")
    
    args = parser.parse_args()
    
    print("\nCleaning up redundant drive-related scripts...")
    
    if args.dry_run:
        print("Dry run mode - no files will be removed\n")
    
    cleanup_files(args.dry_run)
    
    if not args.dry_run:
        print("\nCleanup completed. The following files were made redundant:")
        for file in REDUNDANT_FILES:
            print(f"- {file}")
        print("\nYou should now use the following scripts for Google Drive integration:")
        print("- scripts/setup_google_drive.py - For authenticating with Google Drive")
        print("- scripts/sync_to_drive.py - For syncing files to/from Google Drive")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 