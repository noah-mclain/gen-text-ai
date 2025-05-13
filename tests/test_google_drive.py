#!/usr/bin/env python3
"""
Test script for Google Drive integration.
This script tests various functionalities of the Google Drive manager module.
"""

import os
import sys
import argparse
import tempfile
import logging
from pathlib import Path

# Add the project root directory to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def test_authentication(credentials_path, headless=False):
    """Test Google Drive authentication."""
    try:
        from src.utils.google_drive_manager import (
            authenticate_drive, 
            is_authenticated
        )
        
        logger.info(f"Testing authentication with credentials at: {credentials_path}")
        drive = authenticate_drive(credentials_path, headless=headless)
        
        if drive and is_authenticated(drive):
            logger.info("✅ Authentication successful!")
            return True, drive
        else:
            logger.error("❌ Authentication failed!")
            return False, None
    except Exception as e:
        logger.error(f"❌ Authentication error: {str(e)}")
        return False, None

def test_directory_setup(drive):
    """Test creating and setting up directories in Google Drive."""
    try:
        from src.utils.google_drive_manager import setup_drive_directories
        
        test_base_dir = "TestDriveIntegration"
        logger.info(f"Testing directory setup under: {test_base_dir}")
        
        directories = setup_drive_directories(drive, base_dir=test_base_dir)
        
        if directories and all(key in directories for key in ["root", "data", "models", "results"]):
            logger.info("✅ Directory setup successful!")
            logger.info(f"Created directories: {', '.join(directories.keys())}")
            return True, directories
        else:
            logger.error("❌ Directory setup failed!")
            return False, None
    except Exception as e:
        logger.error(f"❌ Directory setup error: {str(e)}")
        return False, None

def test_file_upload(drive, directory_ids):
    """Test file upload to Google Drive."""
    try:
        from src.utils.google_drive_manager import upload_file
        
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp:
            tmp_path = tmp.name
            tmp.write("This is a test file for Google Drive integration.")
        
        logger.info(f"Testing file upload with temp file: {tmp_path}")
        
        # Upload to the data directory
        file_id = upload_file(
            drive, 
            tmp_path, 
            parent_id=directory_ids["data"],
            filename="test_upload.txt"
        )
        
        if file_id:
            logger.info(f"✅ File upload successful! File ID: {file_id}")
            os.unlink(tmp_path)  # Clean up the temp file
            return True, file_id
        else:
            logger.error("❌ File upload failed!")
            os.unlink(tmp_path)  # Clean up even on failure
            return False, None
    except Exception as e:
        logger.error(f"❌ File upload error: {str(e)}")
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return False, None

def test_file_download(drive, file_id):
    """Test file download from Google Drive."""
    try:
        from src.utils.google_drive_manager import download_file
        
        download_path = tempfile.mktemp(suffix='.txt')
        logger.info(f"Testing file download to: {download_path}")
        
        success = download_file(drive, file_id, download_path)
        
        if success and os.path.exists(download_path):
            with open(download_path, 'r') as f:
                content = f.read()
            
            logger.info(f"✅ File download successful!")
            logger.info(f"Downloaded content: {content}")
            os.unlink(download_path)  # Clean up
            return True
        else:
            logger.error("❌ File download failed!")
            return False
    except Exception as e:
        logger.error(f"❌ File download error: {str(e)}")
        if 'download_path' in locals() and os.path.exists(download_path):
            os.unlink(download_path)
        return False

def test_file_delete(drive, file_id):
    """Test file deletion from Google Drive."""
    try:
        from src.utils.google_drive_manager import delete_file
        
        logger.info(f"Testing file deletion for file ID: {file_id}")
        
        success = delete_file(drive, file_id)
        
        if success:
            logger.info("✅ File deletion successful!")
            return True
        else:
            logger.error("❌ File deletion failed!")
            return False
    except Exception as e:
        logger.error(f"❌ File deletion error: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test Google Drive integration")
    parser.add_argument("--credentials", type=str, default="credentials.json",
                        help="Path to the Google credentials JSON file")
    parser.add_argument("--headless", action="store_true",
                        help="Use headless authentication mode")
    parser.add_argument("--skip-cleanup", action="store_true",
                        help="Skip cleaning up test files on Google Drive")
    args = parser.parse_args()
    
    if not os.path.exists(args.credentials):
        logger.error(f"Credentials file not found at: {args.credentials}")
        return 1
    
    logger.info("Starting Google Drive integration tests...")
    
    # Test authentication
    auth_success, drive = test_authentication(args.credentials, args.headless)
    if not auth_success:
        return 1
    
    # Test directory setup
    dir_success, directories = test_directory_setup(drive)
    if not dir_success:
        return 1
    
    # Test file upload
    upload_success, file_id = test_file_upload(drive, directories)
    if not upload_success:
        return 1
    
    # Test file download
    download_success = test_file_download(drive, file_id)
    if not download_success:
        return 1
    
    # Test file deletion (optional)
    if not args.skip_cleanup:
        delete_success = test_file_delete(drive, file_id)
        if not delete_success:
            logger.warning("Could not clean up test file. May need manual deletion.")
    
    logger.info("All Google Drive integration tests completed successfully! ✅")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 