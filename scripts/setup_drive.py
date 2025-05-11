#!/usr/bin/env python3
"""
Helper script to set up Google Drive API integration on Paperspace.
Run this script to authenticate and initialize Google Drive access.
"""

import os
import sys
import argparse
import logging

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from src.utils.drive_utils import (
        mount_google_drive, 
        setup_drive_directories,
        is_drive_mounted,
        find_files_by_name,
        save_model_to_drive,
        download_file_from_drive
    )
except ImportError:
    logger.error("Failed to import drive utilities. Make sure src/utils/drive_utils.py exists.")
    sys.exit(1)

def install_dependencies():
    """Install required dependencies for Google Drive API"""
    try:
        import pip
        logger.info("Installing Google Drive API dependencies...")
        
        # Install required packages
        pip.main(['install', 'google-api-python-client', 'google-auth-httplib2', 'google-auth-oauthlib'])
        
        logger.info("Dependencies installed successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def setup_drive(base_dir, install_deps=False):
    """Set up Google Drive API integration"""
    if install_deps:
        if not install_dependencies():
            return False
    
    # Authenticate and set up Google Drive
    if not mount_google_drive():
        logger.error("Failed to set up Google Drive API.")
        return False
    
    # Create directory structure
    dir_ids = setup_drive_directories(base_dir)
    if not dir_ids:
        logger.error("Failed to create directory structure.")
        return False
    
    logger.info(f"Successfully set up Google Drive with base directory: {base_dir}")
    logger.info(f"Created directories: {', '.join(dir_ids.keys())}")
    
    return True

def list_files(query, folder=None):
    """List files matching query in Google Drive"""
    if not is_drive_mounted():
        logger.error("Google Drive API not authenticated. Run with --setup first.")
        return False
    
    files = find_files_by_name(query, folder)
    if not files:
        logger.info(f"No files found matching '{query}'")
        return True
    
    logger.info(f"Found {len(files)} files matching '{query}':")
    for file in files:
        modified = file.get('modifiedTime', 'Unknown date')
        size = file.get('size', 'Unknown size')
        logger.info(f"- {file['name']} (ID: {file['id']}, Modified: {modified}, Size: {size})")
    
    return True

def upload_file(local_path, drive_folder, name=None):
    """Upload a file or directory to Google Drive"""
    if not is_drive_mounted():
        logger.error("Google Drive API not authenticated. Run with --setup first.")
        return False
    
    if not os.path.exists(local_path):
        logger.error(f"Local path does not exist: {local_path}")
        return False
    
    success = save_model_to_drive(local_path, drive_folder, name)
    if success:
        logger.info(f"Successfully uploaded {local_path} to Drive folder {drive_folder}")
    else:
        logger.error(f"Failed to upload {local_path}")
    
    return success

def download_file(file_id, local_path):
    """Download a file from Google Drive"""
    if not is_drive_mounted():
        logger.error("Google Drive API not authenticated. Run with --setup first.")
        return False
    
    success = download_file_from_drive(file_id, local_path)
    if success:
        logger.info(f"Successfully downloaded file to {local_path}")
    else:
        logger.error(f"Failed to download file {file_id}")
    
    return success

def main():
    parser = argparse.ArgumentParser(description="Google Drive API setup and utility tool")
    
    # Setup commands
    parser.add_argument("--setup", action="store_true", help="Set up Google Drive API integration")
    parser.add_argument("--install-deps", action="store_true", help="Install required dependencies")
    parser.add_argument("--base-dir", type=str, default="DeepseekCoder", 
                       help="Base directory name on Google Drive")
    
    # Operation commands
    parser.add_argument("--list", type=str, help="List files matching a query")
    parser.add_argument("--folder", type=str, help="Folder ID to search in (for --list)")
    parser.add_argument("--upload", type=str, help="Upload a file or directory to Drive")
    parser.add_argument("--to-folder", type=str, help="Target folder ID or path (for --upload)")
    parser.add_argument("--download", type=str, help="Download a file from Drive (specify file ID)")
    parser.add_argument("--to-path", type=str, help="Local path to save downloaded file")
    
    args = parser.parse_args()
    
    # Handle setup command
    if args.setup:
        if setup_drive(args.base_dir, args.install_deps):
            logger.info("Google Drive API integration successfully set up")
            return 0
        else:
            logger.error("Failed to set up Google Drive API integration")
            return 1
    
    # Handle list command
    if args.list:
        if list_files(args.list, args.folder):
            return 0
        return 1
    
    # Handle upload command
    if args.upload:
        if not args.to_folder:
            logger.error("Missing --to-folder argument for upload operation")
            return 1
        
        if upload_file(args.upload, args.to_folder):
            return 0
        return 1
    
    # Handle download command
    if args.download:
        if not args.to_path:
            logger.error("Missing --to-path argument for download operation")
            return 1
        
        if download_file(args.download, args.to_path):
            return 0
        return 1
    
    # If no command specified, show help
    if not any([args.setup, args.list, args.upload, args.download]):
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 