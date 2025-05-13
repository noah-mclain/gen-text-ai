#!/usr/bin/env python3
"""
Google Drive Manager Helper

This script provides functions to test Google Drive authentication and access.
It's used by the setup_google_drive.py script.
"""

import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import from the consolidated module
try:
    from src.utils.google_drive_manager import (
        drive_manager,
        test_authentication,
        test_drive_mounting
    )
except ImportError:
    logger.error("Failed to import google_drive_manager. Make sure src/utils/google_drive_manager.py exists.")
    sys.exit(1)

def check_credentials():
    """Check if credentials file exists in expected locations."""
    from src.utils.google_drive_manager import CREDENTIALS_PATHS
    
    for path in CREDENTIALS_PATHS:
        if os.path.exists(path):
            logger.info(f"Found credentials at: {path}")
            return path
    
    logger.error("No credentials.json file found")
    logger.info("You need to download OAuth credentials from Google Cloud Console")
    return None

def list_drive_folders():
    """List folder structure in Drive."""
    if not drive_manager.authenticated:
        if not drive_manager.authenticate():
            logger.error("Authentication failed")
            return
    
    logger.info("Google Drive folder structure:")
    for key, folder_id in drive_manager.folder_ids.items():
        logger.info(f"- {key}: {folder_id}")

def main():
    # Simple command line interface
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "check-credentials":
            check_credentials()
        
        elif command == "authenticate":
            if test_authentication():
                logger.info("Authentication successful!")
            else:
                logger.error("Authentication failed")
                sys.exit(1)
        
        elif command == "test-access":
            if test_drive_mounting():
                logger.info("Drive access test successful!")
            else:
                logger.error("Drive access test failed")
                sys.exit(1)
        
        elif command == "list-folders":
            list_drive_folders()
        
        else:
            logger.error(f"Unknown command: {command}")
            logger.info("Available commands: check-credentials, authenticate, test-access, list-folders")
    else:
        logger.info("This script helps test Google Drive integration")
        logger.info("Usage: python scripts/google_drive_manager.py [command]")
        logger.info("Available commands: check-credentials, authenticate, test-access, list-folders")

if __name__ == "__main__":
    main() 