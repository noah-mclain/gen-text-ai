#!/usr/bin/env python3
"""
Google Drive Manager Redirect

This file ensures that all imports of google_drive_manager from the scripts directory
are directed to the main implementation in src/utils/google_drive_manager.py.

This avoids duplication of code and ensures consistency across the project.
"""

import os
import sys
import logging
from pathlib import Path
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add various possible paths to sys.path to handle different environments
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(scripts_dir)
sys.path.append(str(project_root))

# Add the src directory to sys.path
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
    logger.info(f"Added src directory to Python path: {src_path}")

# Import directly from src.utils.google_drive_manager
try:
    from src.utils.google_drive_manager import (
        DriveManager, test_authentication, sync_to_drive, sync_from_drive,
        configure_sync_method, test_drive_mounting, _drive_manager,
        DRIVE_FOLDERS, SCOPES, GOOGLE_API_AVAILABLE, setup_drive_directories
    )
    
    logger.info("Successfully imported from src.utils.google_drive_manager")
except ImportError as e:
    logger.error(f"Failed to import from src.utils.google_drive_manager: {e}")
    sys.exit(1)

# Make all imported components available at the module level
__all__ = [
    'DriveManager',
    'test_authentication',
    'sync_to_drive',
    'sync_from_drive',
    'configure_sync_method', 
    'test_drive_mounting',
    '_drive_manager',
    'DRIVE_FOLDERS',
    'SCOPES',
    'GOOGLE_API_AVAILABLE',
    'setup_drive_directories'
]

# Define functions to ensure they're available directly from this module
def test_authentication():
    """Test authentication with Google Drive."""
    try:
        return module.test_authentication()
    except Exception as e:
        logger.error(f"Error in test_authentication: {e}")
        return False

def sync_to_drive(local_path, drive_folder_key, delete_source=False, update_only=False):
    """Sync files to Google Drive."""
    try:
        return module.sync_to_drive(local_path, drive_folder_key, delete_source, update_only)
    except Exception as e:
        logger.error(f"Error in sync_to_drive: {e}")
        return False

def sync_from_drive(drive_folder_key, local_path):
    """Sync files from Google Drive."""
    try:
        return module.sync_from_drive(drive_folder_key, local_path)
    except Exception as e:
        logger.error(f"Error in sync_from_drive: {e}")
        return False

def configure_sync_method(use_rclone=False, base_dir="DeepseekCoder"):
    """Configure the sync method."""
    try:
        return module.configure_sync_method(use_rclone, base_dir)
    except Exception as e:
        logger.error(f"Error in configure_sync_method: {e}")
        return None 