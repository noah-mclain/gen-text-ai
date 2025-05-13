#!/usr/bin/env python3
"""
Google Drive Manager Import Script

This file ensures backward compatibility by importing and re-exporting
all functionality from the main google_drive_manager module.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    # Import everything from the main module
    from src.utils.google_drive_manager import (
        drive_manager,
        sync_to_drive,
        sync_from_drive,
        configure_sync_method,
        test_authentication,
        test_drive_mounting,
        DRIVE_FOLDERS,
        SCOPES
    )
    
    logger.debug("Successfully imported from src.utils.google_drive_manager")
    
except ImportError as e:
    logger.error(f"Failed to import from src.utils.google_drive_manager: {e}")
    
    # Define fallback functions
    def sync_to_drive(*args, **kwargs):
        logger.error("Drive sync not available. Import of google_drive_manager failed.")
        return False
        
    def sync_from_drive(*args, **kwargs):
        logger.error("Drive sync not available. Import of google_drive_manager failed.")
        return False
        
    def configure_sync_method(*args, **kwargs):
        logger.error("Drive sync not available. Import of google_drive_manager failed.")
        return None
        
    def test_authentication():
        logger.error("Drive authentication not available. Import of google_drive_manager failed.")
        return False
    
    def test_drive_mounting():
        logger.error("Drive mounting not available. Import of google_drive_manager failed.")
        return False
        
    # Define empty drive folders dict
    DRIVE_FOLDERS = {
        "data": "data",
        "models": "models",
        "logs": "logs",
        "results": "results"
    }
    
    # Define scopes
    SCOPES = ['https://www.googleapis.com/auth/drive']
    
    # Create dummy drive manager class
    class DummyDriveManager:
        def __init__(self):
            self.authenticated = False
            self.folder_ids = {}
        
        def authenticate(self):
            logger.error("Drive authentication not available. Import of google_drive_manager failed.")
            return False
    
    # Create drive manager instance
    drive_manager = DummyDriveManager()

# Export all symbols
__all__ = [
    'drive_manager',
    'sync_to_drive',
    'sync_from_drive',
    'configure_sync_method',
    'test_authentication',
    'test_drive_mounting',
    'DRIVE_FOLDERS',
    'SCOPES'
] 