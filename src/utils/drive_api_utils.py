"""
Google Drive API Utils (Compatibility Module)

This is a bridge module that provides backward compatibility for code 
that still uses the old drive_api_utils imports. This redirects to the 
consolidated google_drive_manager module.
"""

import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from the consolidated module
try:
    from src.utils.google_drive_manager import (
        # Classes and main functionality
        DriveManager,
        drive_manager,
        sync_to_drive,
        sync_from_drive,
        configure_sync_method,
        test_authentication,
        test_drive_mounting,
        
        # Constants
        DRIVE_FOLDERS,
        SCOPES,
        GOOGLE_API_AVAILABLE
    )
    
    logger.debug("Successfully redirected from drive_api_utils to google_drive_manager")
    DRIVE_API_AVAILABLE = GOOGLE_API_AVAILABLE
    
except ImportError as e:
    logger.error(f"Error importing from google_drive_manager: {e}")
    logger.warning("Drive API functionality will not be available")
    DRIVE_API_AVAILABLE = False
    
    # Define empty constants
    DRIVE_FOLDERS = {}
    SCOPES = []
    
    # Define dummy class
    class DriveManager:
        def __init__(self, *args, **kwargs):
            logger.error("DriveManager not available - google_drive_manager could not be imported")
        
        def authenticate(self):
            return False

# Re-export compatibility functions that maintain the old API
def initialize_drive_api(credentials_path=None, headless=False, token_path=None):
    """
    Initialize the Google Drive API.
    
    Args:
        credentials_path: Path to the credentials.json file
        headless: Whether to run in headless mode
        token_path: Path to save/load the token file
        
    Returns:
        DriveManager instance if successful, None otherwise
    """
    if not DRIVE_API_AVAILABLE:
        logger.error("Drive API not available - google_drive_manager could not be imported")
        return None
    
    try:
        # Create a new DriveManager instance with the provided credentials
        manager = DriveManager(token_path=token_path)
        
        # Set credentials path if provided
        if credentials_path:
            manager.credentials_path = credentials_path
        
        # Authenticate
        if manager.authenticate():
            logger.info("Successfully initialized Google Drive API")
            return manager
        else:
            logger.error("Failed to authenticate with Google Drive API")
            return None
    except Exception as e:
        logger.error(f"Error initializing Drive API: {e}")
        return None

def setup_drive_directories(service_or_manager, base_dir=None):
    """
    Set up directory structure on Google Drive.
    
    Args:
        service_or_manager: DriveManager instance or Drive service
        base_dir: Base directory name
        
    Returns:
        Dict of folder IDs
    """
    if not DRIVE_API_AVAILABLE:
        logger.error("Drive API not available - google_drive_manager could not be imported")
        return {}
    
    try:
        if isinstance(service_or_manager, DriveManager):
            manager = service_or_manager
        else:
            # If passed a service, use the global manager
            manager = drive_manager
            
        # Set base directory if provided
        if base_dir and manager.base_dir != base_dir:
            manager = DriveManager(base_dir=base_dir)
            if not manager.authenticate():
                return {}
        
        # Set up drive structure
        manager._setup_drive_structure()
        return manager.folder_ids
    except Exception as e:
        logger.error(f"Error setting up Drive directories: {e}")
        return {}

def save_to_drive(local_path, drive_folder_key, manager=None, delete_source=False):
    """
    Save file or folder to Google Drive.
    
    Args:
        local_path: Path to local file or folder
        drive_folder_key: Key for destination folder
        manager: DriveManager instance (optional)
        delete_source: Whether to delete the source after upload
        
    Returns:
        True if successful, False otherwise
    """
    if not DRIVE_API_AVAILABLE:
        logger.error("Drive API not available - google_drive_manager could not be imported")
        return False
    
    try:
        # Use provided manager or global instance
        mgr = manager or drive_manager
        
        if os.path.isdir(local_path):
            return mgr.upload_folder(local_path, drive_folder_key, delete_source)
        else:
            return mgr.upload_file(local_path, drive_folder_key, delete_source)
    except Exception as e:
        logger.error(f"Error saving to Drive: {e}")
        return False

def load_from_drive(drive_folder_key, local_path, manager=None):
    """
    Load file or folder from Google Drive.
    
    Args:
        drive_folder_key: Key for source folder
        local_path: Path to save locally
        manager: DriveManager instance (optional)
        
    Returns:
        True if successful, False otherwise
    """
    if not DRIVE_API_AVAILABLE:
        logger.error("Drive API not available - google_drive_manager could not be imported")
        return False
    
    try:
        # Use provided manager or global instance
        mgr = manager or drive_manager
        
        # Create parent directory if needed
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Find folder ID for the drive folder key
        folder_id = None
        
        # Check if it's a direct key in DRIVE_FOLDERS
        if drive_folder_key in mgr.folder_ids:
            folder_id = mgr.folder_ids[drive_folder_key]
        elif drive_folder_key in DRIVE_FOLDERS:
            # Get the path and find the corresponding ID
            drive_path = DRIVE_FOLDERS[drive_folder_key]
            
            if drive_path in mgr.folder_ids:
                folder_id = mgr.folder_ids[drive_path]
            else:
                # Navigate the path to find the ID
                parts = drive_path.split('/')
                current_id = mgr.folder_ids.get('base')
                
                for part in parts:
                    folder_id = mgr.find_file_id(part, current_id)
                    if not folder_id:
                        logger.error(f"Folder not found: {part} in {drive_path}")
                        return False
                    current_id = folder_id
                
                folder_id = current_id
        else:
            # Treat as direct path
            parts = drive_folder_key.split('/')
            current_id = mgr.folder_ids.get('base')
            
            for part in parts:
                folder_id = mgr.find_file_id(part, current_id)
                if not folder_id:
                    logger.error(f"Folder not found: {part} in {drive_folder_key}")
                    return False
                current_id = folder_id
            
            folder_id = current_id
        
        if not folder_id:
            logger.error(f"Could not find folder ID for: {drive_folder_key}")
            return False
        
        return mgr.download_folder(folder_id, local_path)
    except Exception as e:
        logger.error(f"Error loading from Drive: {e}")
        return False

# Define the exported symbols
__all__ = [
    'DriveManager',
    'drive_manager',
    'initialize_drive_api',
    'DRIVE_API_AVAILABLE',
    'DRIVE_FOLDERS',
    'SCOPES'
] 