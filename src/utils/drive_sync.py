"""
Google Drive Sync (Compatibility Module)

This is a bridge module that provides backward compatibility for code 
that still uses the old drive_sync and drive_api_utils imports. This
redirects to the consolidated google_drive_manager module.
"""

import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path for absolute imports
module_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if module_path not in sys.path:
    sys.path.append(module_path)

# Import from the consolidated module
try:
    from src.utils.google_drive_manager import (
        # Main functions and classes
        DriveManager,
        drive_manager,
        sync_to_drive,
        sync_from_drive,
        configure_sync_method,
        test_authentication,
        test_drive_mounting,
        
        # Constants and globals
        SCOPES,
        DRIVE_FOLDERS,
        CREDENTIALS_PATHS,
        GOOGLE_API_AVAILABLE
    )
    
    logger.info("Successfully imported from google_drive_manager")
    DRIVE_SYNC_AVAILABLE = True
    
except ImportError as e:
    logger.error(f"Error importing from google_drive_manager: {e}")
    logger.warning("Drive sync functionality will be limited")
    DRIVE_SYNC_AVAILABLE = False
    
    # Define fallback functions
    def sync_to_drive(*args, **kwargs):
        logger.error("Drive sync not available - google_drive_manager could not be imported")
        return False
        
    def sync_from_drive(*args, **kwargs):
        logger.error("Drive sync not available - google_drive_manager could not be imported")
        return False
        
    def configure_sync_method(*args, **kwargs):
        logger.error("Drive sync not available - google_drive_manager could not be imported")
        return None
        
    def test_authentication():
        logger.error("Drive authentication not available - google_drive_manager could not be imported")
        return False
        
    def test_drive_mounting():
        logger.error("Drive mounting test not available - google_drive_manager could not be imported")
        return False

    # Define empty class for fallback
    class DriveManager:
        def __init__(self, *args, **kwargs):
            logger.error("DriveManager not available - google_drive_manager could not be imported")
        
        def authenticate(self):
            return False
            
        def upload_file(self, *args, **kwargs):
            return False
            
        def download_file(self, *args, **kwargs):
            return False


# Aliases and compatibility functions for the old drive_api_utils module
def get_authenticated_service(*args, **kwargs):
    """
    Compatibility function for old get_authenticated_service.
    Now returns None or a Drive service from drive_manager.
    """
    if not DRIVE_SYNC_AVAILABLE:
        return None
        
    # Try to authenticate
    if drive_manager.authenticate():
        return drive_manager.service
    return None
    
def create_or_get_folder(name, parent_id=None):
    """
    Compatibility function for old create_or_get_folder.
    
    Args:
        name: Folder name
        parent_id: Parent folder ID
        
    Returns:
        Folder ID or None
    """
    if not DRIVE_SYNC_AVAILABLE:
        return None
        
    # Find or create the folder
    folder_id = drive_manager.find_file_id(name, parent_id)
    if not folder_id:
        folder_id = drive_manager.create_folder(name, parent_id)
    return folder_id
    
def upload_file(service, local_path, folder_id, filename=None):
    """
    Compatibility function for old upload_file.
    
    Args:
        service: Drive service (ignored)
        local_path: Path to local file
        folder_id: ID of destination folder
        filename: Filename to use (optional)
        
    Returns:
        File ID or None
    """
    if not DRIVE_SYNC_AVAILABLE:
        return None
        
    # Determine filename
    if filename is None:
        filename = os.path.basename(local_path)
    
    # Create a temporary file with the desired name if needed
    temp_file = None
    if filename != os.path.basename(local_path):
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)
        shutil.copy2(local_path, temp_path)
        local_path = temp_path
        temp_file = temp_path
    
    try:
        # Upload directly using folder ID
        success = sync_to_drive(local_path, folder_id)
        
        # Find the uploaded file ID
        if success:
            file_id = drive_manager.find_file_id(filename, folder_id)
            return file_id
        return None
    finally:
        # Clean up temp file if created
        if temp_file:
            try:
                os.remove(temp_file)
                os.rmdir(os.path.dirname(temp_file))
            except:
                pass
                
def download_file(service, file_id, local_path):
    """
    Compatibility function for old download_file.
    
    Args:
        service: Drive service (ignored)
        file_id: ID of file to download
        local_path: Path to save file
        
    Returns:
        True if successful, False otherwise
    """
    if not DRIVE_SYNC_AVAILABLE:
        return False
        
    return drive_manager.download_file(file_id, local_path)
    
def setup_drive_structure(service, base_folder_name="DeepseekCoder"):
    """
    Compatibility function for old setup_drive_structure.
    
    Args:
        service: Drive service (ignored)
        base_folder_name: Name of base folder
        
    Returns:
        Dict of folder IDs or empty dict
    """
    if not DRIVE_SYNC_AVAILABLE:
        return {}
        
    # Make sure we're using the right base directory
    global drive_manager
    if drive_manager.base_dir != base_folder_name:
        drive_manager = DriveManager(base_dir=base_folder_name)
        if not drive_manager.authenticate():
            return {}
    
    # Set up drive structure and return folder IDs
    drive_manager._setup_drive_structure()
    return drive_manager.folder_ids 