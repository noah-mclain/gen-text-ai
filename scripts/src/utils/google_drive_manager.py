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

# Define local credential paths to avoid dependency on the main implementation
CREDENTIALS_PATHS = [
    "credentials.json",  # Current directory
    os.path.join(os.path.expanduser("~"), "credentials.json"),  # User's home directory
    "/notebooks/credentials.json",  # Paperspace path
    os.path.join(os.getcwd(), "credentials.json"),  # Current working directory
    os.path.join(project_root, "credentials.json")  # Project root
]

# Create a minimal DriveManager class for fallback
class MinimalDriveManager:
    def __init__(self):
        self.authenticated = False
        self.service = None
        self.base_dir = "DeepseekCoder"
        self.folder_ids = {}
        
    def authenticate(self):
        logger.warning("Using minimal drive manager, actual functionality not available")
        return False

# Create default instances
drive_manager = MinimalDriveManager()
_drive_manager = drive_manager

# Try to find the implementation
main_module = None

# Define possible paths to the actual implementation
possible_paths = [
    # Standard path in regular project structure
    os.path.join(project_root, 'src', 'utils', 'google_drive_manager.py'),
    
    # Paperspace might have a different structure
    os.path.join('/notebooks', 'src', 'utils', 'google_drive_manager.py'),
    
    # Handle case where scripts is the root on Paperspace
    os.path.join(scripts_dir, 'src', 'utils', 'google_drive_manager.py'),
    
    # Try relative path for nested imports
    os.path.join(os.path.dirname(current_dir), '..', 'src', 'utils', 'google_drive_manager.py'),
]

# Try to find an existing implementation
implementation_path = None
for path in possible_paths:
    if os.path.exists(path):
        implementation_path = path
        logger.info(f"Found implementation at {implementation_path}")
        break

# If no implementation is found, try to copy it
if not implementation_path:
    # Check if we have the implementation in this directory
    local_copy = os.path.join(current_dir, 'google_drive_manager_impl.py')
    if os.path.exists(local_copy):
        # Create the target directory if needed
        target_dir = os.path.join(project_root, 'src', 'utils')
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy the implementation to the target location
        target_path = os.path.join(target_dir, 'google_drive_manager.py')
        shutil.copy(local_copy, target_path)
        logger.info(f"Copied implementation from {local_copy} to {target_path}")
        implementation_path = target_path
    else:
        logger.error("No implementation file found and no local copy available")
        logger.error("Please ensure src/utils/google_drive_manager.py exists in your project")

# Add the directory of the implementation to sys.path if found
if implementation_path:
    impl_dir = os.path.dirname(os.path.dirname(implementation_path))
    if impl_dir not in sys.path:
        sys.path.append(impl_dir)

# Try to import from the implementation
try:
    # Import relative to root
    import_path = "src.utils.google_drive_manager"
    try:
        # Try importing using absolute import
        main_module = __import__(import_path, fromlist=['*'])
        
        # Update our instances with the real ones if available
        if hasattr(main_module, 'drive_manager'):
            logger.debug("Using drive_manager from the module")
            drive_manager = main_module.drive_manager
            _drive_manager = drive_manager
        elif hasattr(main_module, '_drive_manager'):
            logger.debug("Using _drive_manager as drive_manager")
            drive_manager = main_module._drive_manager
            _drive_manager = drive_manager
        else:
            # Keep using our minimal implementation
            logger.warning("Could not find drive_manager or _drive_manager in the module")
            # But try to create a real one if DriveManager class is available
            if hasattr(main_module, 'DriveManager'):
                try:
                    drive_manager = main_module.DriveManager()
                    _drive_manager = drive_manager
                    logger.debug("Created a new DriveManager instance")
                except Exception as e:
                    logger.warning(f"Failed to create DriveManager instance: {e}")
                    
        # Import other necessary components from the module
        if hasattr(main_module, 'SCOPES'):
            SCOPES = main_module.SCOPES
        if hasattr(main_module, 'DRIVE_FOLDERS'):
            DRIVE_FOLDERS = main_module.DRIVE_FOLDERS
        if hasattr(main_module, 'GOOGLE_API_AVAILABLE'):
            GOOGLE_API_AVAILABLE = main_module.GOOGLE_API_AVAILABLE
        if hasattr(main_module, 'DriveManager'):
            DriveManager = main_module.DriveManager
            
        # These functions will be defined as wrappers below, don't import directly
                
        logger.debug(f"Successfully imported using absolute import from {import_path}")
    except ImportError as e:
        logger.warning(f"Could not import {import_path}: {e}")
        # Direct import fallback not implemented, continue with minimal functionality
        main_module = None
except Exception as e:
    logger.error(f"Error during import: {e}")
    main_module = None

# Define functions to access the main implementation

def test_authentication():
    """Test authentication with Google Drive."""
    if main_module and hasattr(main_module, 'test_authentication'):
        # To avoid infinite recursion, don't call main_module.test_authentication() directly
        # Instead, if we have access to drive_manager, use it directly
        if hasattr(main_module, 'drive_manager') and hasattr(main_module.drive_manager, 'authenticate'):
            return main_module.drive_manager.authenticate()
        elif hasattr(main_module, '_drive_manager') and hasattr(main_module._drive_manager, 'authenticate'):
            return main_module._drive_manager.authenticate()
        else:
            # Final fallback - create a new instance
            if hasattr(main_module, 'DriveManager'):
                try:
                    temp_manager = main_module.DriveManager()
                    return temp_manager.authenticate()
                except Exception as e:
                    logger.error(f"Error creating DriveManager: {e}")
    
    # If all else fails, use our local drive_manager
    logger.warning("Falling back to local drive_manager.authenticate()")
    return drive_manager.authenticate()

def sync_to_drive(local_path, drive_folder_key, delete_source=False, update_only=False):
    """Sync files to Google Drive."""
    if main_module and hasattr(main_module, 'sync_to_drive'):
        # To avoid potential recursion, check if we're likely to call ourselves
        if main_module is not sys.modules[__name__]:
            return main_module.sync_to_drive(local_path, drive_folder_key, delete_source, update_only)
    
    # Direct implementation or fallback
    logger.warning("Using direct implementation of sync_to_drive")
    if hasattr(drive_manager, 'authenticate') and hasattr(drive_manager, 'upload_file'):
        if not drive_manager.authenticated and not drive_manager.authenticate():
            return False
            
        if os.path.isdir(local_path):
            if hasattr(drive_manager, 'upload_folder'):
                return drive_manager.upload_folder(local_path, drive_folder_key, delete_source)
            else:
                logger.error("upload_folder method not available")
                return False
        else:
            return drive_manager.upload_file(local_path, drive_folder_key, delete_source)
    else:
        logger.error("Drive manager doesn't have required methods")
        return False

def sync_from_drive(drive_folder_key, local_path):
    """Sync files from Google Drive."""
    if main_module and hasattr(main_module, 'sync_from_drive'):
        # To avoid potential recursion, check if we're likely to call ourselves
        if main_module is not sys.modules[__name__]:
            return main_module.sync_from_drive(drive_folder_key, local_path)
    
    # Direct implementation or fallback
    logger.warning("Using direct implementation of sync_from_drive")
    if not drive_manager.authenticated and not drive_manager.authenticate():
        return False
        
    folder_id = drive_manager.folder_ids.get(drive_folder_key)
    if not folder_id:
        logger.error(f"Drive folder key not found: {drive_folder_key}")
        return False
    
    if hasattr(drive_manager, 'download_folder'):
        return drive_manager.download_folder(folder_id, local_path)
    else:
        logger.error("download_folder method not available")
        return False

def configure_sync_method(use_rclone=False, base_dir="DeepseekCoder"):
    """Configure the sync method."""
    if main_module and hasattr(main_module, 'configure_sync_method'):
        # To avoid potential recursion, check if we're likely to call ourselves
        if main_module is not sys.modules[__name__]:
            return main_module.configure_sync_method(use_rclone, base_dir)
    
    # Direct implementation or fallback
    logger.warning("Using direct implementation of configure_sync_method")
    drive_manager.base_dir = base_dir
    if hasattr(drive_manager, 'authenticate'):
        drive_manager.authenticate()
    return drive_manager

def test_drive_mounting():
    """Test if Google Drive is mounted."""
    if main_module and hasattr(main_module, 'test_drive_mounting'):
        # To avoid potential recursion, check if we're likely to call ourselves
        if main_module is not sys.modules[__name__]:
            return main_module.test_drive_mounting()
    
    # Direct implementation or fallback
    drive_path = '/content/drive'
    is_mounted = os.path.exists(drive_path) and os.path.ismount(drive_path)
    
    if is_mounted:
        logger.info("Google Drive is mounted")
    else:
        logger.warning("Google Drive is not mounted")
    
    return is_mounted

def setup_drive_directories(manager=None, base_dir=None):
    """Set up directory structure on Google Drive."""
    if main_module and hasattr(main_module, 'setup_drive_directories'):
        # To avoid potential recursion, check if we're likely to call ourselves
        if main_module is not sys.modules[__name__]:
            return main_module.setup_drive_directories(manager, base_dir)
    
    # Direct implementation or fallback
    logger.warning("Using direct implementation of setup_drive_directories")
    mgr = manager or drive_manager
    
    # Set base directory if provided
    if base_dir and hasattr(mgr, 'base_dir'):
        mgr.base_dir = base_dir
    
    # Authenticate if needed
    if hasattr(mgr, 'authenticate') and not mgr.authenticated:
        if not mgr.authenticate():
            logger.error("Failed to authenticate")
            return {}
    
    # Set up folder structure if possible
    if hasattr(mgr, '_setup_drive_structure'):
        mgr._setup_drive_structure()
        return mgr.folder_ids
    
    return {} 