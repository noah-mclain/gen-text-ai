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

# If no implementation is found, copy it from the current directory if available
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
        raise ImportError("google_drive_manager.py implementation not found")

# Add the directory of the implementation to sys.path
sys.path.append(os.path.dirname(os.path.dirname(implementation_path)))

# Try to import from the implementation
try:
    # Import relative to root
    import_path = "src.utils.google_drive_manager"
    try:
        # Try importing using absolute import
        module = __import__(import_path, fromlist=['*'])
        # Get all attributes from the module
        globals().update({name: getattr(module, name) for name in dir(module) 
                          if not name.startswith('_') or name == '__all__'})
        logger.debug(f"Successfully imported using absolute import from {import_path}")
    except ImportError:
        # Try direct import as fallback
        implementation_dir = os.path.dirname(implementation_path)
        if implementation_dir not in sys.path:
            sys.path.append(implementation_dir)
        
        module_name = os.path.splitext(os.path.basename(implementation_path))[0]
        module = __import__(module_name, fromlist=['*'])
        globals().update({name: getattr(module, name) for name in dir(module) 
                        if not name.startswith('_') or name == '__all__'})
        logger.debug(f"Successfully imported using direct import from {module_name}")
        
except ImportError as e:
    logger.error(f"Failed to import from {implementation_path}: {e}")
    logger.error("Make sure the python path includes the project root directory")
    raise

# Inform users about the redirect
logger.debug(f"google_drive_manager redirected to {implementation_path}")

# Expose important classes and variables
try:
    # Explicitly define DriveManager
    DriveManager = getattr(module, 'DriveManager', None)
    if DriveManager is None:
        # Create a dummy DriveManager class if not available
        class DriveManager:
            def __init__(self, *args, **kwargs):
                logger.warning("Using dummy DriveManager class")
                self.base_dir = kwargs.get('base_dir', 'DeepseekCoder')
                self.folder_ids = {}
                self.credentials_path = kwargs.get('credentials_path', 'credentials.json')
            
            def authenticate(self):
                logger.warning("Dummy authenticate method called")
                return False
            
            def _setup_drive_structure(self):
                logger.warning("Dummy _setup_drive_structure method called")
                return False
                
            def find_file_id(self, *args, **kwargs):
                return None
                
            def upload_folder(self, *args, **kwargs):
                return False
                
            def upload_file(self, *args, **kwargs):
                return False
                
            def download_folder(self, *args, **kwargs):
                return False
    
    # Create a default instance
    _drive_manager = getattr(module, '_drive_manager', None)
    if _drive_manager is None:
        _drive_manager = DriveManager()
    
    # Get other important variables with defaults
    DRIVE_FOLDERS = getattr(module, 'DRIVE_FOLDERS', {})
    SCOPES = getattr(module, 'SCOPES', [])
    GOOGLE_API_AVAILABLE = getattr(module, 'GOOGLE_API_AVAILABLE', False)
    
except Exception as e:
    logger.error(f"Error setting up DriveManager: {e}")
    # Create a dummy class as fallback
    class DriveManager:
        def __init__(self, *args, **kwargs):
            logger.warning("Using fallback dummy DriveManager class")
            self.base_dir = kwargs.get('base_dir', 'DeepseekCoder')
            self.folder_ids = {}
            self.credentials_path = kwargs.get('credentials_path', 'credentials.json')
        
        def authenticate(self):
            logger.warning("Dummy authenticate method called")
            return False
        
        def _setup_drive_structure(self):
            logger.warning("Dummy _setup_drive_structure method called")
            return False
            
        def find_file_id(self, *args, **kwargs):
            return None
            
        def upload_folder(self, *args, **kwargs):
            return False
            
        def upload_file(self, *args, **kwargs):
            return False
            
        def download_folder(self, *args, **kwargs):
            return False
    
    # Create default values
    _drive_manager = DriveManager()
    DRIVE_FOLDERS = {}
    SCOPES = []
    GOOGLE_API_AVAILABLE = False

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