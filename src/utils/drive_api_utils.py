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

# Add project root to Python path to handle both local and Paperspace environments
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    logger.info(f"Added project root to Python path: {project_root}")

# Try multiple import paths to handle different environments
drive_utils_imported = False

# Try options in priority order
import_attempts = [
    # 1. Try direct import first (local path)
    {"module": "utils.google_drive_manager", "log": "direct import"},
    # 2. Try absolute import with project in path
    {"module": "src.utils.google_drive_manager", "log": "absolute import"},
    # 3. Try scripts path (new structure)
    {"module": "scripts.google_drive.google_drive_manager", "log": "scripts import"},
    # 4. Try scripts implementation file directly
    {"module": "scripts.google_drive.google_drive_manager_impl", "log": "scripts implementation import"}
]

for attempt in import_attempts:
    if drive_utils_imported:
        break
        
    try:
        module_name = attempt["module"]
        # Use importlib to dynamically import
        import importlib
        logger.info(f"Attempting to import {module_name}...")
        module = importlib.import_module(module_name)
        
        # Import all necessary components
        try:
            DriveManager = getattr(module, "DriveManager", None)
            if DriveManager is None:
                logger.warning(f"DriveManager class not found in {module_name}")
                continue
                
            drive_manager = getattr(module, "_drive_manager", None)
            sync_to_drive = getattr(module, "sync_to_drive", None)
            if sync_to_drive is None:
                logger.warning(f"sync_to_drive function not found in {module_name}")
                continue
                
            sync_from_drive = getattr(module, "sync_from_drive", None)
            configure_sync_method = getattr(module, "configure_sync_method", None)
            test_authentication = getattr(module, "test_authentication", None)
            test_drive_mounting = getattr(module, "test_drive_mounting", None)
            setup_drive_directories = getattr(module, "setup_drive_directories", None)
            DRIVE_FOLDERS = getattr(module, "DRIVE_FOLDERS", {})
            SCOPES = getattr(module, "SCOPES", [])
            GOOGLE_API_AVAILABLE = getattr(module, "GOOGLE_API_AVAILABLE", True)
            
            logger.info(f"Successfully imported from {module_name} via {attempt['log']}")
            drive_utils_imported = True
            DRIVE_API_AVAILABLE = GOOGLE_API_AVAILABLE
        except AttributeError as e:
            logger.warning(f"Failed to import components from {module_name}: {e}")
    
    except ImportError as e:
        logger.debug(f"Failed to import from {attempt['module']}: {e}")

# If all imports failed, try explicit file path import
if not drive_utils_imported:
    try:
        # Try to locate the implementation file
        impl_paths = [
            os.path.join(project_root, "src", "utils", "google_drive_manager.py"),
            os.path.join(project_root, "scripts", "google_drive", "google_drive_manager_impl.py")
        ]
        
        impl_file = None
        for path in impl_paths:
            if os.path.exists(path):
                impl_file = path
                logger.info(f"Found implementation file at {impl_file}")
                break
        
        if impl_file:
            # Use manual approach to load module from file
            import importlib.util
            logger.info(f"Attempting to import directly from file: {impl_file}")
            
            module_name = "google_drive_manager_manual"
            spec = importlib.util.spec_from_file_location(module_name, impl_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Import necessary components
            DriveManager = module.DriveManager
            drive_manager = getattr(module, "_drive_manager", None)
            sync_to_drive = module.sync_to_drive
            sync_from_drive = module.sync_from_drive
            configure_sync_method = module.configure_sync_method
            test_authentication = module.test_authentication
            test_drive_mounting = getattr(module, "test_drive_mounting", None)
            DRIVE_FOLDERS = module.DRIVE_FOLDERS
            SCOPES = getattr(module, "SCOPES", [])
            GOOGLE_API_AVAILABLE = getattr(module, "GOOGLE_API_AVAILABLE", True)
            
            logger.info(f"Successfully imported from file: {impl_file}")
            drive_utils_imported = True
            DRIVE_API_AVAILABLE = GOOGLE_API_AVAILABLE
    except Exception as e:
        logger.error(f"Failed to import from implementation file: {e}")

# If all attempts failed, fallback to dummy implementations
if not drive_utils_imported:
    logger.error("All import attempts for Drive API utils failed")
    logger.warning("Drive API functionality will not be available")
    
    # Define empty constants
    DRIVE_FOLDERS = {}
    SCOPES = []
    
    # Define dummy class
    class DriveManager:
        def __init__(self, *args, **kwargs):
            logger.error("DriveManager not available - google_drive_manager could not be imported")
            self.folder_ids = {}
            self.base_dir = "DeepseekCoder"
        
        def authenticate(self):
            return False
            
        def find_file_id(self, *args, **kwargs):
            return None
            
        def upload_folder(self, *args, **kwargs):
            return False
            
        def upload_file(self, *args, **kwargs):
            return False
            
        def download_folder(self, *args, **kwargs):
            return False
            
        def _setup_drive_structure(self):
            pass
    
    # Create dummy global instance
    drive_manager = DriveManager()
    
    # Define dummy functions
    def sync_to_drive(*args, **kwargs):
        logger.error("Drive sync function not available")
        return False
        
    def sync_from_drive(*args, **kwargs):
        logger.error("Drive sync function not available")
        return False
        
    def configure_sync_method(*args, **kwargs):
        logger.error("Drive sync configuration not available")
        
    def test_authentication(*args, **kwargs):
        logger.error("Drive authentication test not available")
        return False
        
    def test_drive_mounting(*args, **kwargs):
        logger.error("Drive mounting test not available")
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
    'setup_drive_directories',
    'DRIVE_API_AVAILABLE',
    'DRIVE_FOLDERS',
    'SCOPES'
] 