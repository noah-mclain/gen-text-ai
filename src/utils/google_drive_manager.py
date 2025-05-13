"""
Google Drive Manager

Provides authentication and utilities for interacting with Google Drive
in headless environments like Paperspace.
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import Google API libraries
try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
    from googleapiclient.errors import HttpError
    GOOGLE_API_AVAILABLE = True
except ImportError:
    logger.warning("Google API libraries not installed. Install with: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
    GOOGLE_API_AVAILABLE = False

# Default scopes
SCOPES = ['https://www.googleapis.com/auth/drive']

# Look for credentials in multiple locations
CREDENTIALS_PATHS = [
    os.path.join(os.getcwd(), 'credentials.json'),  # Current directory
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'credentials.json'),  # Project root
    os.path.expanduser('~/credentials.json'),  # Home directory
    '/notebooks/credentials.json',  # Paperspace notebooks directory
]

# Drive folder structure
DRIVE_FOLDERS = {
    "data": "data",
    "preprocessed": "data/processed",
    "raw": "data/raw",
    "models": "models",
    "checkpoints": "models/checkpoints",
    "logs": "logs",
    "results": "results",
    "train": "results/train",
    "test": "results/test",
    "eval": "results/eval",
    "validation": "results/validation",
    "visualizations": "visualizations",
    "cache": "cache",
    "text_models": "text_models",
    "text_checkpoints": "text_models/checkpoints",
    "text_logs": "logs/text_generation"
}

class DriveManager:
    """
    Manages Google Drive interactions using OAuth authentication.
    """
    
    def __init__(self, base_dir: str = "DeepseekCoder", token_path: str = None):
        """
        Initialize the Drive Manager.
        
        Args:
            base_dir: Base directory on Google Drive
            token_path: Path to save/load the token file
        """
        self.base_dir = base_dir
        self.token_path = token_path or os.path.expanduser('~/.drive_token.json')
        self.credentials_path = self._find_credentials_file()
        self.service = None
        self.authenticated = False
        self.folder_ids = {}
    
    def _find_credentials_file(self) -> Optional[str]:
        """Find the credentials.json file."""
        for path in CREDENTIALS_PATHS:
            if os.path.exists(path):
                return path
        return None
    
    def authenticate(self) -> bool:
        """
        Authenticate with Google Drive API using headless OAuth flow.
        
        Returns:
            True if authentication was successful, False otherwise
        """
        if not GOOGLE_API_AVAILABLE:
            logger.error("Google API libraries not available. Install required packages.")
            return False
        
        try:
            creds = None
            
            # Check if token exists and is valid
            if os.path.exists(self.token_path):
                try:
                    with open(self.token_path, 'r') as token_file:
                        creds = Credentials.from_authorized_user_info(json.load(token_file), SCOPES)
                except Exception as e:
                    logger.warning(f"Error loading token: {e}")
            
            # If token is invalid or missing, authenticate
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    try:
                        creds.refresh(Request())
                    except Exception as e:
                        logger.warning(f"Error refreshing token: {e}")
                        creds = None
                
                # If still no valid credentials, need to authenticate
                if not creds:
                    if not self.credentials_path or not os.path.exists(self.credentials_path):
                        logger.error(f"Credentials file not found. Please provide a valid credentials.json file.")
                        return False
                    
                    # Use a different approach to create the flow for headless authentication
                    try:
                        # Read client secret file
                        with open(self.credentials_path, 'r') as f:
                            client_config = json.load(f)
                        
                        # Create flow explicitly defining the redirect URI
                        from google_auth_oauthlib.flow import Flow
                        
                        # Determine if we're using installed app or web app client config
                        if 'installed' in client_config:
                            client_type = 'installed'
                        elif 'web' in client_config:
                            client_type = 'web'
                        else:
                            logger.error("Invalid client config format")
                            return False
                            
                        # Make sure the redirect URI is set correctly for OOB flow
                        redirect_uri = 'urn:ietf:wg:oauth:2.0:oob'
                        client_config[client_type]['redirect_uris'] = [redirect_uri]
                        
                        # Create the flow
                        flow = Flow.from_client_config(
                            client_config,
                            scopes=SCOPES,
                            redirect_uri=redirect_uri
                        )
                        
                        # Generate authorization URL - don't specify redirect_uri here again
                        auth_url, _ = flow.authorization_url(
                            access_type='offline',
                            include_granted_scopes='true',
                            prompt='consent'  # Force consent screen for refresh token
                        )
                    except Exception as e:
                        logger.warning(f"Error with custom flow approach: {e}, falling back to standard method")
                        
                        # Fallback to standard flow
                        flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, SCOPES)
                        auth_url, _ = flow.authorization_url(
                            access_type='offline',
                            include_granted_scopes='true',
                            prompt='consent'  # Force consent screen for refresh token
                        )
                    
                    print("\n" + "="*80)
                    print("Google Drive Authentication".center(80))
                    print("="*80)
                    print("\nTo authenticate with Google Drive, follow these steps:")
                    print("\n1. Visit this URL in your browser:")
                    print(f"\n{auth_url}\n")
                    print("2. Sign in and grant access to your Google Drive")
                    print("3. Copy the authorization code provided")
                    auth_code = input("\nEnter the authorization code: ")
                    
                    # Exchange auth code for credentials
                    flow.fetch_token(code=auth_code)
                    creds = flow.credentials
                    
                    # Save the credentials for future use
                    with open(self.token_path, 'w') as token:
                        token.write(creds.to_json())
                    logger.info(f"Authentication successful. Token saved to {self.token_path}")
            
            # Build the Drive service
            self.service = build('drive', 'v3', credentials=creds)
            self.authenticated = True
            
            # Test API access
            self.service.files().list(pageSize=1).execute()
            logger.info("Successfully authenticated with Google Drive API")
            
            # Setup directory structure
            self._setup_drive_structure()
            
            return True
            
        except Exception as e:
            logger.error(f"Error authenticating with Google Drive API: {str(e)}")
            return False
    
    def _setup_drive_structure(self):
        """Set up the directory structure in Google Drive."""
        try:
            # Find or create base directory
            base_id = self.find_file_id(self.base_dir)
            if not base_id:
                base_id = self.create_folder(self.base_dir)
                if not base_id:
                    logger.error(f"Failed to create base directory: {self.base_dir}")
                    return
            
            self.folder_ids['base'] = base_id
            
            # Create all necessary folders
            for key, path in DRIVE_FOLDERS.items():
                # Split path into components
                parts = path.split('/')
                current_id = base_id
                
                # Navigate and create each level of the path
                for i, part in enumerate(parts):
                    # Find or create folder
                    folder_id = self.find_file_id(part, current_id)
                    if not folder_id:
                        folder_id = self.create_folder(part, current_id)
                    
                    current_id = folder_id
                
                # Store the final ID for this path
                self.folder_ids[key] = current_id
                logger.debug(f"Folder ready: {path} (ID: {current_id})")
            
            logger.info(f"Drive folder structure setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up drive structure: {str(e)}")
    
    def find_file_id(self, name: str, parent_id: str = None) -> Optional[str]:
        """
        Find file or folder ID by name.
        
        Args:
            name: Name of the file or folder
            parent_id: Parent folder ID (optional)
            
        Returns:
            File/folder ID or None if not found
        """
        if not self.authenticated:
            if not self.authenticate():
                return None
        
        try:
            query = f"name = '{name}' and trashed = false"
            if parent_id:
                query += f" and '{parent_id}' in parents"
            
            response = self.service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name)'
            ).execute()
            
            files = response.get('files', [])
            
            if not files:
                return None
            
            return files[0].get('id')
            
        except Exception as e:
            logger.error(f"Error finding file '{name}': {str(e)}")
            return None
    
    def create_folder(self, name: str, parent_id: str = None) -> Optional[str]:
        """
        Create a folder in Google Drive.
        
        Args:
            name: Folder name
            parent_id: Parent folder ID (optional)
            
        Returns:
            Folder ID or None if creation failed
        """
        if not self.authenticated:
            if not self.authenticate():
                return None
        
        try:
            file_metadata = {
                'name': name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            
            if parent_id:
                file_metadata['parents'] = [parent_id]
            
            folder = self.service.files().create(
                body=file_metadata,
                fields='id'
            ).execute()
            
            folder_id = folder.get('id')
            logger.info(f"Created folder: {name}")
            
            return folder_id
            
        except Exception as e:
            logger.error(f"Error creating folder '{name}': {str(e)}")
            return None
    
    def upload_file(self, local_path: str, remote_path: str, delete_source: bool = False) -> bool:
        """
        Upload a file to Google Drive.
        
        Args:
            local_path: Path to local file
            remote_path: Path in drive (folder key or path relative to base_dir)
            delete_source: Whether to delete the local file after upload
            
        Returns:
            True if successful, False otherwise
        """
        if not self.authenticated:
            if not self.authenticate():
                return False
        
        if not os.path.exists(local_path):
            logger.error(f"Local file not found: {local_path}")
            return False
        
        try:
            # Get folder ID from folder key or path
            folder_id = None
            if remote_path in self.folder_ids:
                folder_id = self.folder_ids[remote_path]
            else:
                # Handle path traversal
                parts = remote_path.split('/')
                current_id = self.folder_ids.get('base')
                
                for i, part in enumerate(parts[:-1] if len(parts) > 1 else parts):
                    folder_id = self.find_file_id(part, current_id)
                    if not folder_id:
                        folder_id = self.create_folder(part, current_id)
                    current_id = folder_id
                
                folder_id = current_id
            
            if not folder_id:
                logger.error(f"Failed to find or create destination folder: {remote_path}")
                return False
            
            # Upload the file
            file_name = os.path.basename(local_path)
            
            # Check if file already exists
            existing_file_id = self.find_file_id(file_name, folder_id)
            if existing_file_id:
                # Delete existing file
                self.service.files().delete(fileId=existing_file_id).execute()
                logger.debug(f"Replaced existing file: {file_name}")
            
            # Create the new file
            file_metadata = {'name': file_name, 'parents': [folder_id]}
            media = MediaFileUpload(local_path, resumable=True)
            
            self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            logger.info(f"Uploaded: {file_name} to {remote_path}")
            
            # Delete source if requested
            if delete_source:
                os.remove(local_path)
                logger.info(f"Deleted local file: {local_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error uploading file {local_path}: {str(e)}")
            return False
    
    def upload_folder(self, local_path: str, remote_path: str, delete_source: bool = False) -> bool:
        """
        Upload a folder to Google Drive.
        
        Args:
            local_path: Path to local folder
            remote_path: Path in drive (folder key or path relative to base_dir)
            delete_source: Whether to delete the local folder after upload
            
        Returns:
            True if successful, False otherwise
        """
        if not self.authenticated:
            if not self.authenticate():
                return False
        
        if not os.path.exists(local_path) or not os.path.isdir(local_path):
            logger.error(f"Local directory not found: {local_path}")
            return False
        
        try:
            # Get or create folder ID
            folder_name = os.path.basename(local_path)
            
            # Get parent folder ID from folder key or path
            parent_id = None
            if remote_path in self.folder_ids:
                parent_id = self.folder_ids[remote_path]
            else:
                # Handle path traversal
                parts = remote_path.split('/')
                current_id = self.folder_ids.get('base')
                
                for part in parts:
                    folder_id = self.find_file_id(part, current_id)
                    if not folder_id:
                        folder_id = self.create_folder(part, current_id)
                    current_id = folder_id
                
                parent_id = current_id
            
            if not parent_id:
                logger.error(f"Failed to find or create destination folder: {remote_path}")
                return False
            
            # Create the folder in Drive (or find existing one)
            folder_id = self.find_file_id(folder_name, parent_id)
            if not folder_id:
                folder_id = self.create_folder(folder_name, parent_id)
                if not folder_id:
                    logger.error(f"Failed to create folder: {folder_name}")
                    return False
            
            # Upload all contents
            success = True
            for item in os.listdir(local_path):
                item_path = os.path.join(local_path, item)
                
                if os.path.isdir(item_path):
                    # Recursively upload subdirectory
                    subdir_success = self.upload_folder(
                        item_path, 
                        f"{remote_path}/{folder_name}", 
                        delete_source
                    )
                    success = success and subdir_success
                else:
                    # Upload file
                    file_success = self.upload_file(
                        item_path, 
                        f"{remote_path}/{folder_name}", 
                        delete_source
                    )
                    success = success and file_success
            
            # Delete source folder if requested and all uploads succeeded
            if delete_source and success:
                import shutil
                shutil.rmtree(local_path)
                logger.info(f"Deleted local directory: {local_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error uploading folder {local_path}: {str(e)}")
            return False
    
    def download_file(self, file_id: str, local_path: str) -> bool:
        """
        Download a file from Google Drive.
        
        Args:
            file_id: ID of the file to download
            local_path: Local path to save the file
            
        Returns:
            True if successful, False otherwise
        """
        if not self.authenticated:
            if not self.authenticate():
                return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
            
            # Download the file
            request = self.service.files().get_media(fileId=file_id)
            
            with open(local_path, 'wb') as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
            
            logger.info(f"Downloaded file to: {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading file to {local_path}: {str(e)}")
            return False
    
    def download_folder(self, folder_id: str, local_path: str) -> bool:
        """
        Download a folder from Google Drive.
        
        Args:
            folder_id: ID of the folder to download
            local_path: Local path to save the folder
            
        Returns:
            True if successful, False otherwise
        """
        if not self.authenticated:
            if not self.authenticate():
                return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(local_path, exist_ok=True)
            
            # List all files in the folder
            query = f"'{folder_id}' in parents and trashed = false"
            response = self.service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name, mimeType)'
            ).execute()
            
            files = response.get('files', [])
            
            if not files:
                logger.info(f"No files found in folder (ID: {folder_id})")
                return True
            
            # Download each file/folder
            success = True
            for file in files:
                file_id = file.get('id')
                file_name = file.get('name')
                mime_type = file.get('mimeType')
                
                local_file_path = os.path.join(local_path, file_name)
                
                if mime_type == 'application/vnd.google-apps.folder':
                    # Recursively download subfolder
                    folder_success = self.download_folder(file_id, local_file_path)
                    success = success and folder_success
                else:
                    # Download file
                    file_success = self.download_file(file_id, local_file_path)
                    success = success and file_success
            
            return success
            
        except Exception as e:
            logger.error(f"Error downloading folder to {local_path}: {str(e)}")
            return False

# Create a global instance for easy imports
drive_manager = DriveManager()

def sync_to_drive(local_path: str, drive_folder_key: str, 
                delete_source: bool = False, update_only: bool = False) -> bool:
    """
    Sync local files/folders to Google Drive.
    
    Args:
        local_path: Path to local file or folder
        drive_folder_key: Key for destination folder or path relative to base_dir
        delete_source: Whether to delete local files after sync
        update_only: Not used (for compatibility with rclone version)
        
    Returns:
        True if successful, False otherwise
    """
    if os.path.isdir(local_path):
        return drive_manager.upload_folder(local_path, drive_folder_key, delete_source)
    else:
        return drive_manager.upload_file(local_path, drive_folder_key, delete_source)

def sync_from_drive(drive_folder_key: str, local_path: str) -> bool:
    """
    Sync files/folders from Google Drive to local.
    
    Args:
        drive_folder_key: Key for the source folder or path relative to base_dir
        local_path: Path to save locally
        
    Returns:
        True if successful, False otherwise
    """
    # Find folder ID for the drive folder key
    folder_id = None
    
    # Check if it's a direct key in DRIVE_FOLDERS
    if drive_folder_key in drive_manager.folder_ids:
        folder_id = drive_manager.folder_ids[drive_folder_key]
    elif drive_folder_key in DRIVE_FOLDERS:
        # Get the path and find the corresponding ID
        drive_path = DRIVE_FOLDERS[drive_folder_key]
        
        if drive_path in drive_manager.folder_ids:
            folder_id = drive_manager.folder_ids[drive_path]
        else:
            # Navigate the path to find the ID
            parts = drive_path.split('/')
            current_id = drive_manager.folder_ids.get('base')
            
            for part in parts:
                folder_id = drive_manager.find_file_id(part, current_id)
                if not folder_id:
                    logger.error(f"Folder not found: {part} in {drive_path}")
                    return False
                current_id = folder_id
            
            folder_id = current_id
    else:
        # Treat as direct path
        parts = drive_folder_key.split('/')
        current_id = drive_manager.folder_ids.get('base')
        
        for part in parts:
            folder_id = drive_manager.find_file_id(part, current_id)
            if not folder_id:
                logger.error(f"Folder not found: {part} in {drive_folder_key}")
                return False
            current_id = folder_id
        
        folder_id = current_id
    
    if not folder_id:
        logger.error(f"Could not find folder ID for: {drive_folder_key}")
        return False
    
    return drive_manager.download_folder(folder_id, local_path)

def configure_sync_method(use_rclone: bool = False, base_dir: str = "DeepseekCoder"):
    """
    Configure the global drive_manager instance.
    
    Args:
        use_rclone: Ignored (kept for backward compatibility)
        base_dir: Base directory name on Google Drive
    """
    global drive_manager
    drive_manager = DriveManager(base_dir=base_dir)
    return drive_manager

def test_authentication():
    """Test if authentication works correctly."""
    logger.info("Testing authentication...")
    
    # Try to authenticate
    success = drive_manager.authenticate()
    
    if success:
        logger.info("✓ Authentication successful!")
    else:
        logger.error("✗ Authentication failed")
    
    return success

def test_drive_mounting():
    """Test if Drive access works correctly."""
    if not drive_manager.authenticated:
        if not test_authentication():
            return False
    
    try:
        # Try listing files (minimal permission test)
        drive_manager.service.files().list(pageSize=1).execute()
        logger.info("✓ Google Drive access successful!")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to access Google Drive: {str(e)}")
        return False

def mount_google_drive():
    """
    Mount Google Drive for Colab compatibility.
    
    This is a stub method for compatibility with code that expects to mount Google Drive in Colab.
    On non-Colab environments, it just verifies that the Drive API is available and authenticated.
    
    Returns:
        True if drive is available/mounted, False otherwise
    """
    # Check if we're running in Colab
    try:
        from google.colab import drive
        # We're in Colab, use native mount
        logger.info("Running in Colab. Mounting Google Drive...")
        drive.mount('/content/drive')
        return os.path.isdir('/content/drive/MyDrive')
    except ImportError:
        # We're not in Colab, check if Drive API is available
        logger.info("Not running in Colab. Using Drive API instead of mounting.")
        return test_authentication()
        
def setup_drive_directories(base_dir=None):
    """
    Set up directory structure on Google Drive.
    
    Args:
        base_dir: Base directory on Google Drive (can be a path or a folder name)
        
    Returns:
        Dict of folder IDs or empty dict on failure
    """
    if not GOOGLE_API_AVAILABLE:
        logger.error("Google API libraries not available")
        return {}
        
    # Create a new manager with the specified base directory if needed
    global drive_manager
    if base_dir and drive_manager.base_dir != base_dir:
        new_manager = DriveManager(base_dir=base_dir)
        if new_manager.authenticate():
            drive_manager = new_manager
        else:
            logger.error(f"Failed to authenticate with new base directory: {base_dir}")
            return {}
            
    # Make sure we're authenticated
    if not drive_manager.authenticated and not drive_manager.authenticate():
        logger.error("Failed to authenticate with Google Drive")
        return {}
        
    # Set up directory structure
    drive_manager._setup_drive_structure()
    return drive_manager.folder_ids
    
def get_drive_path(local_path, drive_folder_id=None, fallback_path=None):
    """
    Get the equivalent path on Google Drive.
    
    Args:
        local_path: Local file or directory path
        drive_folder_id: ID of the Drive folder or folder key from DRIVE_FOLDERS
        fallback_path: Fallback path to use if Drive is not available
        
    Returns:
        Path on Google Drive or fallback path
    """
    if not GOOGLE_API_AVAILABLE or not drive_manager.authenticated:
        return fallback_path or local_path
        
    # If drive_folder_id is a key in DRIVE_FOLDERS, get the actual ID
    folder_id = None
    
    if drive_folder_id in drive_manager.folder_ids:
        folder_id = drive_manager.folder_ids[drive_folder_id]
    elif drive_folder_id in DRIVE_FOLDERS:
        folder_key = drive_folder_id
        if folder_key in drive_manager.folder_ids:
            folder_id = drive_manager.folder_ids[folder_key]
            
    if not folder_id:
        return fallback_path or local_path
        
    # Construct a Drive path (actually just returns a reference, not a real path)
    # This is mostly for compatibility with code that expects Colab Drive paths
    drive_path = f"drive://{folder_id}/{os.path.basename(local_path)}"
    return drive_path

# Define all exported symbols for clarity
__all__ = [
    # Classes
    'DriveManager',
    
    # Module-level instances
    'drive_manager',
    
    # Core functions
    'sync_to_drive',
    'sync_from_drive',
    'configure_sync_method',
    
    # Utility functions
    'test_authentication',
    'test_drive_mounting',
    'mount_google_drive',
    'setup_drive_directories',
    'get_drive_path',
    
    # Constants
    'DRIVE_FOLDERS',
    'SCOPES',
    'GOOGLE_API_AVAILABLE'
]

# Initialize global drive manager
drive_manager = DriveManager() 