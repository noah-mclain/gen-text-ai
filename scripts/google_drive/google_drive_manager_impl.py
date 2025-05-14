"""
Google Drive Manager Implementation

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
        return next((path for path in CREDENTIALS_PATHS if os.path.exists(path)), None)
    
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
            File ID if found, None otherwise
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
                fields='files(id, name, mimeType)'
            ).execute()
            
            files = response.get('files', [])
            if not files:
                return None
            
            # Return the first match
            return files[0]['id']
            
        except Exception as e:
            logger.error(f"Error finding file {name}: {str(e)}")
            return None
    
    def create_folder(self, name: str, parent_id: str = None) -> Optional[str]:
        """
        Create a folder in Google Drive.
        
        Args:
            name: Name of the folder
            parent_id: Parent folder ID (optional)
            
        Returns:
            Folder ID if created successfully, None otherwise
        """
        if not self.authenticated:
            if not self.authenticate():
                return None
        
        try:
            # Prepare folder metadata
            file_metadata = {
                'name': name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            
            if parent_id:
                file_metadata['parents'] = [parent_id]
            
            # Create the folder
            file = self.service.files().create(
                body=file_metadata,
                fields='id'
            ).execute()
            
            logger.info(f"Created folder: {name} (ID: {file.get('id')})")
            return file.get('id')
            
        except Exception as e:
            logger.error(f"Error creating folder {name}: {str(e)}")
            return None
    
    def upload_file(self, local_path: str, remote_path: str, delete_source: bool = False) -> bool:
        """
        Upload a file to Google Drive.
        
        Args:
            local_path: Path to the local file
            remote_path: Path in Drive (folder key from DRIVE_FOLDERS)
            delete_source: Whether to delete the source file after upload
            
        Returns:
            True if upload was successful, False otherwise
        """
        if not self.authenticated:
            if not self.authenticate():
                return False
        
        try:
            # Check that the local file exists
            if not os.path.exists(local_path) or not os.path.isfile(local_path):
                logger.error(f"Local file not found: {local_path}")
                return False
            
            # Get the folder ID for the remote path
            folder_id = None
            if remote_path in self.folder_ids:
                folder_id = self.folder_ids[remote_path]
            else:
                logger.error(f"Remote path not found: {remote_path}")
                return False
            
            # Check if file already exists
            file_name = os.path.basename(local_path)
            existing_file_id = self.find_file_id(file_name, folder_id)
            
            # Prepare file metadata
            file_metadata = {
                'name': file_name,
            }
            
            if not existing_file_id:
                file_metadata['parents'] = [folder_id]
            
            # Create media
            media = MediaFileUpload(
                local_path,
                resumable=True
            )
            
            if existing_file_id:
                # Update existing file
                file = self.service.files().update(
                    fileId=existing_file_id,
                    body=file_metadata,
                    media_body=media,
                    fields='id'
                ).execute()
                logger.info(f"Updated file: {file_name} (ID: {file.get('id')})")
            else:
                # Upload new file
                file = self.service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id'
                ).execute()
                logger.info(f"Uploaded file: {file_name} (ID: {file.get('id')})")
            
            # Delete source file if requested
            if delete_source:
                os.remove(local_path)
                logger.info(f"Deleted source file: {local_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error uploading file {local_path}: {str(e)}")
            return False
    
    def upload_folder(self, local_path: str, remote_path: str, delete_source: bool = False) -> bool:
        """
        Upload a folder to Google Drive.
        
        Args:
            local_path: Path to the local folder
            remote_path: Path in Drive (folder key from DRIVE_FOLDERS)
            delete_source: Whether to delete the source folder after upload
            
        Returns:
            True if upload was successful, False otherwise
        """
        if not self.authenticated:
            if not self.authenticate():
                return False
        
        try:
            # Check that the local folder exists
            if not os.path.exists(local_path) or not os.path.isdir(local_path):
                logger.error(f"Local folder not found: {local_path}")
                return False
            
            # Get the folder ID for the remote path
            parent_folder_id = None
            if remote_path in self.folder_ids:
                parent_folder_id = self.folder_ids[remote_path]
            else:
                logger.error(f"Remote path not found: {remote_path}")
                return False
            
            # Create a folder for the upload
            folder_name = os.path.basename(local_path)
            folder_id = self.find_file_id(folder_name, parent_folder_id)
            if not folder_id:
                folder_id = self.create_folder(folder_name, parent_folder_id)
                if not folder_id:
                    logger.error(f"Failed to create folder: {folder_name}")
                    return False
            
            # Upload all files in the folder
            success = True
            for root, dirs, files in os.walk(local_path):
                # Calculate relative path from the local folder
                rel_path = os.path.relpath(root, local_path)
                current_folder_id = folder_id
                
                # Navigate to or create subfolder in Drive
                if rel_path != '.':
                    # Split the relative path into components
                    path_components = rel_path.split(os.sep)
                    
                    for component in path_components:
                        # Find or create the subfolder
                        subfolder_id = self.find_file_id(component, current_folder_id)
                        if not subfolder_id:
                            subfolder_id = self.create_folder(component, current_folder_id)
                            if not subfolder_id:
                                logger.error(f"Failed to create subfolder: {component}")
                                success = False
                                continue
                        
                        current_folder_id = subfolder_id
                
                # Upload files in this directory
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    
                    # Prepare file metadata
                    file_metadata = {
                        'name': file_name,
                        'parents': [current_folder_id]
                    }
                    
                    # Check if file already exists
                    existing_file_id = self.find_file_id(file_name, current_folder_id)
                    
                    # Create media
                    media = MediaFileUpload(
                        file_path,
                        resumable=True
                    )
                    
                    try:
                        if existing_file_id:
                            # Update existing file
                            file = self.service.files().update(
                                fileId=existing_file_id,
                                body={'name': file_name},
                                media_body=media,
                                fields='id'
                            ).execute()
                            logger.debug(f"Updated file: {file_name} (ID: {file.get('id')})")
                        else:
                            # Upload new file
                            file = self.service.files().create(
                                body=file_metadata,
                                media_body=media,
                                fields='id'
                            ).execute()
                            logger.debug(f"Uploaded file: {file_name} (ID: {file.get('id')})")
                    except Exception as e:
                        logger.error(f"Error uploading file {file_path}: {str(e)}")
                        success = False
            
            # Delete source folder if requested and upload was successful
            if delete_source and success:
                import shutil
                shutil.rmtree(local_path)
                logger.info(f"Deleted source folder: {local_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error uploading folder {local_path}: {str(e)}")
            return False
    
    def download_file(self, file_id: str, local_path: str) -> bool:
        """
        Download a file from Google Drive.
        
        Args:
            file_id: File ID in Google Drive
            local_path: Local path to save the file
            
        Returns:
            True if download was successful, False otherwise
        """
        if not self.authenticated:
            if not self.authenticate():
                return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download the file
            request = self.service.files().get_media(fileId=file_id)
            with open(local_path, 'wb') as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    logger.debug(f"Download progress: {int(status.progress() * 100)}%")
            
            logger.info(f"Downloaded file to: {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading file {file_id}: {str(e)}")
            return False
    
    def download_folder(self, folder_id: str, local_path: str) -> bool:
        """
        Download a folder from Google Drive.
        
        Args:
            folder_id: Folder ID in Google Drive
            local_path: Local path to save the folder
            
        Returns:
            True if download was successful, False otherwise
        """
        if not self.authenticated:
            if not self.authenticate():
                return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(local_path, exist_ok=True)
            
            # Get all files in the folder
            query = f"'{folder_id}' in parents and trashed = false"
            response = self.service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name, mimeType)'
            ).execute()
            
            files = response.get('files', [])
            success = True
            
            for file in files:
                file_name = file['name']
                file_id = file['id']
                mime_type = file['mimeType']
                
                if mime_type == 'application/vnd.google-apps.folder':
                    # Create subfolder and download recursively
                    subfolder_path = os.path.join(local_path, file_name)
                    os.makedirs(subfolder_path, exist_ok=True)
                    if not self.download_folder(file_id, subfolder_path):
                        success = False
                else:
                    # Download file
                    file_path = os.path.join(local_path, file_name)
                    if not self.download_file(file_id, file_path):
                        success = False
            
            return success
            
        except Exception as e:
            logger.error(f"Error downloading folder {folder_id}: {str(e)}")
            return False


# Singleton instance for reuse
_drive_manager = None

def _get_drive_manager(base_dir: str = "DeepseekCoder") -> DriveManager:
    """Get or create the DriveManager singleton instance."""
    global _drive_manager
    if _drive_manager is None:
        _drive_manager = DriveManager(base_dir=base_dir)
    elif _drive_manager.base_dir != base_dir:
        # Reinitialize if base directory changed
        _drive_manager = DriveManager(base_dir=base_dir)
    return _drive_manager

# Global configuration for sync method
_use_rclone = False
_drive_base_dir = "DeepseekCoder"

def sync_to_drive(local_path: str, drive_folder_key: str, 
                delete_source: bool = False, update_only: bool = False) -> bool:
    """
    Sync local files to Google Drive.
    
    Args:
        local_path: Path to the local file or folder
        drive_folder_key: Key of the folder in Drive (from DRIVE_FOLDERS)
        delete_source: Whether to delete the source after upload
        update_only: Only update existing files, don't add new ones
        
    Returns:
        True if sync was successful, False otherwise
    """
    if _use_rclone:
        return _sync_to_drive_rclone(local_path, drive_folder_key, delete_source)
    else:
        manager = _get_drive_manager(_drive_base_dir)
        if os.path.isdir(local_path):
            return manager.upload_folder(local_path, drive_folder_key, delete_source)
        else:
            return manager.upload_file(local_path, drive_folder_key, delete_source)

def sync_from_drive(drive_folder_key: str, local_path: str) -> bool:
    """
    Sync files from Google Drive to local storage.
    
    Args:
        drive_folder_key: Key of the folder in Drive (from DRIVE_FOLDERS)
        local_path: Path to save locally
        
    Returns:
        True if sync was successful, False otherwise
    """
    if _use_rclone:
        return _sync_from_drive_rclone(drive_folder_key, local_path)
    else:
        manager = _get_drive_manager(_drive_base_dir)
        if drive_folder_key not in manager.folder_ids:
            return False
        return manager.download_folder(manager.folder_ids[drive_folder_key], local_path)

def _sync_to_drive_rclone(local_path: str, drive_folder_key: str, delete_source: bool = False) -> bool:
    """Sync to Drive using rclone."""
    import subprocess
    import shutil
    
    # Check if rclone is installed
    if shutil.which("rclone") is None:
        logger.error("rclone is not installed. Please install it first.")
        return False
    
    try:
        # Construct the remote path
        if drive_folder_key in DRIVE_FOLDERS:
            remote_path = f"gdrive:{_drive_base_dir}/{DRIVE_FOLDERS[drive_folder_key]}"
        else:
            remote_path = f"gdrive:{_drive_base_dir}/{drive_folder_key}"
        
        # Build rclone command
        cmd = ["rclone", "sync"]
        if delete_source:
            cmd = ["rclone", "move"]
        
        cmd.extend([local_path, remote_path, "--progress"])
        
        # Run the command
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)
        
        return result.returncode == 0
        
    except subprocess.CalledProcessError as e:
        logger.error(f"rclone error: {e}")
        return False
    except Exception as e:
        logger.error(f"Error using rclone: {e}")
        return False

def _sync_from_drive_rclone(drive_folder_key: str, local_path: str) -> bool:
    """Sync from Drive using rclone."""
    import subprocess
    import shutil
    
    # Check if rclone is installed
    if shutil.which("rclone") is None:
        logger.error("rclone is not installed. Please install it first.")
        return False
    
    try:
        # Construct the remote path
        if drive_folder_key in DRIVE_FOLDERS:
            remote_path = f"gdrive:{_drive_base_dir}/{DRIVE_FOLDERS[drive_folder_key]}"
        else:
            remote_path = f"gdrive:{_drive_base_dir}/{drive_folder_key}"
        
        # Ensure the local directory exists
        os.makedirs(local_path, exist_ok=True)
        
        # Build rclone command
        cmd = ["rclone", "sync", remote_path, local_path, "--progress"]
        
        # Run the command
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)
        
        return result.returncode == 0
        
    except subprocess.CalledProcessError as e:
        logger.error(f"rclone error: {e}")
        return False
    except Exception as e:
        logger.error(f"Error using rclone: {e}")
        return False

def configure_sync_method(use_rclone: bool = False, base_dir: str = "DeepseekCoder"):
    """
    Configure the sync method.
    
    Args:
        use_rclone: Whether to use rclone for syncing
        base_dir: Base directory in Google Drive
    """
    global _use_rclone, _drive_base_dir
    _use_rclone = use_rclone
    _drive_base_dir = base_dir
    logger.info(f"Configured sync method: {'rclone' if use_rclone else 'API'}, base_dir: {base_dir}")

def test_authentication():
    """Test authentication with Google Drive."""
    manager = _get_drive_manager()
    return manager.authenticate()

def test_drive_mounting():
    """Test if Google Drive is mounted."""
    if not _use_rclone:
        return False
    
    import subprocess
    import shutil
    
    if shutil.which("rclone") is None:
        logger.error("rclone is not installed. Please install it first.")
        return False
    
    try:
        result = subprocess.run(
            ["rclone", "lsd", "gdrive:"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        return result.returncode == 0
    except Exception:
        return False

def mount_google_drive():
    """Mount Google Drive using rclone."""
    import subprocess
    import shutil
    
    if shutil.which("rclone") is None:
        logger.error("rclone is not installed. Please install it first.")
        return False
    
    try:
        # Check if already mounted
        if test_drive_mounting():
            logger.info("Google Drive already mounted")
            return True
        
        # Configure rclone if needed
        result = subprocess.run(
            ["rclone", "config", "show"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        if "gdrive" not in result.stdout:
            logger.error("rclone not configured for Google Drive. Run 'rclone config' first.")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error mounting Google Drive: {e}")
        return False

def setup_drive_directories(base_dir=None):
    """Set up directory structure in Google Drive."""
    if base_dir:
        global _drive_base_dir
        _drive_base_dir = base_dir
    
    manager = _get_drive_manager(_drive_base_dir)
    if not manager.authenticate():
        return False
    
    try:
        # Check if base directory exists
        base_id = manager.find_file_id(_drive_base_dir)
        if not base_id:
            logger.info(f"Creating base directory: {_drive_base_dir}")
            base_id = manager.create_folder(_drive_base_dir)
            if not base_id:
                logger.error(f"Failed to create base directory: {_drive_base_dir}")
                return False
        
        # Create all necessary folders
        for key, path in DRIVE_FOLDERS.items():
            folder_path = os.path.join(_drive_base_dir, path)
            logger.info(f"Setting up directory: {folder_path}")
            
            # Create each level of the path
            parent_id = base_id
            for part in path.split('/'):
                folder_id = manager.find_file_id(part, parent_id)
                if not folder_id:
                    folder_id = manager.create_folder(part, parent_id)
                    if not folder_id:
                        logger.error(f"Failed to create folder: {part}")
                        return False
                parent_id = folder_id
        
        logger.info("Drive directory structure setup complete")
        return True
    
    except Exception as e:
        logger.error(f"Error setting up drive directories: {e}")
        return False

def get_drive_path(local_path, drive_folder_id=None, fallback_path=None):
    """Get the corresponding path in Google Drive."""
    if drive_folder_id:
        return f"gdrive:{_drive_base_dir}/{drive_folder_id}/{os.path.basename(local_path)}"
    elif fallback_path:
        return f"gdrive:{_drive_base_dir}/{fallback_path}/{os.path.basename(local_path)}"
    else:
        return f"gdrive:{_drive_base_dir}/{os.path.basename(local_path)}" 