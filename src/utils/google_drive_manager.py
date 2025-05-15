#!/usr/bin/env python3
"""
Google Drive Manager

This module provides functionality to interact with Google Drive
for uploading, downloading, and syncing files and folders.
"""

import os
import sys
import json
import logging
from pathlib import Path
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import Google API libraries
GOOGLE_API_AVAILABLE = False
try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
    import io
    GOOGLE_API_AVAILABLE = True
    logger.info("Google API libraries successfully imported")
except ImportError as e:
    logger.warning(f"Google API libraries not available: {e}")
    logger.warning("Install with: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")

# Define Drive folders structure
DRIVE_FOLDERS = {
    "models": "models",
    "datasets": "datasets",
    "raw_data": "datasets/raw",
    "processed_data": "datasets/processed",
    "results": "results",
    "logs": "logs",
    "configs": "configs"
}

# Define OAuth 2.0 scopes
SCOPES = ['https://www.googleapis.com/auth/drive']

class DriveManager:
    def __init__(self, base_dir="DeepseekCoder", token_path="token.json", credentials_path="credentials.json"):
        """
        Initialize DriveManager with the given base directory.
        
        Args:
            base_dir: Base directory name in Google Drive
            token_path: Path to save/load the token file
            credentials_path: Path to the credentials.json file
        """
        self.base_dir = base_dir
        self.folder_ids = {}
        self.service = None
        self.token_path = token_path
        self.credentials_path = credentials_path
        
        # Make sure Google API is available
        if not GOOGLE_API_AVAILABLE:
            logger.error("Google API libraries not available. DriveManager will not work.")
            logger.error("Install with: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
    
    def authenticate(self):
        """
        Authenticate with Google Drive API.
        
        Returns:
            True if authentication was successful, False otherwise
        """
        if not GOOGLE_API_AVAILABLE:
            logger.error("Google API libraries not available. Cannot authenticate.")
            return False
            
        try:
            creds = None
            # Check if token file exists
            if os.path.exists(self.token_path):
                try:
                    creds = Credentials.from_authorized_user_info(
                        json.load(open(self.token_path)), SCOPES)
                except Exception as e:
                    logger.warning(f"Error loading token file: {e}")
            
            # If no valid credentials, authenticate
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    # Check if credentials file exists
                    if not os.path.exists(self.credentials_path):
                        logger.error(f"Credentials file not found: {self.credentials_path}")
                        return False
                        
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_path, SCOPES)
                    creds = flow.run_local_server(port=0)
                
                # Save the credentials for the next run
                with open(self.token_path, 'w') as token:
                    token.write(creds.to_json())
            
            # Build the Drive service
            self.service = build('drive', 'v3', credentials=creds)
            
            # Set up drive structure
            self._setup_drive_structure()
            
            logger.info("Successfully authenticated with Google Drive")
            return True
        except Exception as e:
            logger.error(f"Error authenticating with Google Drive: {e}")
            return False
    
    def _setup_drive_structure(self):
        """Set up the folder structure on Google Drive."""
        try:
            # Find root folder
            root_id = self._find_or_create_folder(self.base_dir, None)
            if root_id is None:
                logger.error(f"Failed to create root folder '{self.base_dir}'")
                return False
                
            self.folder_ids['base'] = root_id
            
            # Create folder structure
            for key, path in DRIVE_FOLDERS.items():
                parts = path.split('/')
                parent_id = root_id
                
                # Create folders in path
                for i, part in enumerate(parts):
                    # Create or find folder
                    folder_id = self._find_or_create_folder(part, parent_id)
                    if folder_id is None:
                        logger.error(f"Failed to create folder '{part}' in path '{path}'")
                        continue
                        
                    parent_id = folder_id
                    
                    # Save the ID for the full path
                    if i == len(parts) - 1:
                        self.folder_ids[key] = folder_id
            
            logger.info(f"Drive structure set up with {len(self.folder_ids)} folders")
            return True
        except Exception as e:
            logger.error(f"Error setting up drive structure: {e}")
            return False
    
    def _find_or_create_folder(self, folder_name, parent_id):
        """
        Find or create a folder on Google Drive.
        
        Args:
            folder_name: Name of the folder
            parent_id: ID of the parent folder, or None for root
            
        Returns:
            ID of the folder
        """
        # Check if service is available
        if self.service is None:
            logger.error("Google Drive service not initialized. Call authenticate() first.")
            return None
            
        # Check if folder exists
        folder_id = self.find_file_id(folder_name, parent_id, is_folder=True)
        
        if folder_id:
            return folder_id
        
        # Create folder if it doesn't exist
        try:
            folder_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            
            if parent_id:
                folder_metadata['parents'] = [parent_id]
            
            folder = self.service.files().create(
                body=folder_metadata,
                fields='id'
            ).execute()
            
            logger.info(f"Created folder '{folder_name}' with ID: {folder.get('id')}")
            return folder.get('id')
        except Exception as e:
            logger.error(f"Error creating folder '{folder_name}': {e}")
            return None
    
    def find_file_id(self, file_name, parent_id=None, is_folder=False):
        """
        Find a file or folder on Google Drive.
        
        Args:
            file_name: Name of the file or folder
            parent_id: ID of the parent folder, or None for root
            is_folder: True if looking for a folder, False for a file
            
        Returns:
            ID of the file or folder, or None if not found
        """
        # Check if service is available
        if self.service is None:
            logger.error("Google Drive service not initialized. Call authenticate() first.")
            return None
            
        query = f"name = '{file_name}'"
        
        if is_folder:
            query += " and mimeType = 'application/vnd.google-apps.folder'"
        
        if parent_id:
            query += f" and '{parent_id}' in parents"
        
        try:
            results = self.service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name)'
            ).execute()
            
            items = results.get('files', [])
            
            if items:
                return items[0]['id']
            return None
        except Exception as e:
            logger.error(f"Error finding file ID for {file_name}: {e}")
            return None
    
    def upload_file(self, local_path, drive_folder_key, delete_source=False):
        """
        Upload a file to Google Drive.
        
        Args:
            local_path: Path to the local file
            drive_folder_key: Key for the destination folder
            delete_source: Whether to delete the source file after upload
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Make sure file exists
            if not os.path.exists(local_path):
                logger.error(f"File not found: {local_path}")
                return False
            
            # Get folder ID
            folder_id = self.folder_ids.get(drive_folder_key)
            if not folder_id:
                logger.error(f"Drive folder key not found: {drive_folder_key}")
                return False
            
            # Create media
            file_name = os.path.basename(local_path)
            file_metadata = {
                'name': file_name,
                'parents': [folder_id]
            }
            
            media = MediaFileUpload(
                local_path,
                resumable=True
            )
            
            # Upload file
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            logger.info(f"Uploaded file '{file_name}' to Drive folder '{drive_folder_key}'")
            
            # Delete source if requested
            if delete_source:
                os.remove(local_path)
                logger.info(f"Deleted source file: {local_path}")
            
            return True
        except Exception as e:
            logger.error(f"Error uploading file {local_path}: {e}")
            return False
    
    def download_file(self, file_id, local_path):
        """
        Download a file from Google Drive.
        
        Args:
            file_id: ID of the file to download
            local_path: Path to save the file locally
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Make sure parent directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Request file
            request = self.service.files().get_media(fileId=file_id)
            
            # Download file
            with open(local_path, 'wb') as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    logger.debug(f"Download progress: {int(status.progress() * 100)}%")
            
            logger.info(f"Downloaded file to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Error downloading file {file_id}: {e}")
            return False
    
    def upload_folder(self, local_folder, drive_folder_key, delete_source=False):
        """
        Upload a folder to Google Drive.
        
        Args:
            local_folder: Path to the local folder
            drive_folder_key: Key for the destination folder
            delete_source: Whether to delete the source folder after upload
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Make sure folder exists
            if not os.path.exists(local_folder) or not os.path.isdir(local_folder):
                logger.error(f"Folder not found: {local_folder}")
                return False
            
            # Get folder ID
            parent_id = self.folder_ids.get(drive_folder_key)
            if not parent_id:
                logger.error(f"Drive folder key not found: {drive_folder_key}")
                return False
            
            # Create folder on Drive
            folder_name = os.path.basename(local_folder)
            folder_id = self._find_or_create_folder(folder_name, parent_id)
            
            # Upload contents
            success = True
            for item in os.listdir(local_folder):
                item_path = os.path.join(local_folder, item)
                
                if os.path.isdir(item_path):
                    # Create subfolder
                    subfolder_id = self._find_or_create_folder(item, folder_id)
                    
                    # Upload contents recursively
                    for subitem in os.listdir(item_path):
                        subitem_path = os.path.join(item_path, subitem)
                        
                        if os.path.isfile(subitem_path):
                            # Upload file
                            file_metadata = {
                                'name': subitem,
                                'parents': [subfolder_id]
                            }
                            
                            media = MediaFileUpload(
                                subitem_path,
                                resumable=True
                            )
                            
                            self.service.files().create(
                                body=file_metadata,
                                media_body=media,
                                fields='id'
                            ).execute()
                            
                            logger.debug(f"Uploaded file '{subitem}' to subfolder '{item}'")
                else:
                    # Upload file
                    file_metadata = {
                        'name': item,
                        'parents': [folder_id]
                    }
                    
                    media = MediaFileUpload(
                        item_path,
                        resumable=True
                    )
                    
                    self.service.files().create(
                        body=file_metadata,
                        media_body=media,
                        fields='id'
                    ).execute()
                    
                    logger.debug(f"Uploaded file '{item}' to folder '{folder_name}'")
            
            logger.info(f"Uploaded folder '{folder_name}' to Drive folder '{drive_folder_key}'")
            
            # Delete source if requested
            if delete_source:
                shutil.rmtree(local_folder)
                logger.info(f"Deleted source folder: {local_folder}")
            
            return success
        except Exception as e:
            logger.error(f"Error uploading folder {local_folder}: {e}")
            return False
    
    def download_folder(self, folder_id, local_folder):
        """
        Download a folder from Google Drive.
        
        Args:
            folder_id: ID of the folder to download
            local_folder: Path to save the folder locally
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Make sure local folder exists
            os.makedirs(local_folder, exist_ok=True)
            
            # List files in the folder
            query = f"'{folder_id}' in parents"
            results = self.service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name, mimeType)'
            ).execute()
            
            items = results.get('files', [])
            
            if not items:
                logger.warning(f"No files found in folder {folder_id}")
                return True
            
            # Download each item
            for item in items:
                name = item['name']
                item_id = item['id']
                mime_type = item['mimeType']
                
                if mime_type == 'application/vnd.google-apps.folder':
                    # Create subfolder
                    subfolder_path = os.path.join(local_folder, name)
                    os.makedirs(subfolder_path, exist_ok=True)
                    
                    # Download contents recursively
                    self.download_folder(item_id, subfolder_path)
                else:
                    # Download file
                    file_path = os.path.join(local_folder, name)
                    self.download_file(item_id, file_path)
            
            logger.info(f"Downloaded folder to {local_folder}")
            return True
        except Exception as e:
            logger.error(f"Error downloading folder {folder_id}: {e}")
            return False

# Create a global instance
_drive_manager = DriveManager()
# Add an alias without underscore for external imports
drive_manager = _drive_manager

# Helper functions

def test_authentication():
    """Test authentication with Google Drive."""
    global _drive_manager
    return _drive_manager.authenticate()

def setup_drive_directories(manager=None, base_dir=None):
    """
    Set up directory structure on Google Drive.
    
    Args:
        manager: DriveManager instance (optional)
        base_dir: Base directory name
        
    Returns:
        Dict of folder IDs
    """
    # Use provided manager or global instance
    mgr = manager or _drive_manager
    
    # Set base directory if provided
    if base_dir and mgr.base_dir != base_dir:
        mgr = DriveManager(base_dir=base_dir)
    
    # Ensure authentication is complete
    if not mgr.authenticate():
        logger.error("Failed to authenticate with Google Drive")
        return {}
    
    # Set up drive structure
    if mgr._setup_drive_structure():
        return mgr.folder_ids
    return {}

def sync_to_drive(local_path, drive_folder_key, delete_source=False, update_only=False):
    """
    Sync files to Google Drive.
    
    Args:
        local_path: Path to local file or folder
        drive_folder_key: Key for destination folder
        delete_source: Whether to delete the source after upload
        update_only: Whether to only upload files that don't exist on Drive
        
    Returns:
        True if successful, False otherwise
    """
    global _drive_manager
    
    if not GOOGLE_API_AVAILABLE:
        logger.error("Google API not available")
        return False
    
    # Authenticate if needed
    if not _drive_manager.service:
        if not _drive_manager.authenticate():
            return False
    
    # Upload file or folder
    if os.path.isdir(local_path):
        return _drive_manager.upload_folder(local_path, drive_folder_key, delete_source)
    else:
        return _drive_manager.upload_file(local_path, drive_folder_key, delete_source)

def sync_from_drive(drive_folder_key, local_path):
    """
    Sync files from Google Drive.
    
    Args:
        drive_folder_key: Key for source folder
        local_path: Path to save locally
        
    Returns:
        True if successful, False otherwise
    """
    global _drive_manager
    
    if not GOOGLE_API_AVAILABLE:
        logger.error("Google API not available")
        return False
    
    # Authenticate if needed
    if not _drive_manager.service:
        if not _drive_manager.authenticate():
            return False
    
    # Get folder ID
    folder_id = _drive_manager.folder_ids.get(drive_folder_key)
    if not folder_id:
        logger.error(f"Drive folder key not found: {drive_folder_key}")
        return False
    
    # Download folder
    return _drive_manager.download_folder(folder_id, local_path)

def configure_sync_method(use_rclone=False, base_dir="DeepseekCoder"):
    """
    Configure the sync method.
    
    Args:
        use_rclone: Whether to use rclone instead of the API
        base_dir: Base directory name in Google Drive
        
    Returns:
        DriveManager instance
    """
    global _drive_manager
    
    if use_rclone:
        logger.warning("rclone sync method not implemented yet")
    
    # Reconfigure drive manager
    if _drive_manager.base_dir != base_dir:
        _drive_manager = DriveManager(base_dir=base_dir)
        _drive_manager.authenticate()
    
    return _drive_manager

def test_drive_mounting():
    """Test if Google Drive is mounted."""
    # Check if Drive is mounted at /content/drive
    drive_path = '/content/drive'
    is_mounted = os.path.exists(drive_path) and os.path.ismount(drive_path)
    
    if is_mounted:
        logger.info("Google Drive is mounted")
    else:
        logger.warning("Google Drive is not mounted")
    
    return is_mounted 