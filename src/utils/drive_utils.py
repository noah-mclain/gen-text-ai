"""
Google Drive API integration for Paperspace environments.
This module provides functions to access Google Drive without requiring FUSE mounting.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple, Any
import io

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
TOKEN_PATH = os.path.expanduser('~/.drive_token.json')
CREDENTIALS_PATH = os.path.join(os.getcwd(), 'credentials.json')

# Global service object
_drive_service = None

def get_credentials(credentials_path: str = CREDENTIALS_PATH, token_path: str = TOKEN_PATH) -> Optional[Credentials]:
    """
    Get Google Drive API credentials.
    
    Args:
        credentials_path: Path to the credentials.json file
        token_path: Path to save/load the token
        
    Returns:
        Credentials object or None if unavailable
    """
    if not GOOGLE_API_AVAILABLE:
        logger.error("Google API libraries not available")
        return None
        
    creds = None
    
    # Check if token exists and is valid
    if os.path.exists(token_path):
        try:
            creds = Credentials.from_authorized_user_info(
                json.loads(open(token_path).read()), SCOPES)
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
            if not os.path.exists(credentials_path):
                logger.error(f"Credentials file not found at {credentials_path}")
                return None
                
            try:
                flow = InstalledAppFlow.from_client_secrets_file(
                    credentials_path, SCOPES)
                
                # For headless environment, use console authentication
                creds = flow.run_console()
                
                # Save the credentials for future use
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())
                logger.info(f"Authentication successful. Token saved to {token_path}")
            except Exception as e:
                logger.error(f"Error during authentication: {e}")
                return None
    
    return creds

def get_drive_service() -> Any:
    """
    Build and return the Drive API service object.
    Uses a global service object to avoid repeated initialization.
    
    Returns:
        Drive API service object or None if unavailable
    """
    global _drive_service
    
    if _drive_service:
        return _drive_service
        
    if not GOOGLE_API_AVAILABLE:
        logger.error("Google API libraries not available")
        return None
    
    creds = get_credentials()
    if not creds:
        return None
        
    try:
        _drive_service = build('drive', 'v3', credentials=creds)
        return _drive_service
    except Exception as e:
        logger.error(f"Error building Drive service: {e}")
        return None

def is_drive_mounted() -> bool:
    """
    Check if Google Drive API is authenticated and ready to use.
    
    Returns:
        True if Drive is accessible, False otherwise
    """
    service = get_drive_service()
    if not service:
        return False
        
    try:
        # Try a simple request to check if API is working
        service.files().list(pageSize=1).execute()
        return True
    except Exception as e:
        logger.error(f"Drive API check failed: {e}")
        return False

def find_or_create_folder(folder_name: str, parent_id: Optional[str] = None) -> Optional[str]:
    """
    Find or create a folder in Google Drive.
    
    Args:
        folder_name: Name of the folder to find or create
        parent_id: ID of the parent folder (None for root)
        
    Returns:
        Folder ID or None if operation failed
    """
    service = get_drive_service()
    if not service:
        return None
    
    # Query to find existing folder
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    
    try:
        response = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        files = response.get('files', [])
        
        # Return existing folder ID if found
        if files:
            logger.info(f"Found existing folder: {folder_name} (ID: {files[0]['id']})")
            return files[0]['id']
        
        # Create new folder if not found
        folder_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        
        if parent_id:
            folder_metadata['parents'] = [parent_id]
            
        folder = service.files().create(body=folder_metadata, fields='id').execute()
        logger.info(f"Created new folder: {folder_name} (ID: {folder['id']})")
        return folder['id']
    except Exception as e:
        logger.error(f"Error finding/creating folder {folder_name}: {e}")
        return None

def setup_drive_directories(base_dir: str) -> Dict[str, str]:
    """
    Set up a directory structure in Google Drive.
    
    Args:
        base_dir: Base directory name to create
        
    Returns:
        Dictionary mapping directory names to their Google Drive IDs
    """
    if not is_drive_mounted():
        logger.error("Google Drive is not accessible. Cannot set up directories.")
        return {}
    
    # Split base_dir into components and create folder hierarchy
    base_name = os.path.basename(base_dir)
    dir_ids = {}
    
    # Create base directory
    base_id = find_or_create_folder(base_name)
    if not base_id:
        return {}
    
    dir_ids['base'] = base_id
    
    # Create subdirectories
    subdirs = ['models', 'data', 'logs', 'checkpoints']
    for subdir in subdirs:
        subdir_id = find_or_create_folder(subdir, base_id)
        if subdir_id:
            dir_ids[subdir] = subdir_id
    
    # Create data subdirectories
    if 'data' in dir_ids:
        data_subdirs = ['raw', 'processed']
        for data_subdir in data_subdirs:
            data_subdir_id = find_or_create_folder(data_subdir, dir_ids['data'])
            if data_subdir_id:
                dir_ids[f'data/{data_subdir}'] = data_subdir_id
    
    return dir_ids

def get_drive_path(local_path: str, drive_base_path: str, fallback_path: Optional[str] = None) -> str:
    """
    This doesn't literally map a path but serves as a utility for logical path mapping.
    For the Drive API implementation, we use it to provide a consistent interface.
    
    Args:
        local_path: Local path to the file/directory
        drive_base_path: Base path in Drive for mapping
        fallback_path: Fallback path to use if mapping fails
        
    Returns:
        Path for logical reference (not a real filesystem path)
    """
    if not is_drive_mounted():
        return fallback_path or local_path
    
    # Just concatenate paths to create a logical path reference
    # For the Drive API version, this is just for consistency with the interface
    rel_path = os.path.relpath(local_path, os.getcwd()) if os.path.isabs(local_path) else local_path
    drive_path = os.path.join(drive_base_path, rel_path)
    
    return drive_path

def save_model_to_drive(model_path: str, drive_dir: str, model_name: Optional[str] = None) -> bool:
    """
    Save a model directory to Google Drive.
    
    Args:
        model_path: Local path to the model directory
        drive_dir: Google Drive folder ID or logical path
        model_name: Optional name to use in Drive (defaults to directory name)
        
    Returns:
        True if successful, False otherwise
    """
    if not is_drive_mounted():
        logger.error("Google Drive is not accessible.")
        return False
    
    if not os.path.exists(model_path):
        logger.error(f"Model path {model_path} does not exist.")
        return False
    
    service = get_drive_service()
    if not service:
        return False
    
    # Get the folder ID if a path was provided
    folder_id = drive_dir
    if not folder_id.startswith("1") and "/" in folder_id:
        # This is a path, not an ID - find or create the folder
        folder_name = os.path.basename(drive_dir)
        folder_id = find_or_create_folder(folder_name)
        if not folder_id:
            return False
    
    # Default model name to directory name
    if not model_name:
        model_name = os.path.basename(model_path)
    
    # If model_path is a directory, upload all files
    if os.path.isdir(model_path):
        success = True
        for root, _, files in os.walk(model_path):
            for filename in files:
                local_path = os.path.join(root, filename)
                rel_path = os.path.relpath(local_path, model_path)
                
                # Create necessary parent folders
                parent_id = folder_id
                if os.path.dirname(rel_path):
                    for parent in Path(os.path.dirname(rel_path)).parts:
                        parent_id = find_or_create_folder(parent, parent_id)
                        if not parent_id:
                            success = False
                            break
                
                if not parent_id:
                    continue
                
                # Upload file
                try:
                    file_metadata = {'name': os.path.basename(filename), 'parents': [parent_id]}
                    media = MediaFileUpload(local_path)
                    service.files().create(body=file_metadata, media_body=media, fields='id').execute()
                    logger.info(f"Uploaded {rel_path} to Drive")
                except Exception as e:
                    logger.error(f"Error uploading {rel_path}: {e}")
                    success = False
        
        return success
    else:
        # Upload single file
        try:
            file_metadata = {'name': model_name, 'parents': [folder_id]}
            media = MediaFileUpload(model_path)
            service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            logger.info(f"Uploaded {model_path} to Drive")
            return True
        except Exception as e:
            logger.error(f"Error uploading {model_path}: {e}")
            return False

def download_file_from_drive(file_id: str, local_path: str) -> bool:
    """
    Download a file from Google Drive.
    
    Args:
        file_id: Google Drive file ID
        local_path: Local path to save the file
        
    Returns:
        True if successful, False otherwise
    """
    service = get_drive_service()
    if not service:
        return False
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download file
        request = service.files().get_media(fileId=file_id)
        
        with open(local_path, 'wb') as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                logger.info(f"Download {int(status.progress() * 100)}%")
        
        logger.info(f"Downloaded file to {local_path}")
        return True
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return False

def find_files_by_name(name: str, folder_id: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Find files in Google Drive by name.
    
    Args:
        name: Name of the file(s) to find
        folder_id: Optional folder ID to search in
        
    Returns:
        List of file metadata dictionaries
    """
    service = get_drive_service()
    if not service:
        return []
    
    query = f"name contains '{name}' and trashed=false"
    if folder_id:
        query += f" and '{folder_id}' in parents"
    
    try:
        response = service.files().list(q=query, spaces='drive', 
                                      fields='files(id, name, mimeType, size, modifiedTime)').execute()
        return response.get('files', [])
    except Exception as e:
        logger.error(f"Error finding files: {e}")
        return []

def mount_google_drive() -> bool:
    """
    Set up Google Drive API for use (not actual mounting).
    
    Returns:
        True if successful, False otherwise
    """
    # Check if API libraries are available
    if not GOOGLE_API_AVAILABLE:
        logger.error("Google API libraries not installed. Cannot mount Drive.")
        return False
    
    # Check if credentials.json exists
    if not os.path.exists(CREDENTIALS_PATH):
        logger.error(f"credentials.json not found at {CREDENTIALS_PATH}")
        logger.info("Please create OAuth 2.0 credentials in Google Cloud Console and download as credentials.json")
        return False
    
    # Initialize the Drive service
    service = get_drive_service()
    if not service:
        return False
    
    try:
        # Test API access
        service.files().list(pageSize=1).execute()
        logger.info("Google Drive API setup successful. Drive is accessible.")
        return True
    except Exception as e:
        logger.error(f"Failed to access Google Drive API: {e}")
        return False 