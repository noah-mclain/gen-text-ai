import os
import io
import logging
import tempfile
import pickle
from pathlib import Path
from typing import Optional, Dict, List, BinaryIO, Union

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define scope
SCOPES = ['https://www.googleapis.com/auth/drive']

class DriveAPI:
    """Class to interact with Google Drive API directly instead of mounting."""
    
    def __init__(self, credentials_path: str = 'credentials.json', token_path: str = 'token.pickle'):
        """
        Initialize the DriveAPI.
        
        Args:
            credentials_path: Path to the credentials.json file downloaded from Google Cloud Console
            token_path: Path to save/load the authentication token
        """
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.creds = None
        self.service = None
        self.authenticated = False
        self.file_id_cache = {}
        
    def authenticate(self, headless: bool = False) -> bool:
        """
        Authenticate with Google Drive API.
        
        Args:
            headless: If True, use console flow instead of local server (for headless environments)
            
        Returns:
            True if authentication was successful, False otherwise
        """
        try:
            # Load credentials from token.pickle if it exists
            if os.path.exists(self.token_path):
                with open(self.token_path, 'rb') as token:
                    self.creds = pickle.load(token)
            
            # If no valid credentials available, let the user log in
            if not self.creds or not self.creds.valid:
                if self.creds and self.creds.expired and self.creds.refresh_token:
                    self.creds.refresh(Request())
                else:
                    if not os.path.exists(self.credentials_path):
                        logger.error(f"Credentials file not found at {self.credentials_path}")
                        return False
                    
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_path, SCOPES)
                    
                    if headless:
                        # Use console flow for headless environments
                        logger.info("Using console-based authentication flow for headless environment")
                        self.creds = flow.run_console()
                    else:
                        # Use local server flow for environments with a browser
                        self.creds = flow.run_local_server(port=0)
                
                # Save the credentials for the next run
                with open(self.token_path, 'wb') as token:
                    pickle.dump(self.creds, token)
            
            # Build the Drive service
            self.service = build('drive', 'v3', credentials=self.creds)
            self.authenticated = True
            logger.info("Successfully authenticated with Google Drive API")
            return True
            
        except Exception as e:
            logger.error(f"Error authenticating with Google Drive API: {str(e)}")
            return False
    
    def _ensure_authenticated(self, headless: bool = False) -> bool:
        """Ensure API is authenticated before use."""
        if not self.authenticated:
            return self.authenticate(headless=headless)
        return True
    
    def create_folder(self, folder_name: str, parent_id: Optional[str] = None) -> Optional[str]:
        """
        Create a folder in Google Drive.
        
        Args:
            folder_name: Name of the folder to create
            parent_id: ID of the parent folder (if None, folder will be created in the root)
            
        Returns:
            ID of the created folder, or None if creation failed
        """
        if not self._ensure_authenticated():
            return None
        
        try:
            file_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            
            if parent_id:
                file_metadata['parents'] = [parent_id]
                
            folder = self.service.files().create(
                body=file_metadata,
                fields='id'
            ).execute()
            
            folder_id = folder.get('id')
            logger.info(f"Created folder: {folder_name} with ID: {folder_id}")
            
            # Cache the folder ID
            self.file_id_cache[folder_name] = folder_id
            
            return folder_id
            
        except HttpError as error:
            logger.error(f"Error creating folder: {str(error)}")
            return None
    
    def find_file_id(self, name: str, folder_id: Optional[str] = None) -> Optional[str]:
        """
        Find the ID of a file or folder by name.
        
        Args:
            name: Name of the file or folder to find
            folder_id: ID of the parent folder to search in (if None, search everywhere)
            
        Returns:
            ID of the file or folder, or None if not found
        """
        if not self._ensure_authenticated():
            return None
        
        # Check cache first
        if name in self.file_id_cache:
            return self.file_id_cache[name]
        
        try:
            query = f"name = '{name}'"
            if folder_id:
                query += f" and '{folder_id}' in parents"
                
            response = self.service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name)'
            ).execute()
            
            files = response.get('files', [])
            
            if not files:
                logger.debug(f"File/folder '{name}' not found")
                return None
                
            file_id = files[0].get('id')
            
            # Cache the result
            self.file_id_cache[name] = file_id
            
            return file_id
            
        except HttpError as error:
            logger.error(f"Error finding file: {str(error)}")
            return None
    
    def setup_directories(self, base_dir: str) -> Dict[str, str]:
        """
        Create necessary directories on Google Drive.
        
        Args:
            base_dir: Base directory name on Google Drive for the project
            
        Returns:
            Dictionary with folder names and their IDs
        """
        if not self._ensure_authenticated():
            return {}
        
        # Find or create base directory
        base_id = self.find_file_id(base_dir)
        if not base_id:
            base_id = self.create_folder(base_dir)
            if not base_id:
                logger.error(f"Failed to create base directory: {base_dir}")
                return {}
        
        # Define directory structure
        directories = {
            "data": "data",
            "preprocessed": "data/processed",
            "raw": "data/raw",
            "models": "models",
            "logs": "logs",
            "results": "results",
            "visualizations": "visualizations",
            "cache": "cache"
        }
        
        # Create directories and store their IDs
        directory_ids = {}
        
        for key, path in directories.items():
            # Split path into components
            parts = path.split('/')
            parent_id = base_id
            current_path = base_dir
            
            # Create each level of the path
            for part in parts:
                current_path = f"{current_path}/{part}"
                folder_id = self.find_file_id(part, parent_id)
                
                if not folder_id:
                    folder_id = self.create_folder(part, parent_id)
                    if not folder_id:
                        logger.error(f"Failed to create directory: {current_path}")
                        break
                
                parent_id = folder_id
            
            # Store the final directory ID
            if parent_id != base_id:
                directory_ids[key] = parent_id
                logger.info(f"Directory ready: {path} (ID: {parent_id})")
        
        return directory_ids
    
    def upload_file(self, local_path: str, remote_name: Optional[str] = None, 
                   parent_id: Optional[str] = None) -> Optional[str]:
        """
        Upload a file to Google Drive.
        
        Args:
            local_path: Path to the local file
            remote_name: Name to use in Drive (defaults to local filename)
            parent_id: ID of the parent folder (if None, file will be uploaded to root)
            
        Returns:
            ID of the uploaded file, or None if upload failed
        """
        if not self._ensure_authenticated():
            return None
        
        try:
            if not os.path.exists(local_path):
                logger.error(f"Local file not found: {local_path}")
                return None
            
            file_name = remote_name or os.path.basename(local_path)
            file_metadata = {'name': file_name}
            
            if parent_id:
                file_metadata['parents'] = [parent_id]
            
            media = MediaFileUpload(local_path, resumable=True)
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            file_id = file.get('id')
            logger.info(f"Uploaded file: {file_name} with ID: {file_id}")
            
            return file_id
            
        except HttpError as error:
            logger.error(f"Error uploading file: {str(error)}")
            return None
    
    def download_file(self, file_id: str, local_path: str) -> bool:
        """
        Download a file from Google Drive.
        
        Args:
            file_id: ID of the file to download
            local_path: Path where to save the downloaded file
            
        Returns:
            True if download was successful, False otherwise
        """
        if not self._ensure_authenticated():
            return False
        
        try:
            request = self.service.files().get_media(fileId=file_id)
            
            # Create directory if needed
            os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
            
            with open(local_path, 'wb') as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    logger.debug(f"Download progress: {int(status.progress() * 100)}%")
            
            logger.info(f"Downloaded file to: {local_path}")
            return True
            
        except HttpError as error:
            logger.error(f"Error downloading file: {str(error)}")
            return False
    
    def download_folder(self, folder_id: str, local_dir: str) -> bool:
        """
        Download entire folder from Google Drive.
        
        Args:
            folder_id: ID of the folder to download
            local_dir: Local directory to save the downloaded files
            
        Returns:
            True if download was successful, False otherwise
        """
        if not self._ensure_authenticated():
            return False
        
        try:
            # Create local directory
            os.makedirs(local_dir, exist_ok=True)
            
            # Get all files in the folder
            query = f"'{folder_id}' in parents"
            response = self.service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name, mimeType)'
            ).execute()
            
            files = response.get('files', [])
            
            if not files:
                logger.info(f"No files found in folder {folder_id}")
                return True
            
            success = True
            for file in files:
                file_id = file.get('id')
                file_name = file.get('name')
                mime_type = file.get('mimeType')
                
                local_path = os.path.join(local_dir, file_name)
                
                if mime_type == 'application/vnd.google-apps.folder':
                    # Recursive call for subfolders
                    subfolder_success = self.download_folder(file_id, local_path)
                    success = success and subfolder_success
                else:
                    # Download file
                    file_success = self.download_file(file_id, local_path)
                    success = success and file_success
            
            return success
            
        except HttpError as error:
            logger.error(f"Error downloading folder: {str(error)}")
            return False
    
    def upload_folder(self, local_dir: str, parent_id: Optional[str] = None) -> Optional[str]:
        """
        Upload an entire folder to Google Drive.
        
        Args:
            local_dir: Path to the local folder
            parent_id: ID of the parent folder (if None, folder will be uploaded to root)
            
        Returns:
            ID of the uploaded folder, or None if upload failed
        """
        if not self._ensure_authenticated():
            return None
        
        try:
            if not os.path.exists(local_dir) or not os.path.isdir(local_dir):
                logger.error(f"Local directory not found: {local_dir}")
                return None
            
            folder_name = os.path.basename(local_dir)
            folder_id = self.create_folder(folder_name, parent_id)
            
            if not folder_id:
                logger.error(f"Failed to create folder: {folder_name}")
                return None
            
            # Upload all files in the directory
            for item in os.listdir(local_dir):
                item_path = os.path.join(local_dir, item)
                
                if os.path.isdir(item_path):
                    # Recursive call for subdirectories
                    self.upload_folder(item_path, folder_id)
                else:
                    # Upload file
                    self.upload_file(item_path, parent_id=folder_id)
            
            return folder_id
            
        except HttpError as error:
            logger.error(f"Error uploading folder: {str(error)}")
            return None

# Helper functions to make the API easier to use

def initialize_drive_api(credentials_path: str = 'credentials.json', token_path: str = 'token.pickle', headless: bool = False) -> DriveAPI:
    """
    Initialize and authenticate the Drive API.
    
    Args:
        credentials_path: Path to the credentials.json file
        token_path: Path to save/load the token
        headless: If True, use console flow instead of local server (for headless environments)
        
    Returns:
        Authenticated DriveAPI instance
    """
    drive_api = DriveAPI(credentials_path, token_path)
    drive_api.authenticate(headless=headless)
    return drive_api

def setup_drive_directories(api: DriveAPI, base_dir: str) -> Dict[str, str]:
    """
    Setup the directory structure in Google Drive.
    
    Args:
        api: Authenticated DriveAPI instance
        base_dir: Base directory name
        
    Returns:
        Dictionary with directory names and IDs
    """
    return api.setup_directories(base_dir)

def save_to_drive(api: DriveAPI, local_path: str, folder_id: str) -> Optional[str]:
    """
    Save a file or folder to Google Drive.
    
    Args:
        api: Authenticated DriveAPI instance
        local_path: Path to the local file or folder
        folder_id: ID of the destination folder
        
    Returns:
        ID of the uploaded file/folder
    """
    if os.path.isdir(local_path):
        return api.upload_folder(local_path, folder_id)
    else:
        return api.upload_file(local_path, parent_id=folder_id)

def load_from_drive(api: DriveAPI, file_id: str, local_path: str) -> bool:
    """
    Load a file or folder from Google Drive.
    
    Args:
        api: Authenticated DriveAPI instance
        file_id: ID of the file or folder to download
        local_path: Path where to save the downloaded items
        
    Returns:
        True if download was successful
    """
    # Check if it's a folder
    try:
        file = api.service.files().get(fileId=file_id, fields='mimeType').execute()
        mime_type = file.get('mimeType')
        
        if mime_type == 'application/vnd.google-apps.folder':
            return api.download_folder(file_id, local_path)
        else:
            return api.download_file(file_id, local_path)
    except:
        return False

def is_api_authenticated(api: DriveAPI) -> bool:
    """Check if the API is authenticated."""
    return api.authenticated 