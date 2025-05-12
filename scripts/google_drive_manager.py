#!/usr/bin/env python3
"""
Google Drive Manager
===================

A comprehensive tool for setting up and managing Google Drive integration
with the gen-text-ai project. This script combines setup, authentication,
testing, and utility functions in one place.

Usage:
    python scripts/google_drive_manager.py --action setup
    python scripts/google_drive_manager.py --action test
    python scripts/google_drive_manager.py --action check
    python scripts/google_drive_manager.py --action create_folders
    python scripts/google_drive_manager.py --action upload --local_path path/to/file --remote_folder folder_id
    
For headless environments like Paperspace:
    python scripts/google_drive_manager.py --action setup --headless
    
Requirements:
    - google-api-python-client
    - google-auth-httplib2
    - google-auth-oauthlib
"""

import os
import sys
import json
import time
import logging
import argparse
import tempfile
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

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
    logger.error("Google API libraries not installed.")
    logger.info("Install with: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
    GOOGLE_API_AVAILABLE = False

# Define paths and scopes
CREDENTIALS_PATH = project_root / "credentials.json"
TOKEN_PATH = Path.home() / ".drive_token.json"
SCOPES = ['https://www.googleapis.com/auth/drive']

# Global service object
_drive_service = None

def create_credentials_template():
    """Create a template credentials.json file with instructions."""
    template = {
        "_comment": "Replace this template with your actual OAuth credentials from Google Cloud Console",
        "installed": {
            "client_id": "YOUR_CLIENT_ID.apps.googleusercontent.com",
            "project_id": "YOUR_PROJECT_ID",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_secret": "YOUR_CLIENT_SECRET",
            "redirect_uris": ["http://localhost:8080", "urn:ietf:wg:oauth:2.0:oob"]
        }
    }
    
    with open(CREDENTIALS_PATH, "w") as f:
        json.dump(template, f, indent=4)
    
    logger.info(f"Created credentials template at {CREDENTIALS_PATH}")

def print_setup_instructions():
    """Print instructions for setting up OAuth credentials."""
    print("\n" + "="*80)
    print("Google Drive API Authentication Setup Instructions".center(80))
    print("="*80)
    print("""
1. Go to the Google Cloud Console: https://console.cloud.google.com/
2. Create a new project or select an existing one
3. Enable the Google Drive API:
   - Navigate to APIs & Services > Library
   - Search for 'Google Drive API' and enable it
4. Create OAuth credentials:
   - Go to APIs & Services > Credentials
   - Click 'Create Credentials' > 'OAuth client ID'
   - Choose 'Desktop application' as the application type
   - Give it a name like 'Gen-Text-AI Drive Client'
   - Click 'Create'
5. Download the JSON file and save it as 'credentials.json' in the project root directory
   - The file should replace the template that was created

IMPORTANT: For 'redirect_uris' make sure to include:
   - http://localhost:8080
   - urn:ietf:wg:oauth:2.0:oob

This is essential for both regular and headless authentication to work properly.
""")
    print("="*80)
    print("\n")

def check_credentials_format():
    """Check if the credentials.json file has been properly updated."""
    try:
        with open(CREDENTIALS_PATH, "r") as f:
            creds = json.load(f)
            
        if "_comment" in creds:
            logger.warning("The credentials.json file appears to still be a template.")
            return False
            
        installed = creds.get("installed", {})
        web = creds.get("web", {})
        
        # Check for required fields in either installed or web app credentials
        client_id = installed.get("client_id") or web.get("client_id")
        client_secret = installed.get("client_secret") or web.get("client_secret")
        
        if not client_id or client_id.startswith("YOUR_CLIENT_ID"):
            logger.warning("Client ID not properly configured in credentials.json")
            return False
            
        if not client_secret or client_secret == "YOUR_CLIENT_SECRET":
            logger.warning("Client secret not properly configured in credentials.json")
            return False
            
        # Check redirect URIs
        redirect_uris = installed.get("redirect_uris", []) or web.get("redirect_uris", [])
        localhost_redirect = any("localhost" in uri for uri in redirect_uris)
        oob_redirect = any("oob" in uri for uri in redirect_uris)
        
        if not localhost_redirect:
            logger.warning("Missing localhost redirect URI in credentials.json")
            print("Please add 'http://localhost:8080' to the redirect_uris list in your credentials")
            return False
            
        if not oob_redirect and not any("urn:ietf" in uri for uri in redirect_uris):
            logger.warning("Missing OOB redirect URI in credentials.json")
            print("Please add 'urn:ietf:wg:oauth:2.0:oob' to the redirect_uris list in your credentials")
            print("This is required for headless environments like Paperspace notebooks.")
            return False
            
        return True
        
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logger.error(f"Error checking credentials: {e}")
        return False

def get_credentials(headless: bool = False, token_path: str = None) -> Optional[Credentials]:
    """
    Get Google Drive API credentials with flexible token storage location.
    
    Args:
        headless: Whether to use headless authentication
        token_path: Custom path for the token file
        
    Returns:
        Credentials object or None if unavailable
    """
    if not GOOGLE_API_AVAILABLE:
        logger.error("Google API libraries not available")
        return None
        
    # Use custom token path if provided, otherwise use default
    token_path = token_path or TOKEN_PATH
    
    creds = None
    
    # Check if token exists and is valid
    if os.path.exists(token_path):
        try:
            # Convert the token_path to string before checking endswith
            token_path_str = str(token_path)
            if token_path_str.endswith('.pickle'):
                # Handle pickle format token files (older format)
                import pickle
                with open(token_path, 'rb') as token:
                    creds = pickle.load(token)
            else:
                # Handle JSON format token files
                with open(token_path, 'r') as token:
                    creds = Credentials.from_authorized_user_info(json.loads(token.read()), SCOPES)
        except Exception as e:
            logger.warning(f"Error loading token: {e}")
    
    # If token is invalid or missing, authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                logger.info("Refreshed expired token")
            except Exception as e:
                logger.warning(f"Error refreshing token: {e}")
                creds = None
                
        # If still no valid credentials, need to authenticate
        if not creds:
            if not os.path.exists(CREDENTIALS_PATH):
                logger.error(f"Credentials file not found at {CREDENTIALS_PATH}")
                return None
                
            try:
                # Create flow without specifying redirect_uri in constructor
                flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_PATH), SCOPES)
                
                if headless:
                    # Use console-based auth flow for headless environments like Paperspace
                    logger.info("Using headless authentication flow")
                    
                    # Generate authorization URL with explicit redirect_uri for OOB flow
                    auth_url, _ = flow.authorization_url(
                        access_type='offline',
                        include_granted_scopes='true',
                        redirect_uri='urn:ietf:wg:oauth:2.0:oob')  # Explicitly set the redirect_uri for headless
                    
                    print("\n" + "="*80)
                    print("Headless Authentication Required".center(80))
                    print("="*80)
                    print("\nPlease follow these steps:")
                    print("1. Open this URL in any browser (on your local machine):")
                    print(f"\n{auth_url}\n")
                    print("2. Log in with your Google account and authorize the application")
                    print("3. Copy the authorization code from the browser")
                    auth_code = input("\nEnter the authorization code: ")
                    
                    # Exchange code for credentials with the same redirect_uri
                    flow.fetch_token(code=auth_code, redirect_uri='urn:ietf:wg:oauth:2.0:oob')
                    creds = flow.credentials
                else:
                    # Use browser-based auth for environments with a browser
                    try:
                        logger.info("Launching web browser for authentication...")
                        creds = flow.run_local_server(port=8080)
                    except Exception as e:
                        # Fall back to headless authentication if browser launch fails
                        logger.warning(f"Browser authentication failed: {e}. Falling back to headless mode.")
                        
                        # Generate authorization URL
                        auth_url, _ = flow.authorization_url(
                            access_type='offline',
                            include_granted_scopes='true',
                            redirect_uri='urn:ietf:wg:oauth:2.0:oob')  # Explicitly set redirect_uri
                        
                        print("\nPlease visit this URL in your browser:")
                        print(f"\n{auth_url}\n")
                        auth_code = input("Enter the authorization code: ")
                        
                        # Exchange code for credentials
                        flow.fetch_token(code=auth_code, redirect_uri='urn:ietf:wg:oauth:2.0:oob')  # Use same redirect_uri
                        creds = flow.credentials
                
                # Save the credentials for future use
                token_dir = os.path.dirname(token_path)
                if token_dir:
                    os.makedirs(token_dir, exist_ok=True)
                    
                if token_path.endswith('.pickle'):
                    # Save in pickle format if that's what was requested
                    import pickle
                    with open(token_path, 'wb') as token:
                        pickle.dump(creds, token)
                else:
                    # Save in JSON format (default)
                    with open(token_path, 'w') as token:
                        token.write(creds.to_json())
                        
                logger.info(f"Authentication successful. Token saved to {token_path}")
            except Exception as e:
                logger.error(f"Error during authentication: {e}")
                return None
    
    return creds

def get_drive_service(headless: bool = False, token_path: str = None) -> Any:
    """
    Build and return the Drive API service object.
    
    Args:
        headless: Whether to use headless authentication
        token_path: Custom path for the token file
        
    Returns:
        Drive API service object or None if unavailable
    """
    global _drive_service
    
    if _drive_service:
        return _drive_service
        
    if not GOOGLE_API_AVAILABLE:
        logger.error("Google API libraries not available")
        return None
    
    creds = get_credentials(headless=headless, token_path=token_path)
    if not creds:
        return None
        
    try:
        _drive_service = build('drive', 'v3', credentials=creds)
        return _drive_service
    except Exception as e:
        logger.error(f"Error building Drive service: {e}")
        return None

def is_drive_mounted(headless: bool = False, token_path: str = None) -> bool:
    """
    Check if Google Drive API is authenticated and ready to use.
    
    Args:
        headless: Whether to use headless authentication
        token_path: Custom path for the token file
        
    Returns:
        True if Drive is accessible, False otherwise
    """
    service = get_drive_service(headless=headless, token_path=token_path)
    if not service:
        return False
        
    try:
        # Try a simple request to check if API is working
        service.files().list(pageSize=1).execute()
        return True
    except Exception as e:
        logger.error(f"Drive API check failed: {e}")
        return False

def find_or_create_folder(folder_name: str, parent_id: Optional[str] = None, headless: bool = False) -> Optional[str]:
    """
    Find or create a folder in Google Drive.
    
    Args:
        folder_name: Name of the folder to find or create
        parent_id: ID of the parent folder (None for root)
        headless: Whether to use headless authentication
        
    Returns:
        Folder ID or None if operation failed
    """
    service = get_drive_service(headless=headless)
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

def setup_drive_directories(base_dir: str, headless: bool = False) -> Dict[str, str]:
    """
    Set up a directory structure in Google Drive.
    
    Args:
        base_dir: Base directory name to create
        headless: Whether to use headless authentication
        
    Returns:
        Dictionary mapping directory names to their Google Drive IDs
    """
    if not is_drive_mounted(headless=headless):
        logger.error("Google Drive is not accessible. Cannot set up directories.")
        return {}
    
    # Split base_dir into components and create folder hierarchy
    base_name = os.path.basename(base_dir)
    dir_ids = {}
    
    # Create base directory
    base_id = find_or_create_folder(base_name, headless=headless)
    if not base_id:
        return {}
    
    dir_ids['base'] = base_id
    
    # Create subdirectories
    subdirs = ['models', 'data', 'logs', 'checkpoints', 'results', 'cache']
    for subdir in subdirs:
        subdir_id = find_or_create_folder(subdir, base_id, headless=headless)
        if subdir_id:
            dir_ids[subdir] = subdir_id
    
    # Create data subdirectories
    if 'data' in dir_ids:
        data_subdirs = ['raw', 'processed']
        for data_subdir in data_subdirs:
            data_subdir_id = find_or_create_folder(data_subdir, dir_ids['data'], headless=headless)
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

def save_model_to_drive(model_path: str, drive_dir: str, model_name: Optional[str] = None, headless: bool = False) -> bool:
    """
    Save a model directory or file to Google Drive.
    
    Args:
        model_path: Local path to the model directory or file
        drive_dir: Google Drive folder ID or logical path
        model_name: Optional name to use in Drive (defaults to directory/file name)
        headless: Whether to use headless authentication
        
    Returns:
        True if successful, False otherwise
    """
    if not is_drive_mounted(headless=headless):
        logger.error("Google Drive is not accessible.")
        return False
    
    if not os.path.exists(model_path):
        logger.error(f"Model path {model_path} does not exist.")
        return False
    
    service = get_drive_service(headless=headless)
    if not service:
        return False
    
    # Get the folder ID if a path was provided
    folder_id = drive_dir
    if not folder_id.startswith("1") and "/" in folder_id:
        # This is a path, not an ID - find or create the folder
        folder_name = os.path.basename(drive_dir)
        folder_id = find_or_create_folder(folder_name, headless=headless)
        if not folder_id:
            return False
    
    # Default model name to directory/file name
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
                        parent_id = find_or_create_folder(parent, parent_id, headless=headless)
                        if not parent_id:
                            success = False
                            break
                
                if not parent_id:
                    continue
                
                # Upload file
                try:
                    # Check if file already exists and delete it
                    query = f"name='{os.path.basename(filename)}' and '{parent_id}' in parents and trashed=false"
                    response = service.files().list(q=query, fields='files(id)').execute()
                    for file in response.get('files', []):
                        service.files().delete(fileId=file['id']).execute()
                        
                    # Upload new file
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
            # Check if file already exists and delete it
            query = f"name='{model_name}' and '{folder_id}' in parents and trashed=false"
            response = service.files().list(q=query, fields='files(id)').execute()
            for file in response.get('files', []):
                service.files().delete(fileId=file['id']).execute()
                
            # Upload new file
            file_metadata = {'name': model_name, 'parents': [folder_id]}
            media = MediaFileUpload(model_path)
            service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            logger.info(f"Uploaded {model_path} to Drive")
            return True
        except Exception as e:
            logger.error(f"Error uploading {model_path}: {e}")
            return False

def download_file_from_drive(file_id: str, local_path: str, headless: bool = False) -> bool:
    """
    Download a file from Google Drive.
    
    Args:
        file_id: Google Drive file ID
        local_path: Local path to save the file
        headless: Whether to use headless authentication
        
    Returns:
        True if successful, False otherwise
    """
    service = get_drive_service(headless=headless)
    if not service:
        return False
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
        
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

def find_files_by_name(name: str, folder_id: Optional[str] = None, headless: bool = False) -> List[Dict[str, str]]:
    """
    Find files in Google Drive by name.
    
    Args:
        name: Name of the file(s) to find
        folder_id: Optional folder ID to search in
        headless: Whether to use headless authentication
        
    Returns:
        List of file metadata dictionaries
    """
    service = get_drive_service(headless=headless)
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

def test_authentication(headless: bool = False):
    """Test authentication with Google Drive."""
    logger.info("Testing authentication...")
    
    creds = get_credentials(headless=headless)
    if creds and creds.valid:
        logger.info("✓ Authentication successful!")
        return True
    else:
        logger.error("✗ Authentication failed")
        return False

def test_drive_mounting(headless: bool = False):
    """Test if drive is accessible."""
    logger.info("Testing if Google Drive is accessible...")
    
    if is_drive_mounted(headless=headless):
        logger.info("✓ Google Drive is accessible!")
        return True
    else:
        logger.error("✗ Google Drive is not accessible")
        return False

def test_directory_creation(headless: bool = False):
    """Test directory creation on Google Drive."""
    logger.info("Testing directory creation...")
    
    test_folder = f"gen-text-ai-test-{os.urandom(4).hex()}"
    folder_id = find_or_create_folder(test_folder, headless=headless)
    
    if folder_id:
        logger.info(f"✓ Successfully created test folder: {test_folder}")
        return True, folder_id, test_folder
    else:
        logger.error("✗ Failed to create test folder")
        return False, None, test_folder

def test_file_upload(folder_id, headless: bool = False):
    """Test file upload to Google Drive."""
    logger.info("Testing file upload...")
    
    if not folder_id:
        logger.error("✗ No folder ID provided for upload test")
        return False
    
    # Create a temporary test file
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.txt') as temp:
        temp.write("This is a test file for Google Drive integration in gen-text-ai.")
        temp_path = temp.name
    
    success = save_model_to_drive(temp_path, folder_id, "test_file.txt", headless=headless)
    
    # Clean up the temporary file
    os.unlink(temp_path)
    
    if success:
        logger.info("✓ Successfully uploaded test file")
        return True
    else:
        logger.error("✗ Failed to upload test file")
        return False

def cleanup_test_files(folder_id, folder_name, headless: bool = False):
    """Clean up test files and folders."""
    if not folder_id:
        return
    
    logger.info(f"Cleaning up test folder '{folder_name}' ({folder_id})...")
    try:
        service = get_drive_service(headless=headless)
        if service:
            service.files().delete(fileId=folder_id).execute()
            logger.info("✓ Test folder deleted successfully")
    except Exception as e:
        logger.error(f"✗ Error deleting test folder: {str(e)}")

def authenticate(headless: bool = False):
    """
    Run the authentication flow.
    
    Args:
        headless: Whether to use headless authentication
    
    Returns:
        True if authentication was successful, False otherwise
    """
    # Check if credentials file exists
    if not CREDENTIALS_PATH.exists():
        print("No credentials.json file found.")
        create_credentials_template()
        print_setup_instructions()
        return False
    
    # Check if the credentials file is properly formatted
    if not check_credentials_format():
        print_setup_instructions()
        proceed = input("Continue anyway with the current credentials? (y/n): ")
        if proceed.lower() != 'y':
            return False
    
    # Try to authenticate
    print("\nAttempting to authenticate with Google Drive API...")
    return test_authentication(headless=headless)

def run_tests(headless: bool = False):
    """
    Run tests to verify Google Drive integration.
    
    Args:
        headless: Whether to use headless authentication
    
    Returns:
        True if all tests pass, False otherwise
    """
    print("\n=== Google Drive Integration Test ===\n")
    
    # Test authentication
    if not test_authentication(headless=headless):
        print("\n❌ Authentication test failed. Please run with --action setup first.")
        return False
    
    # Test drive mounting/access
    if not test_drive_mounting(headless=headless):
        print("\n❌ Drive access test failed. Check your Google Drive setup.")
        return False
    
    # Test directory creation
    dir_success, folder_id, folder_name = test_directory_creation(headless=headless)
    if not dir_success:
        print("\n❌ Directory creation test failed.")
        return False
    
    # Test file upload
    if not test_file_upload(folder_id, headless=headless):
        print("\n❌ File upload test failed.")
        cleanup_test_files(folder_id, folder_name, headless=headless)
        return False
    
    # Clean up
    cleanup_test_files(folder_id, folder_name, headless=headless)
    
    print("\n✅ All Google Drive integration tests passed successfully!")
    print("The Google Drive integration is working properly.\n")
    return True

def main():
    parser = argparse.ArgumentParser(description="Google Drive Manager for gen-text-ai")
    parser.add_argument("--action", choices=["setup", "test", "check", "create_folders", "upload", "download", "list"], 
                      required=True, help="Action to perform")
    parser.add_argument("--headless", action="store_true", help="Use headless authentication (for Paperspace etc.)")
    parser.add_argument("--local_path", help="Local file or directory path (for upload/download)")
    parser.add_argument("--remote_folder", help="Remote folder ID or path (for upload/download)")
    parser.add_argument("--remote_file", help="Remote file ID (for download)")
    parser.add_argument("--base_dir", default="gen-text-ai", help="Base directory name for creating folder structure")
    parser.add_argument("--token_path", help="Custom path for token storage")
    parser.add_argument("--search", help="Search term for listing files")
    
    args = parser.parse_args()
    
    # Set global token path if provided
    global TOKEN_PATH
    if args.token_path:
        TOKEN_PATH = Path(args.token_path)
    
    if not GOOGLE_API_AVAILABLE:
        print("Google API libraries not installed. Please install them with:")
        print("pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
        return 1
    
    # Process the action
    if args.action == "setup":
        print("\nWelcome to the Google Drive API Authentication Setup\n")
        success = authenticate(headless=args.headless)
        if success:
            print("\nSetup complete! You can now use the Google Drive integration.")
        else:
            print("\nAuthentication failed. Please check your credentials and try again.")
        
    elif args.action == "test":
        run_tests(headless=args.headless)
        
    elif args.action == "check":
        if test_authentication(headless=args.headless) and test_drive_mounting(headless=args.headless):
            print("\n✅ Google Drive is properly authenticated and accessible.")
        else:
            print("\n❌ Google Drive integration is not working correctly.")
        
    elif args.action == "create_folders":
        directories = setup_drive_directories(args.base_dir, headless=args.headless)
        if directories:
            print(f"\n✅ Successfully created directory structure in '{args.base_dir}':")
            for name, folder_id in directories.items():
                print(f"  - {name}: {folder_id}")
        else:
            print(f"\n❌ Failed to create directory structure in '{args.base_dir}'.")
            
    elif args.action == "upload":
        if not args.local_path:
            print("Error: --local_path is required for upload action")
            return 1
        if not args.remote_folder:
            print("Error: --remote_folder is required for upload action")
            return 1
            
        if save_model_to_drive(args.local_path, args.remote_folder, headless=args.headless):
            print(f"\n✅ Successfully uploaded {args.local_path} to Drive folder {args.remote_folder}")
        else:
            print(f"\n❌ Failed to upload {args.local_path} to Drive")
            
    elif args.action == "download":
        if not args.local_path:
            print("Error: --local_path is required for download action")
            return 1
        if not args.remote_file:
            print("Error: --remote_file is required for download action")
            return 1
            
        if download_file_from_drive(args.remote_file, args.local_path, headless=args.headless):
            print(f"\n✅ Successfully downloaded file {args.remote_file} to {args.local_path}")
        else:
            print(f"\n❌ Failed to download file {args.remote_file}")
            
    elif args.action == "list":
        search_term = args.search or ""
        files = find_files_by_name(search_term, args.remote_folder, headless=args.headless)
        
        if files:
            print(f"\nFound {len(files)} file(s):")
            for file in files:
                file_type = "📁 Folder" if file.get('mimeType') == 'application/vnd.google-apps.folder' else "📄 File"
                print(f"{file_type}: {file.get('name')} (ID: {file.get('id')})")
        else:
            print(f"\nNo files found matching '{search_term}'")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 