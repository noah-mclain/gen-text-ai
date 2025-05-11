#!/usr/bin/env python3
"""
Standalone script for Google Drive API authentication in headless environments.
This script does not require the src module or any project structure.

Usage:
  python direct_auth.py --credentials credentials.json
"""

import os
import pickle
import logging
import argparse
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define Google API scopes
SCOPES = ['https://www.googleapis.com/auth/drive']

def authenticate_drive_api(credentials_path, token_path='token.pickle'):
    """
    Authenticate with Google Drive API using headless flow.
    
    Args:
        credentials_path: Path to credentials.json file
        token_path: Path to save the token
        
    Returns:
        True if authentication was successful, False otherwise
    """
    try:
        # Import Google API libraries
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
        except ImportError:
            logger.error("Required Google API libraries not found.")
            logger.error("Please run: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
            return False

        # Check if credentials file exists
        if not os.path.exists(credentials_path):
            logger.error(f"Credentials file not found at {credentials_path}")
            return False
            
        # Read the client secrets to get client_id and client_secret
        try:
            with open(credentials_path, 'r') as f:
                client_config = json.load(f)
                
            # Check whether this is a web or installed app credentials file
            if 'installed' in client_config:
                client_type = 'installed'
            elif 'web' in client_config:
                client_type = 'web'
            else:
                logger.error("Credentials file has invalid format - must contain 'web' or 'installed' key")
                return False
                
            client_id = client_config[client_type]['client_id']
            client_secret = client_config[client_type]['client_secret']
            
            # Ensure redirect_uri is set correctly (use the first one from the credentials)
            redirect_uri = client_config[client_type]['redirect_uris'][0]
            logger.info(f"Using redirect URI: {redirect_uri}")
            
        except Exception as e:
            logger.error(f"Error parsing credentials file: {str(e)}")
            return False
            
        creds = None
        # Load credentials from token.pickle if it exists
        if os.path.exists(token_path):
            with open(token_path, 'rb') as token:
                creds = pickle.load(token)
        
        # If no valid credentials available, let the user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                # Create a flow instance with client secrets and specific redirect URI
                flow = InstalledAppFlow.from_client_secrets_file(
                    credentials_path, 
                    scopes=SCOPES,
                    redirect_uri=redirect_uri
                )
                
                # Get authorization URL with explicit redirect_uri
                auth_url, _ = flow.authorization_url(
                    access_type='offline',
                    include_granted_scopes='true',
                    prompt='consent'
                )
                
                # Display instructions for manual authorization
                logger.info("=" * 70)
                logger.info("HEADLESS AUTHENTICATION REQUIRED")
                logger.info("1. Go to the following URL in your browser:")
                logger.info(f"\n{auth_url}\n")
                logger.info("2. Log in and grant permissions")
                logger.info("3. You will be redirected to a page that might show an error (this is expected)")
                logger.info("4. Copy the entire URL from your browser's address bar after redirection")
                logger.info("=" * 70)
                
                # Get the full redirect URL from user input
                redirect_url = input("Enter the full URL after redirection: ").strip()
                
                # Extract the authorization code from the URL
                try:
                    from urllib.parse import urlparse, parse_qs
                    query = parse_qs(urlparse(redirect_url).query)
                    auth_code = query.get('code', [''])[0]
                    
                    if not auth_code:
                        logger.error("Could not extract authorization code from URL")
                        logger.info("Make sure you copied the entire URL after redirection")
                        return False
                        
                    logger.info("Successfully extracted authorization code")
                    
                    # Exchange the authorization code for credentials
                    flow.fetch_token(code=auth_code)
                    creds = flow.credentials
                    
                except Exception as e:
                    logger.error(f"Error extracting authorization code: {str(e)}")
                    logger.error("You might need to manually extract the code parameter from the URL")
                    auth_code = input("Please manually enter just the 'code' parameter from the URL: ").strip()
                    
                    # Exchange the authorization code for credentials
                    flow.fetch_token(code=auth_code)
                    creds = flow.credentials
            
            # Save the credentials for the next run
            with open(token_path, 'wb') as token:
                pickle.dump(creds, token)
        
        # Test the connection by listing a few files
        service = build('drive', 'v3', credentials=creds)
        results = service.files().list(pageSize=5).execute()
        
        logger.info("Successfully authenticated with Google Drive API!")
        logger.info(f"Authentication token saved to {token_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error authenticating with Google Drive API: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Authenticate with Google Drive API in headless environments")
    parser.add_argument("--credentials", type=str, default="credentials.json",
                      help="Path to the credentials.json file from Google Cloud Console")
    parser.add_argument("--token", type=str, default="token.pickle",
                      help="Path to save the authentication token")
    
    args = parser.parse_args()
    
    success = authenticate_drive_api(args.credentials, args.token)
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 