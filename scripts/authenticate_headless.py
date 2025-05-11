#!/usr/bin/env python3
"""
Helper script for authenticating with Google Drive API in headless environments like Paperspace.
This script will:
1. Generate an authentication URL
2. Ask you to open it in your local browser
3. Request the authorization code to complete the authentication

After completion, a token.pickle file will be created for future use.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

# Add the parent directory to the Python path so we can import src
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Authenticate with Google Drive API in headless environments")
    parser.add_argument("--credentials", type=str, default="credentials.json",
                      help="Path to the credentials.json file from Google Cloud Console")
    parser.add_argument("--token", type=str, default="token.pickle",
                      help="Path to save the authentication token")
    
    args = parser.parse_args()
    
    # Check if credentials file exists
    if not os.path.exists(args.credentials):
        logger.error(f"Credentials file not found at {args.credentials}")
        logger.error("Please download your credentials from Google Cloud Console.")
        return 1
    
    # Import here to avoid issues if Google API libraries aren't installed
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        import pickle
        from urllib.parse import urlparse, parse_qs
    except ImportError as e:
        logger.error(f"Import error: {str(e)}")
        logger.error(f"Current PYTHONPATH: {sys.path}")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error("Make sure you have the Google API libraries installed:")
        logger.error("pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
        return 1
    
    try:
        logger.info("Starting headless authentication process...")
        
        # Define the scopes
        SCOPES = ['https://www.googleapis.com/auth/drive']
        token_path = args.token
        
        # Read the client secrets to get client_id and client_secret
        try:
            with open(args.credentials, 'r') as f:
                client_config = json.load(f)
                
            # Check whether this is a web or installed app credentials file
            if 'installed' in client_config:
                client_type = 'installed'
            elif 'web' in client_config:
                client_type = 'web'
            else:
                logger.error("Credentials file has invalid format - must contain 'web' or 'installed' key")
                return 1
                
            # Ensure redirect_uri is set correctly (use the first one from the credentials)
            redirect_uri = client_config[client_type]['redirect_uris'][0]
            logger.info(f"Using redirect URI: {redirect_uri}")
            
        except Exception as e:
            logger.error(f"Error parsing credentials file: {str(e)}")
            return 1
        
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
                    args.credentials,
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
                    query = parse_qs(urlparse(redirect_url).query)
                    auth_code = query.get('code', [''])[0]
                    
                    if not auth_code:
                        logger.error("Could not extract authorization code from URL")
                        logger.info("Make sure you copied the entire URL after redirection")
                        return 1
                        
                    logger.info("Successfully extracted authorization code")
                    
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
        files = service.files().list(pageSize=5).execute()
        
        logger.info("Authentication successful!")
        logger.info(f"Token saved to {token_path}")
        logger.info("Successfully connected to Google Drive API")
        return 0
        
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main()) 