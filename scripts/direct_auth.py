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
                # Create a flow instance with client secrets
                flow = InstalledAppFlow.from_client_secrets_file(
                    credentials_path, SCOPES)
                
                # Get authorization URL for manual flow
                auth_url = flow.authorization_url()[0]
                
                # Display instructions for manual authorization
                logger.info("=" * 70)
                logger.info("HEADLESS AUTHENTICATION REQUIRED")
                logger.info("1. Go to the following URL in your browser:")
                logger.info(f"\n{auth_url}\n")
                logger.info("2. Log in and grant permissions")
                logger.info("3. Copy the authorization code provided")
                logger.info("=" * 70)
                
                # Get the authorization code from user input
                auth_code = input("Enter the authorization code: ").strip()
                
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