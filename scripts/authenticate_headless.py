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
        token_path = 'token.pickle'
        
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
                    args.credentials, SCOPES)
                
                # Get the authorization URL for manual flow
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