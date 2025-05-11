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
        from src.utils.drive_api_utils import initialize_drive_api
    except ImportError as e:
        logger.error(f"Import error: {str(e)}")
        logger.error(f"Current PYTHONPATH: {sys.path}")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error("Make sure you have the Google API libraries installed:")
        logger.error("pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
        return 1
    
    logger.info("Starting headless authentication process...")
    logger.info("=" * 70)
    logger.info("IMPORTANT: You will need to copy a URL, open it in your local browser,")
    logger.info("and then paste the authorization code back here.")
    logger.info("=" * 70)
    
    # Initialize Drive API with headless mode
    drive_api = initialize_drive_api(args.credentials, headless=True)
    
    if drive_api.authenticated:
        logger.info("Authentication successful!")
        logger.info("The token has been saved to 'token.pickle' and will be reused in future sessions.")
        
        # Test if we can access the Drive API
        try:
            files = drive_api.service.files().list(pageSize=5).execute()
            logger.info("Successfully connected to Google Drive API")
            return 0
        except Exception as e:
            logger.error(f"Authentication successful but encountered an error accessing Drive: {str(e)}")
            return 1
    else:
        logger.error("Authentication failed.")
        return 1

if __name__ == "__main__":
    exit(main()) 