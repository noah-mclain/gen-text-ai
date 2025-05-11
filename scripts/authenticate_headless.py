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
import argparse
import logging

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
    from src.utils.drive_api_utils import initialize_drive_api
    
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