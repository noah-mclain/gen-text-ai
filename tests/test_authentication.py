#!/usr/bin/env python3
"""
Test Google Drive authentication in headless mode.

This script tests the authentication process for Google Drive in a headless environment,
allowing users to copy an authorization URL, authenticate in a browser, and paste the
resulting code back into the script.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def create_simplified_credentials():
    """Create a simplified credentials file for testing."""
    template_path = os.path.join(project_root, "credentials.json")
    
    # Only create if it doesn't exist or is a template
    if os.path.exists(template_path):
        try:
            with open(template_path, 'r') as f:
                data = json.load(f)
                if '_comment' not in data:
                    logger.info(f"Using existing credentials file: {template_path}")
                    return template_path
        except:
            pass  # If there's an error, we'll create a new file
    
    # Create a template credentials file
    template = {
        "_comment": "This is a template. Replace with your actual OAuth credentials.",
        "installed": {
            "client_id": "YOUR_CLIENT_ID.apps.googleusercontent.com",
            "project_id": "YOUR_PROJECT_ID",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_secret": "YOUR_CLIENT_SECRET",
            "redirect_uris": [
                "http://localhost:8080", 
                "urn:ietf:wg:oauth:2.0:oob"
            ]
        }
    }
    
    with open(template_path, 'w') as f:
        json.dump(template, f, indent=4)
    
    logger.info(f"Created template credentials file at: {template_path}")
    logger.info("You need to replace this with real credentials before authentication will work")
    
    return template_path

def main():
    # Ensure credentials file exists
    create_simplified_credentials()
    
    # Try direct import first
    try:
        # First try the redirect module
        from scripts.src.utils.google_drive_manager import test_authentication, drive_manager
        logger.info("Successfully imported from scripts.src.utils.google_drive_manager")
        
        # Try authentication - this will attempt headless auth if credentials are valid
        print("\n" + "="*80)
        print("TESTING HEADLESS AUTHENTICATION".center(80))
        print("="*80)
        print("""
This script will test the headless authentication flow for Google Drive.
If you have valid credentials.json, you'll be shown an authorization URL
to open in your browser. After authenticating, you'll receive a code to
paste back here.

If you don't have valid credentials yet, you'll be prompted to create them.
""")
        print("="*80 + "\n")
        
        proceed = input("Ready to proceed with testing? (y/n): ").strip().lower()
        if proceed != 'y':
            print("Test cancelled.")
            return
        
        print("\nAttempting authentication...")
        success = test_authentication()
        
        if success:
            print("\n✅ Authentication successful!")
            print("Your token has been saved for future use.")
        else:
            print("\n❌ Authentication failed.")
            print("Check that your credentials.json file contains valid OAuth credentials.")
            print("Make sure it includes 'urn:ietf:wg:oauth:2.0:oob' in the redirect_uris.")
            
    except Exception as e:
        logger.error(f"Error during import or authentication: {str(e)}")
    
    logger.info("Test complete")

if __name__ == "__main__":
    main() 