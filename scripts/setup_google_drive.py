#!/usr/bin/env python3
"""
Setup Google Drive Authentication

This script helps set up Google Drive authentication in headless environments
like Paperspace. It uses OAuth flow rather than service accounts.

Usage:
    python scripts/setup_google_drive.py
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def print_setup_instructions():
    """Print detailed instructions for setting up Google Cloud OAuth."""
    print("\n" + "="*80)
    print("Google Drive OAuth Setup Instructions".center(80))
    print("="*80)
    print("""
Follow these steps to set up Google Drive access:

1. Go to the Google Cloud Console: https://console.cloud.google.com/
2. Create a new project or select an existing one
3. Enable the Google Drive API:
   - Navigate to APIs & Services > Library
   - Search for 'Google Drive API' and enable it
4. Create OAuth client ID:
   - Go to APIs & Services > Credentials
   - Click 'Create Credentials' > 'OAuth client ID'
   - Select 'Desktop app' as the application type
   - Give it a name like 'Gen-Text-AI Drive Client'
   - Click 'Create'
5. Download the client configuration:
   - Click the download button (JSON) for the created OAuth client ID
   - Rename the downloaded file to 'credentials.json'
   - Move this file to your Paperspace instance

IMPORTANT: The first time you run this tool, you'll need to follow a link
and authenticate with your Google account. This is a one-time process.
""")
    print("="*80)
    print("\n")

def setup_google_drive():
    """Guide the user through Google Drive setup."""
    print("\nWelcome to the Google Drive Setup\n")
    print("This tool will help you set up Google Drive authentication in headless environments.")
    
    # Show setup instructions
    print_setup_instructions()
    
    # Import the drive manager
    try:
        from src.utils.google_drive_manager import test_authentication, test_drive_mounting
    except ImportError as e:
        logger.error(f"Error importing Google Drive manager: {str(e)}")
        logger.info("Make sure the src/utils/google_drive_manager.py file exists and all dependencies are installed.")
        logger.info("You can install dependencies with: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
        return False
    
    # Ask user if they want to proceed
    proceed = input("\nReady to set up Google Drive authentication? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Setup cancelled.")
        return False
    
    # Attempt authentication
    print("\nStarting Google Drive authentication...")
    success = test_authentication()
    
    if not success:
        print("\nAuthentication failed. Please check your credentials file and try again.")
        return False
    
    # Test access
    print("\nTesting Google Drive access...")
    access_success = test_drive_mounting()
    
    if not access_success:
        print("\nDrive access failed. Please make sure you granted appropriate permissions.")
        return False
    
    print("\n✅ Google Drive setup completed successfully!")
    print("You're now ready to use all Drive features with this application.")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Set up Google Drive authentication")
    
    # No additional arguments for now, but could be extended in the future
    
    args = parser.parse_args()
    
    try:
        success = setup_google_drive()
        if success:
            return 0
        else:
            print("\nGoogle Drive setup failed.")
            return 1
    except KeyboardInterrupt:
        print("\nSetup cancelled by user.")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 