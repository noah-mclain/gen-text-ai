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
import json
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
   - Move this file to your project directory

IMPORTANT: Make sure your credentials.json file includes the following redirect URI:
   urn:ietf:wg:oauth:2.0:oob

This is required for headless authentication in environments like Paperspace.
""")
    print("="*80)
    print("\n")
    
def print_rclone_instructions():
    """Print instructions for setting up rclone for users who prefer it."""
    print("\n" + "="*80)
    print("Alternative: Using rclone for Google Drive Access".center(80))
    print("="*80)
    print("""
If you're having issues with the OAuth authentication, you can use rclone as an alternative:

1. Install rclone if not already installed:
   curl https://rclone.org/install.sh | sudo bash

2. Configure rclone:
   rclone config

3. Follow the interactive setup to create a new Google Drive remote:
   - Choose 'n' for new remote
   - Give it a name like 'gdrive'
   - Select 'Google Drive' as the storage type
   - For client_id and client_secret, you can either:
     a. Leave blank to use rclone's default (slower but works)
     b. Enter your own from the same Google Cloud project you created
   - Choose 'drive' for full access scope
   - Leave service_account_file blank
   - Choose 'n' for auto config on headless systems
   - Copy the authorization URL to your local machine's browser
   - Authenticate and get the verification code
   - Paste the verification code back in the terminal

4. Once configured, you can use rclone directly:
   rclone ls gdrive:

This method is independent of the built-in Google Drive manager but can be more
reliable in some environments.
""")
    print("="*80)
    print("\n")

def create_credentials_template():
    """Create a template credentials.json file with instructions."""
    credentials_path = os.path.join(os.getcwd(), 'credentials.json')
    
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
    
    with open(credentials_path, "w") as f:
        json.dump(template, f, indent=4)
    
    logger.info(f"Created credentials template at {credentials_path}")
    logger.info("Please replace the contents with your actual OAuth credentials")
    logger.info("Make sure to include 'urn:ietf:wg:oauth:2.0:oob' in redirect_uris")
    
    return credentials_path

def check_credentials():
    """Check if credentials file exists and has correct format."""
    try:
        from src.utils.google_drive_manager import CREDENTIALS_PATHS
        
        # Try to find existing credentials file
        for path in CREDENTIALS_PATHS:
            if os.path.exists(path):
                # Check if it's properly formatted
                try:
                    with open(path, 'r') as f:
                        creds = json.load(f)
                    
                    # Check if it has the template comment
                    if "_comment" in creds:
                        logger.warning(f"Found credentials template at {path} but it hasn't been filled in yet")
                        return None
                    
                    # Check for required fields
                    client_type = None
                    if 'installed' in creds:
                        client_type = 'installed'
                    elif 'web' in creds:
                        client_type = 'web'
                    
                    if not client_type:
                        logger.warning(f"Invalid credentials format at {path}")
                        return None
                    
                    # Check for OOB redirect URI
                    redirect_uris = creds.get(client_type, {}).get('redirect_uris', [])
                    has_oob = any('oob' in uri for uri in redirect_uris)
                    
                    if not has_oob:
                        logger.warning(f"Credentials at {path} doesn't include OOB redirect URI")
                        logger.warning("This is required for headless authentication")
                        logger.warning("Please add 'urn:ietf:wg:oauth:2.0:oob' to the redirect_uris list")
                        return None
                    
                    logger.info(f"Found valid credentials at {path}")
                    return path
                    
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Error parsing credentials at {path}: {e}")
                    return None
        
        # No valid credentials found, create template
        logger.info("No credentials.json file found, creating template")
        return create_credentials_template()
    
    except ImportError:
        logger.error("Could not import google_drive_manager module")
        return None

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
    
    # Check for credentials file
    valid_credentials = check_credentials()
    if not valid_credentials:
        print("\nYou need to set up a valid credentials.json file before proceeding.")
        print("Follow the instructions above to create your OAuth credentials.")
        print("Once you've downloaded the credentials file, run this script again.")
        
        # Show rclone alternative
        print("\nAlternatively, you can use rclone for Google Drive access.")
        use_rclone = input("Would you like to see instructions for setting up rclone instead? (y/n): ").strip().lower()
        if use_rclone == 'y':
            print_rclone_instructions()
        
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
        print("\nAuthentication failed. There might be an issue with your credentials.")
        
        # Offer rclone as an alternative
        print("\nWould you like to try the rclone method instead?")
        use_rclone = input("Show rclone setup instructions? (y/n): ").strip().lower()
        if use_rclone == 'y':
            print_rclone_instructions()
        
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
    
    # Add optional arguments
    parser.add_argument("--show-rclone", action="store_true",
                      help="Show instructions for setting up rclone")
    parser.add_argument("--base_dir", type=str, default="DeepseekCoder",
                      help="Base directory name on Google Drive")
    
    args = parser.parse_args()
    
    if args.show_rclone:
        print_rclone_instructions()
        return 0
    
    # Load and set configuration
    try:
        from src.utils.google_drive_manager import _drive_manager as drive_manager, configure_sync_method
        # Set the base directory
        if args.base_dir:
            configure_sync_method(base_dir=args.base_dir)
            logger.info(f"Set Drive base directory to: {args.base_dir}")
    except ImportError:
        # Try alternative import
        try:
            from src.utils.google_drive_manager import drive_manager, configure_sync_method
            # Set the base directory
            if args.base_dir:
                configure_sync_method(base_dir=args.base_dir)
                logger.info(f"Set Drive base directory to: {args.base_dir}")
        except Exception as e:
            logger.error(f"Error configuring Drive base directory: {e}")
    except Exception as e:
        logger.error(f"Error configuring Drive base directory: {e}")
        
    # Run the setup
    success = setup_google_drive()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 