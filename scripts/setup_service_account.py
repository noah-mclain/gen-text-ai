#!/usr/bin/env python3
"""
Service Account Setup for Google Drive

This script helps set up Google Drive authentication using a service account
for headless environments like Paperspace.

Usage:
    python scripts/setup_service_account.py
"""

import os
import sys
import json
import logging
import argparse
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_setup_instructions():
    """Print detailed instructions for setting up a service account."""
    print("\n" + "="*80)
    print("Google Drive Service Account Setup Instructions".center(80))
    print("="*80)
    print("""
Follow these steps to set up a Google Drive service account:

1. Go to the Google Cloud Console: https://console.cloud.google.com/
2. Create a new project or select an existing one
3. Enable the Google Drive API:
   - Navigate to APIs & Services > Library
   - Search for 'Google Drive API' and enable it
4. Create service account credentials:
   - Go to APIs & Services > Credentials
   - Click 'Create Credentials' > 'Service Account'
   - Enter a name like 'Gen-Text-AI Drive Service'
   - Click 'Create and Continue'
   - For role, select 'Project' > 'Editor' (or a custom role with Drive access)
   - Click 'Continue' and then 'Done'
5. Create and download the service account key:
   - Find your new service account in the list and click on it
   - Go to the 'Keys' tab
   - Click 'Add Key' > 'Create new key'
   - Choose JSON format and click 'Create'
   - The key file will download automatically
   - Move this file to your Paperspace instance

IMPORTANT: Share your Google Drive folders with the service account email address!
The service account email is shown in the service account details and in the downloaded JSON file
(look for the "client_email" field).
""")
    print("="*80)
    print("\n")

def validate_service_account_file(file_path):
    """Validate that the file is a proper service account credentials file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Check for required fields
        required_fields = ['type', 'project_id', 'private_key_id', 'private_key', 
                          'client_email', 'client_id', 'auth_uri', 'token_uri']
        
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
                
        # Verify it's a service account
        if data.get('type') != 'service_account':
            return False, "Not a service account credentials file"
            
        # Display service account email
        email = data.get('client_email', 'unknown')
        return True, email
        
    except json.JSONDecodeError:
        return False, "Invalid JSON format"
    except Exception as e:
        return False, f"Error validating file: {str(e)}"

def setup_service_account():
    """Guide the user through service account setup."""
    print("\nWelcome to the Google Drive Service Account Setup\n")
    print("This tool will help you set up Google Drive authentication using a service account.")
    print("This is the recommended way to authenticate in headless environments like Paperspace.")
    
    # Show setup instructions
    print_setup_instructions()
    
    # Ask for the service account credentials file
    print("\nPlease provide the path to your service account credentials JSON file.")
    file_path = input("Path to service account JSON file: ").strip()
    
    if not file_path:
        print("No file path provided. Exiting setup.")
        return False
        
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return False
    
    # Validate the file
    valid, message = validate_service_account_file(file_path)
    if not valid:
        print(f"Error: {message}")
        print("The file doesn't appear to be a valid service account credentials file.")
        return False
    
    # Determine destination path
    dest_dir = os.getcwd()
    dest_path = os.path.join(dest_dir, "service-account.json")
    
    # Make a copy of the file if it's not already in the right place
    try:
        # Convert to absolute paths for comparison
        abs_file_path = os.path.abspath(file_path)
        abs_dest_path = os.path.abspath(dest_path)
        
        if abs_file_path == abs_dest_path:
            print(f"\n✅ Service account credentials are already at the correct location: {dest_path}")
        else:
            shutil.copy2(file_path, dest_path)
            print(f"\n✅ Service account credentials copied to {dest_path}")
    except Exception as e:
        print(f"Error copying file: {str(e)}")
        return False
    
    # Set the environment variable
    print(f"\nTo use the service account, set this environment variable:")
    print(f"export GOOGLE_APPLICATION_CREDENTIALS={dest_path}")
    
    # Add to .bashrc or .zshrc if requested
    add_to_shell = input("\nWould you like to add this to your shell profile? (y/n): ").strip().lower()
    if add_to_shell == 'y':
        shell_profile = None
        
        # Determine shell profile file
        if 'SHELL' in os.environ:
            if 'zsh' in os.environ['SHELL']:
                shell_profile = os.path.expanduser("~/.zshrc")
            elif 'bash' in os.environ['SHELL']:
                shell_profile = os.path.expanduser("~/.bashrc")
        
        if not shell_profile:
            shell_profile = os.path.expanduser("~/.bashrc")  # Default to .bashrc
            
        try:
            with open(shell_profile, 'a') as f:
                f.write(f"\n# Google Drive service account for gen-text-ai\n")
                f.write(f"export GOOGLE_APPLICATION_CREDENTIALS={dest_path}\n")
            print(f"✅ Added environment variable to {shell_profile}")
            print(f"   You need to restart your shell or run 'source {shell_profile}' for this to take effect")
        except Exception as e:
            print(f"Error updating shell profile: {str(e)}")
            print(f"Please manually add the export command to your shell profile")
    
    # Set for current session
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = dest_path
    print("\n✅ Environment variable set for current session")
    
    # Show info about the service account
    print(f"\nService account email: {message}")
    print("IMPORTANT: Remember to share your Google Drive folders with this email address!")
    
    # Test the setup
    test_setup = input("\nWould you like to test the setup now? (y/n): ").strip().lower()
    if test_setup == 'y':
        try:
            # Add the project root to path for imports
            project_root = Path(__file__).parent.parent
            sys.path.append(str(project_root))
            
            # Try to import and use the drive manager
            print("\nTesting Google Drive connection...")
            try:
                from scripts.google_drive_manager import test_authentication, test_drive_mounting
                
                if test_authentication():
                    print("✅ Authentication successful!")
                    if test_drive_mounting():
                        print("✅ Google Drive access successful!")
                        return True
                    else:
                        print("❌ Failed to access Google Drive. Are the folders shared with the service account email?")
                else:
                    print("❌ Authentication failed. Please check the credentials file.")
            except ImportError as e:
                print(f"❌ Error importing Google Drive manager: {str(e)}")
                print("Make sure the scripts/google_drive_manager.py file exists and all dependencies are installed.")
                print("You can install dependencies with: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
                return False
        except Exception as e:
            print(f"Error testing setup: {str(e)}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Set up Google Drive service account authentication")
    
    # No additional arguments for now, but could be extended in the future
    
    args = parser.parse_args()
    
    try:
        success = setup_service_account()
        if success:
            print("\nService account setup completed!")
            print("\nYou should now be able to run your scripts with Google Drive integration.")
            print("If you encounter any issues, make sure the GOOGLE_APPLICATION_CREDENTIALS")
            print("environment variable is set and your folders are shared with the service account email.")
            return 0
        else:
            print("\nService account setup failed.")
            return 1
    except KeyboardInterrupt:
        print("\nSetup cancelled by user.")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 