#!/usr/bin/env python3
"""
Google Drive Setup Utility

This script guides users through setting up Google Drive integration by:
1. Validating the credentials file
2. Guiding through OAuth authentication
3. Testing Drive access
4. Creating necessary directories

For headless environments (like Paperspace), use the --headless flag.
"""

import os
import sys
import json
import logging
import argparse
import tempfile
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add the project root directory to sys.path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

def check_credentials_file(credentials_path):
    """Check if the credentials file exists and is properly formatted."""
    if not os.path.exists(credentials_path):
        logger.error(f"❌ Credentials file not found at {credentials_path}")
        logger.info("Please download your OAuth credentials from Google Cloud Console.")
        logger.info("1. Go to https://console.cloud.google.com/")
        logger.info("2. Navigate to APIs & Services > Credentials")
        logger.info("3. Create or select an OAuth 2.0 Client ID")
        logger.info("4. Download the JSON file and save it as 'credentials.json'")
        return False

    try:
        with open(credentials_path, 'r') as f:
            creds_data = json.load(f)
        
        # Check required fields
        if 'installed' not in creds_data and 'web' not in creds_data:
            logger.error("❌ Invalid credentials format: missing 'installed' or 'web' section")
            return False
        
        client_section = creds_data.get('installed') or creds_data.get('web', {})
        required_fields = ['client_id', 'client_secret', 'auth_uri', 'token_uri']
        
        for field in required_fields:
            if field not in client_section:
                logger.error(f"❌ Invalid credentials format: missing '{field}'")
                return False
        
        # Check redirect URIs for headless support
        redirect_uris = client_section.get('redirect_uris', [])
        oob_uri = "urn:ietf:wg:oauth:2.0:oob"
        
        if oob_uri not in redirect_uris:
            logger.warning(f"⚠️ Credentials missing OOB redirect URI: {oob_uri}")
            logger.warning("This may cause issues in headless environments")
            
            # Offer to fix the credentials file
            if input("Would you like to add the OOB URI to your credentials? (y/n): ").lower() == 'y':
                if 'installed' in creds_data:
                    creds_data['installed']['redirect_uris'].append(oob_uri)
                elif 'web' in creds_data:
                    creds_data['web']['redirect_uris'].append(oob_uri)
                
                # Save updated credentials
                with open(credentials_path, 'w') as f:
                    json.dump(creds_data, f, indent=2)
                
                logger.info("✅ Updated credentials file with OOB URI")
                return True
        
        logger.info("✅ Credentials file is valid")
        return True
        
    except json.JSONDecodeError:
        logger.error("❌ Invalid credentials file: not a valid JSON file")
        return False
    except Exception as e:
        logger.error(f"❌ Error validating credentials: {str(e)}")
        return False

def setup_drive_integration(credentials_path, headless=False):
    """Set up Google Drive integration using the credentials file."""
    try:
        # Import Drive manager after checking credentials
        from src.utils.google_drive_manager import (
            authenticate_drive, 
            is_authenticated,
            setup_drive_directories
        )
        
        # Attempt to authenticate
        logger.info("🔑 Starting Google Drive authentication...")
        drive = authenticate_drive(credentials_path, headless=headless)
        
        if drive and is_authenticated(drive):
            logger.info("✅ Authentication successful!")
            
            # Create a test file to verify write access
            logger.info("📝 Testing file operations...")
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp:
                tmp_path = tmp.name
                tmp.write("This is a test file created during setup.")
            
            try:
                # Create directory structure
                logger.info("🗂️ Setting up directory structure in Google Drive...")
                directories = setup_drive_directories(drive, base_dir="GenTextAI")
                
                if directories:
                    logger.info("✅ Directory structure created successfully!")
                    logger.info(f"Created directories: {', '.join(directories.keys())}")
                    
                    # Try importing and using the test module
                    try:
                        logger.info("🧪 Running comprehensive tests...")
                        
                        # Keep track of imported modules for rollback if needed
                        imported_modules = set(sys.modules.keys())
                        
                        # Import the test module
                        from tests.test_google_drive import (
                            test_authentication,
                            test_directory_setup,
                            test_file_upload,
                            test_file_download,
                            test_file_delete
                        )
                        
                        # Run basic tests
                        auth_success, test_drive = test_authentication(credentials_path, headless)
                        if auth_success:
                            dir_success, test_dirs = test_directory_setup(test_drive)
                            if dir_success:
                                upload_success, file_id = test_file_upload(test_drive, test_dirs)
                                if upload_success:
                                    download_success = test_file_download(test_drive, file_id)
                                    if download_success:
                                        test_file_delete(test_drive, file_id)
                                        logger.info("✅ All tests completed successfully!")
                    except ImportError:
                        logger.warning("⚠️ Test module not found, skipping comprehensive tests")
                    except Exception as e:
                        logger.warning(f"⚠️ Tests encountered an issue: {str(e)}")
                        
                        # Roll back imported modules to avoid side effects
                        for mod in set(sys.modules.keys()) - imported_modules:
                            if mod in sys.modules:
                                del sys.modules[mod]
                else:
                    logger.warning("⚠️ Failed to create directory structure. Check permissions.")
            finally:
                # Clean up the test file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            
            logger.info("\n===== SETUP COMPLETED SUCCESSFULLY =====")
            logger.info("You can now use Google Drive integration in the training pipeline!")
            logger.info("Example commands:")
            logger.info("  Process datasets: python main_api.py --mode process --use_drive")
            logger.info("  Train model:      python main_api.py --mode train --use_drive")
            logger.info("  Full pipeline:    python main_api.py --mode all --use_drive")
            
            if headless:
                logger.info("\nFor headless environments, always add the --headless flag:")
                logger.info("  python main_api.py --mode all --use_drive --headless")
            
            return True
        else:
            logger.error("❌ Authentication failed.")
            return False
    except ImportError as e:
        logger.error(f"❌ Failed to import Google Drive module: {str(e)}")
        logger.error("Please ensure you've installed the required packages:")
        logger.error("  pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
        return False
    except Exception as e:
        logger.error(f"❌ Error setting up Google Drive integration: {str(e)}")
        return False

def recommend_alternatives():
    """Recommend alternatives if Drive setup fails."""
    logger.info("\n===== ALTERNATIVE OPTIONS =====")
    logger.info("If you're having trouble with Google Drive API, consider these alternatives:")
    
    logger.info("\n1. Using rclone (command-line tool for cloud storage):")
    logger.info("   a. Install rclone: https://rclone.org/install/")
    logger.info("   b. Configure for Google Drive: rclone config")
    logger.info("   c. Use our rclone integration: --use_drive --drive_tool rclone")
    
    logger.info("\n2. Use local storage with manual backup:")
    logger.info("   a. Process and train locally without Drive integration")
    logger.info("   b. Manually download important files after training")
    
    logger.info("\n3. Use HuggingFace Hub for model storage:")
    logger.info("   a. Set HF_TOKEN environment variable: export HF_TOKEN=your_token")
    logger.info("   b. Use --push_to_hub flag during training")
    logger.info("   c. Specify repo with --hub_model_id your-username/model-name")

def main():
    parser = argparse.ArgumentParser(description="Set up Google Drive integration")
    parser.add_argument("--credentials", type=str, default="credentials.json",
                        help="Path to the Google credentials JSON file")
    parser.add_argument("--headless", action="store_true",
                        help="Use headless authentication mode")
    parser.add_argument("--non-interactive", action="store_true",
                        help="Run in non-interactive mode (no prompts)")
    args = parser.parse_args()
    
    logger.info("===== GOOGLE DRIVE INTEGRATION SETUP =====")
    
    # Check if credentials file exists and is valid
    if not check_credentials_file(args.credentials):
        if not args.non_interactive:
            logger.info("\nWould you like to create a credentials file now? (y/n): ")
            if input().lower() == 'y':
                logger.info("Please follow these steps:")
                logger.info("1. Go to https://console.cloud.google.com/")
                logger.info("2. Create a new project or select an existing one")
                logger.info("3. Navigate to APIs & Services > Library")
                logger.info("4. Search for and enable the Google Drive API")
                logger.info("5. Go to APIs & Services > Credentials")
                logger.info("6. Create OAuth client ID (application type: Desktop)")
                logger.info("7. Download the JSON and save as 'credentials.json'")
                logger.info("\nOnce you've done that, run this setup script again.")
            else:
                recommend_alternatives()
        return 1
    
    # Set up Drive integration
    if setup_drive_integration(args.credentials, args.headless):
        return 0
    else:
        if not args.non_interactive:
            recommend_alternatives()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 