#!/usr/bin/env python3
"""
Fix Drive Auth Script
====================

This script fixes the OAuth flow for Google Drive authentication by ensuring
the required redirect URIs are included in the credentials.json file.

Use this to make your existing "jarvis" project credentials work with the gen-text-ai
codebase without creating new credentials.

Usage:
    python scripts/fix_drive_auth.py
"""

import os
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_credentials_file(credentials_path="credentials.json"):
    """
    Fix the credentials.json file by ensuring it has the required redirect URIs.
    """
    # Get the absolute path
    credentials_path = os.path.abspath(credentials_path)
    
    if not os.path.exists(credentials_path):
        logger.error(f"Credentials file not found at {credentials_path}")
        logger.info("Creating a template file instead...")
        
        # Create template credentials.json
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
            
        logger.info(f"Created template credentials file at {credentials_path}")
        logger.info("Please update it with your actual OAuth credentials from Google Cloud Console")
        return False
    
    # Load and modify existing credentials.json file
    try:
        with open(credentials_path, "r") as f:
            credentials = json.load(f)
        
        # Make a backup of the original file
        backup_path = f"{credentials_path}.backup"
        with open(backup_path, "w") as f:
            json.dump(credentials, f, indent=4)
        logger.info(f"Backed up original credentials to {backup_path}")
        
        # Check if it has installed or web credentials
        is_installed = "installed" in credentials
        is_web = "web" in credentials
        
        # Determine which key to use
        key = "installed" if is_installed else "web" if is_web else None
        
        if key is None:
            # Create the installed key if missing
            credentials["installed"] = credentials.get("installed", {})
            key = "installed"
            logger.info("Added 'installed' key to credentials")
        
        # Make sure redirect_uris exists and includes the required values
        redirect_uris = credentials[key].get("redirect_uris", [])
        
        # Add the required redirect URIs if not already present
        required_uris = ["http://localhost:8080", "urn:ietf:wg:oauth:2.0:oob"]
        changed = False
        
        for uri in required_uris:
            if uri not in redirect_uris:
                redirect_uris.append(uri)
                logger.info(f"Added missing redirect URI: {uri}")
                changed = True
        
        # Update the redirect_uris in the credentials
        credentials[key]["redirect_uris"] = redirect_uris
        
        # Save the modified credentials
        with open(credentials_path, "w") as f:
            json.dump(credentials, f, indent=4)
        
        if changed:
            logger.info(f"Updated credentials.json with required redirect URIs")
        else:
            logger.info(f"Credentials.json already had the required redirect URIs")
        
        return True
        
    except Exception as e:
        logger.error(f"Error fixing credentials file: {e}")
        return False

def main():
    print("\n================= Google Drive Auth Fixer =================\n")
    print("This tool will update your credentials.json file to work with")
    print("the gen-text-ai project's authentication system.\n")
    
    # Ask for credentials path
    default_path = "credentials.json"
    path_input = input(f"Enter the path to your credentials.json file [default: {default_path}]: ")
    credentials_path = path_input if path_input.strip() else default_path
    
    success = fix_credentials_file(credentials_path)
    
    if success:
        print("\n✅ Credentials file successfully updated!")
        print("\nYou can now run the Google Drive setup command again:")
        print("    python scripts/google_drive_manager.py --action setup --headless\n")
    else:
        print("\n❌ Failed to update credentials file.")
        print("Please check the logs above for more information.\n")

if __name__ == "__main__":
    main() 