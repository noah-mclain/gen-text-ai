#!/usr/bin/env python3
"""
Script to extract Hugging Face token from credentials.json and set it as an environment variable.
This is useful for accessing gated datasets that require authentication.
"""

import os
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_hf_token(credentials_path="credentials.json"):
    """Extract Hugging Face token from credentials.json and set as environment variable."""
    try:
        # Check if credentials file exists
        if not os.path.exists(credentials_path):
            logger.error(f"Credentials file not found at {credentials_path}")
            return False
            
        # Load credentials
        with open(credentials_path, 'r') as f:
            credentials = json.load(f)
            
        # Extract HF token if available
        hf_token = None
        
        # Check different possible paths in the JSON structure
        if "huggingface" in credentials:
            hf_token = credentials["huggingface"].get("token")
        elif "hf_token" in credentials:
            hf_token = credentials["hf_token"]
        elif "api_keys" in credentials and "huggingface" in credentials["api_keys"]:
            hf_token = credentials["api_keys"]["huggingface"]
        
        # Set environment variable if token was found
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            logger.info("Successfully set HF_TOKEN environment variable")
            return True
        else:
            logger.error("No Hugging Face token found in credentials file")
            logger.info("Please add your Hugging Face token to credentials.json under huggingface.token or hf_token")
            return False
            
    except Exception as e:
        logger.error(f"Error setting HF_TOKEN: {str(e)}")
        return False
        
def main():
    parser = argparse.ArgumentParser(description="Set Hugging Face token from credentials file")
    parser.add_argument("--credentials", default="credentials.json",
                        help="Path to credentials file (default: credentials.json)")
    
    args = parser.parse_args()
    
    if set_hf_token(args.credentials):
        logger.info("HF_TOKEN has been set successfully")
    else:
        logger.error("Failed to set HF_TOKEN")
        
if __name__ == "__main__":
    main() 