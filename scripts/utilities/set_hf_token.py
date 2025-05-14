#!/usr/bin/env python3
"""
Script to extract Hugging Face token from credentials.json and set it as an environment variable.
This is useful for accessing gated datasets that require authentication.
"""

import os
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_hf_token(credentials_path="credentials.json"):
    """Set the HF_TOKEN environment variable from credentials file."""
    
    # Skip if HF_TOKEN is already set
    if os.environ.get("HF_TOKEN"):
        logger.info("HF_TOKEN is already set in environment variables")
        return True
        
    # Check if credentials file exists
    if not os.path.exists(credentials_path):
        logger.warning(f"Credentials file not found at: {credentials_path}")
        return False
        
    # Load the credentials
    try:
        with open(credentials_path, 'r') as f:
            credentials = json.load(f)
            
        # Look for token in various locations in the JSON structure
        token = None
        
        # Direct token field
        if "hf_token" in credentials:
            token = credentials["hf_token"]
        # Nested under huggingface
        elif "huggingface" in credentials and isinstance(credentials["huggingface"], dict):
            if "token" in credentials["huggingface"]:
                token = credentials["huggingface"]["token"]
            else:
                token = next(iter(credentials["huggingface"].values()), None)
        # Nested under api_keys
        elif "api_keys" in credentials and isinstance(credentials["api_keys"], dict):
            if "huggingface" in credentials["api_keys"]:
                token = credentials["api_keys"]["huggingface"]
            elif "hf" in credentials["api_keys"]:
                token = credentials["api_keys"]["hf"]
        
        # Set the token if found
        if token and isinstance(token, str) and len(token) > 10:
            os.environ["HF_TOKEN"] = token
            logger.info("Successfully set HF_TOKEN from credentials")
            return True
        else:
            logger.warning("Could not find valid HF_TOKEN in credentials file")
            return False
            
    except Exception as e:
        logger.error(f"Error loading credentials: {str(e)}")
        return False

def check_hf_token():
    """Check if HF_TOKEN is properly set."""
    token = os.environ.get("HF_TOKEN")
    
    if not token:
        logger.warning("HF_TOKEN is not set!")
        logger.warning("Some gated datasets might be inaccessible.")
        logger.warning("Set your token using: export HF_TOKEN=your_huggingface_token")
        logger.warning("Or add it to your credentials.json file.")
        return False
        
    if len(token) < 10 or not token.startswith("hf_"):
        logger.warning(f"HF_TOKEN value looks invalid: {token[:5]}...")
        return False
        
    logger.info("HF_TOKEN is set properly! You should be able to access gated datasets.")
    return True

if __name__ == "__main__":
    # Try to find credentials in common locations
    candidate_paths = [
        "credentials.json",
        "../credentials.json",
        "../../credentials.json",
        os.path.expanduser("~/credentials.json"),
        os.path.expanduser("~/.huggingface/token")
    ]
    
    for path in candidate_paths:
        if os.path.exists(path):
            logger.info(f"Found credentials at: {path}")
            if set_hf_token(path):
                break
    
    # Check if token is set properly
    check_hf_token() 