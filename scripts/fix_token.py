#!/usr/bin/env python3
"""
Fix Token File Issues

This script fixes issues with the Google Drive token file,
particularly problems related to JSON parsing scope.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_token_file(token_path=None):
    """
    Fix issues with token file by reading and rewriting it.
    
    Args:
        token_path: Path to the token file (defaults to ~/.drive_token.json)
    
    Returns:
        True if successful, False otherwise
    """
    # Use default path if none provided
    if token_path is None:
        token_path = os.path.expanduser('~/.drive_token.json')
    
    logger.info(f"Checking token file: {token_path}")
    
    # Check if token file exists
    if not os.path.exists(token_path):
        logger.error(f"Token file not found: {token_path}")
        return False
    
    try:
        # Read the token file
        with open(token_path, 'r') as f:
            token_data = json.load(f)
        
        logger.info(f"Successfully read token file")
        
        # Write back to ensure proper format
        with open(token_path, 'w') as f:
            json.dump(token_data, f, indent=2)
            
        logger.info(f"Successfully fixed token file: {token_path}")
        return True
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON error in token file: {e}")
        try:
            # Attempt recovery by reading as text and parsing
            with open(token_path, 'r') as f:
                content = f.read().strip()
            
            # Try to fix common issues
            # Remove potential comments
            lines = [line for line in content.split('\n') if not line.strip().startswith('#')]
            cleaned_content = '\n'.join(lines)
            
            # Try to parse
            token_data = json.loads(cleaned_content)
            
            # Write back in correct format
            with open(token_path, 'w') as f:
                json.dump(token_data, f, indent=2)
                
            logger.info(f"Recovery successful for token file: {token_path}")
            return True
        except Exception as recovery_error:
            logger.error(f"Recovery failed: {recovery_error}")
            return False
    
    except Exception as e:
        logger.error(f"Error fixing token file: {e}")
        return False

def main():
    # Check if token path is provided as argument
    token_path = None
    if len(sys.argv) > 1:
        token_path = sys.argv[1]
    
    success = fix_token_file(token_path)
    
    if success:
        print("✅ Token file fixed successfully")
        return 0
    else:
        print("❌ Failed to fix token file")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 