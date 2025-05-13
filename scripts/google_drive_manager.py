#!/usr/bin/env python3
"""
Google Drive Manager Redirect

This file ensures that all imports of google_drive_manager from the scripts directory
are directed to the main implementation in src/utils/google_drive_manager.py.

This avoids duplication of code and ensures consistency across the project.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Define the path to the actual implementation
implementation_path = os.path.join(project_root, 'src', 'utils', 'google_drive_manager.py')

# Verify that the implementation exists
if not os.path.exists(implementation_path):
    logger.error(f"Main implementation not found at {implementation_path}")
    logger.error("Please ensure the src/utils/google_drive_manager.py file exists")
    raise ImportError("google_drive_manager.py implementation not found")

# Import directly from the main module
try:
    # Import everything from the main module to make it available to importers
    from src.utils.google_drive_manager import *
    logger.debug("Successfully imported from src.utils.google_drive_manager")
    
except ImportError as e:
    logger.error(f"Failed to import from src.utils.google_drive_manager: {e}")
    logger.error("Make sure the python path includes the project root directory")
    raise

# Inform users about the redirect
logger.debug(f"google_drive_manager redirected from scripts to {implementation_path}") 