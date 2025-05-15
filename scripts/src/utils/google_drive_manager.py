#!/usr/bin/env python3
"""
Google Drive Manager Redirect

This file redirects all imports from scripts.src.utils.google_drive_manager to 
the main implementation in src.utils.google_drive_manager.

IMPORTANT: The main implementation is maintained in src/utils/google_drive_manager.py.
Do not modify this file directly - any changes should be made to the main implementation.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_src_dir = os.path.dirname(current_dir)
scripts_dir = os.path.dirname(scripts_src_dir)
project_root = os.path.dirname(scripts_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    logger.debug(f"Added project root to Python path: {project_root}")

# Import all components from the main implementation
try:
    # Import everything from the main implementation
    from src.utils.google_drive_manager import *
    
    # Log success with the file location for clarity
    main_impl_path = os.path.join(project_root, 'src', 'utils', 'google_drive_manager.py')
    if os.path.exists(main_impl_path):
        logger.debug(f"Successfully imported from main implementation: {main_impl_path}")
    else:
        logger.debug("Successfully imported from src.utils.google_drive_manager (path not verified)")
except ImportError as e:
    logger.error(f"Failed to import from src.utils.google_drive_manager: {e}")
    logger.error("Please ensure the main implementation exists at src/utils/google_drive_manager.py")
    
    # Re-raise the exception to fail loudly - this ensures developers address the issue
    raise ImportError(f"Cannot import from src.utils.google_drive_manager: {e}") from e 