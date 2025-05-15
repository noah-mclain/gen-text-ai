#!/usr/bin/env python3
"""
[DEPRECATED] Google Drive Manager Implementation

This file is DEPRECATED. 

The main implementation is now maintained in:
src/utils/google_drive_manager.py

Please use that file directly for any imports or modifications.
"""

import os
import sys
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Issue a deprecation warning
warning_message = """
DEPRECATION WARNING:
The google_drive_manager_impl.py file is deprecated.
The main implementation is now in src/utils/google_drive_manager.py.
Please update your imports to use the main implementation.
"""

warnings.warn(warning_message, DeprecationWarning, stacklevel=2)
logger.warning(warning_message)

# Redirecting to the main implementation to prevent breakage
try:
    from src.utils.google_drive_manager import *
    logger.info("Successfully redirected to src.utils.google_drive_manager")
except ImportError as e:
    logger.error(f"Failed to import from src.utils.google_drive_manager: {e}")
    logger.error("Please ensure src/utils/google_drive_manager.py exists")
    raise ImportError(f"Failed to import from main implementation: {e}") from e 