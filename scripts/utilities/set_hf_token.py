#!/usr/bin/env python3
"""
[REDIRECT] Hugging Face Token Utility

This is a redirect to the main implementation in:
src/utils/set_hf_token.py

Please use that module directly for imports and modifications.
"""

import os
import sys
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Issue a deprecation warning
warning_message = """
DEPRECATION WARNING:
The scripts/utilities/set_hf_token.py file is a redirect.
The main implementation is now in src/utils/set_hf_token.py.
Please update your imports to use the main implementation.
"""

warnings.warn(warning_message, DeprecationWarning, stacklevel=2)
logger.debug(warning_message)

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(scripts_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    logger.debug(f"Added project root to Python path: {project_root}")

# Import all components from the main implementation
try:
    # Import everything from the main implementation
    from src.utils.set_hf_token import *
    logger.debug("Successfully redirected to src.utils.set_hf_token")
except ImportError as e:
    logger.error(f"Failed to import from src.utils.set_hf_token: {e}")
    logger.error("Please ensure src/utils/set_hf_token.py exists")
    raise ImportError(f"Failed to import from main implementation: {e}") from e 