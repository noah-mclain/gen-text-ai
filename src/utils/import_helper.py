"""
Import Helper

Utility functions to help with proper importing across different environments.
This is particularly useful for the Paperspace environment where the Python path
may be different from the local development environment.
"""

import os
import sys
import logging
import importlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Tuple

logger = logging.getLogger(__name__)

def add_project_root_to_path() -> str:
    """
    Add the project root directory to sys.path.
    
    Returns:
        The absolute path to the project root.
    """
    # Get the path to the current file
    current_file = os.path.abspath(__file__)
    
    # Navigate up to the project root (assuming this file is in src/utils)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    
    # Add to sys.path if not already there
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        logger.info(f"Added project root to Python path: {project_root}")
    
    return project_root

def try_multiple_imports(import_attempts: List[Dict[str, str]]) -> Tuple[Any, str]:
    """
    Try importing a module from multiple possible locations.
    
    Args:
        import_attempts: List of dictionaries with 'module' and 'log' keys,
                         specifying import paths to try in order of preference.
                         
    Returns:
        Tuple of (imported module, import path used) or (None, '') if all fail.
    """
    for attempt in import_attempts:
        try:
            module_name = attempt["module"]
            module = importlib.import_module(module_name)
            logger.info(f"Successfully imported {module_name} via {attempt['log']}")
            return module, module_name
        except ImportError as e:
            logger.debug(f"Failed to import {attempt['module']}: {e}")
    
    logger.error(f"All import attempts failed for: {[a['module'] for a in import_attempts]}")
    return None, ''

def setup_module_for_paperspace() -> None:
    """
    Setup the current module to work in both local and Paperspace environments.
    
    This function should be called at the top of any module that might 
    run in the Paperspace environment.
    """
    # Add project root to path
    add_project_root_to_path()
    
    # Create commonly needed directory structure
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

def get_common_import_paths(base_module: str) -> List[Dict[str, str]]:
    """
    Get a list of common import paths to try for the given module.
    
    Args:
        base_module: The base module name without path prefixes
        
    Returns:
        List of import attempts to try with try_multiple_imports()
    """
    return [
        # Direct import (within same package)
        {"module": base_module, "log": "direct import"},
        # From src package
        {"module": f"src.{base_module}", "log": "src. import"},
        # From scripts package 
        {"module": f"scripts.{base_module}", "log": "scripts. import"},
    ]

def fix_pythonpath_for_command(cmd: str) -> str:
    """
    Fix a command to include the proper PYTHONPATH.
    
    Args:
        cmd: The command to run
        
    Returns:
        Modified command with PYTHONPATH set properly
    """
    pythonpath_cmd = f"PYTHONPATH={os.getcwd()}:$PYTHONPATH "
    return pythonpath_cmd + cmd 