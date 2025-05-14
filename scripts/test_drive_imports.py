#!/usr/bin/env python3
"""
Test Drive Imports

This script tests the import of the Google Drive manager modules
to help diagnose any import issues.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def add_project_root():
    """Add the project root to sys.path."""
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file))
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        logger.info(f"Added project root to Python path: {project_root}")
    
    return project_root

def test_import_paths():
    """Test all possible import paths."""
    import_paths = [
        "scripts.google_drive.google_drive_manager",
        "scripts.google_drive.google_drive_manager_impl",
        "src.utils.google_drive_manager",
        "utils.google_drive_manager",
        "src.utils.drive_api_utils"
    ]
    
    results = {}
    
    for path in import_paths:
        logger.info(f"Testing import: {path}")
        try:
            module = __import__(path, fromlist=["*"])
            logger.info(f"✓ Successfully imported {path}")
            
            # Check for key components
            for component in ["DriveManager", "sync_to_drive", "sync_from_drive"]:
                if hasattr(module, component):
                    logger.info(f"  ✓ Found {component} in {path}")
                else:
                    logger.warning(f"  ✗ {component} not found in {path}")
            
            results[path] = True
        except ImportError as e:
            logger.error(f"✗ Failed to import {path}: {e}")
            results[path] = False
        except Exception as e:
            logger.error(f"✗ Error while importing {path}: {e}")
            results[path] = False
    
    return results

def test_file_loading():
    """Test direct file loading."""
    project_root = add_project_root()
    
    # Find all potential implementation files
    impl_files = [
        os.path.join(project_root, "src", "utils", "google_drive_manager.py"),
        os.path.join(project_root, "scripts", "google_drive", "google_drive_manager.py"),
        os.path.join(project_root, "scripts", "google_drive", "google_drive_manager_impl.py")
    ]
    
    results = {}
    
    for file_path in impl_files:
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            results[file_path] = False
            continue
        
        logger.info(f"Testing direct file loading: {file_path}")
        try:
            import importlib.util
            
            module_name = f"google_drive_manager_{os.path.basename(os.path.dirname(file_path))}"
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            logger.info(f"✓ Successfully loaded file: {file_path}")
            
            # Check for key components
            for component in ["DriveManager", "sync_to_drive", "sync_from_drive"]:
                if hasattr(module, component):
                    logger.info(f"  ✓ Found {component} in {file_path}")
                else:
                    logger.warning(f"  ✗ {component} not found in {file_path}")
            
            results[file_path] = True
        except Exception as e:
            logger.error(f"✗ Error loading file {file_path}: {e}")
            results[file_path] = False
    
    return results

def check_dependencies():
    """Check if Google API dependencies are installed."""
    dependencies = [
        ("google", "google-api-python-client"),
        ("googleapiclient", "google-api-python-client"),
        ("google.auth", "google-auth"),
        ("google_auth_oauthlib", "google-auth-oauthlib"),
        ("google.oauth2", "google-auth")
    ]
    
    missing = []
    
    for module, package in dependencies:
        try:
            __import__(module)
            logger.info(f"✓ Dependency installed: {package}")
        except ImportError:
            logger.error(f"✗ Missing dependency: {package}")
            missing.append(package)
    
    if missing:
        logger.info(f"To install missing dependencies: pip install {' '.join(missing)}")
    
    return missing

def main():
    logger.info("Testing Google Drive imports...")
    
    # Add project root to path
    project_root = add_project_root()
    
    # Check dependencies
    logger.info("\nChecking Google API dependencies...")
    missing_deps = check_dependencies()
    
    # Test import paths
    logger.info("\nTesting import paths...")
    import_results = test_import_paths()
    
    # Test direct file loading
    logger.info("\nTesting direct file loading...")
    file_results = test_file_loading()
    
    # Check sys.path
    logger.info("\nCurrent sys.path:")
    for i, path in enumerate(sys.path):
        logger.info(f"  {i}: {path}")
    
    # Summary
    logger.info("\n=== SUMMARY ===")
    
    if missing_deps:
        logger.warning(f"Missing dependencies: {', '.join(missing_deps)}")
    else:
        logger.info("All Google API dependencies are installed")
    
    successful_imports = [p for p, success in import_results.items() if success]
    failed_imports = [p for p, success in import_results.items() if not success]
    
    if successful_imports:
        logger.info(f"Successful imports: {', '.join(successful_imports)}")
    
    if failed_imports:
        logger.warning(f"Failed imports: {', '.join(failed_imports)}")
    
    successful_files = [os.path.basename(p) for p, success in file_results.items() if success]
    failed_files = [os.path.basename(p) for p, success in file_results.items() if not success]
    
    if successful_files:
        logger.info(f"Successfully loaded files: {', '.join(successful_files)}")
    
    if failed_files:
        logger.warning(f"Failed to load files: {', '.join(failed_files)}")
    
    # Final assessment
    if successful_imports or successful_files:
        logger.info("\n✓ Google Drive functionality is available through at least one import path")
        logger.info(f"Recommended import: {successful_imports[0] if successful_imports else 'Direct file loading'}")
    else:
        logger.error("\n✗ Google Drive functionality is NOT available - all import paths failed")
        if missing_deps:
            logger.info("Fix by installing missing dependencies")

if __name__ == "__main__":
    main() 