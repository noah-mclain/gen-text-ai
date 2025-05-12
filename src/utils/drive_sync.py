#!/usr/bin/env python3
"""
Google Drive Sync Utilities

This module provides utilities for syncing data with Google Drive, supporting both
the Google Drive API and rclone approaches. It handles datasets, checkpoints,
logs, and other project artifacts.
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Drive folder structure
DRIVE_FOLDERS = {
    "data": "data",
    "preprocessed": "data/processed",
    "raw": "data/raw",
    "models": "models",
    "checkpoints": "models/checkpoints",
    "logs": "logs",
    "results": "results",
    "train": "results/train",
    "test": "results/test",
    "eval": "results/eval",
    "validation": "results/validation", 
    "visualizations": "visualizations",
    "cache": "cache",
    "text_models": "text_models",
    "text_checkpoints": "text_models/checkpoints",
    "text_logs": "logs/text_generation"
}

class DriveSync:
    """
    Class to handle syncing with Google Drive using either the Drive API or rclone.
    """
    
    def __init__(self, base_dir: str = "DeepseekCoder", use_rclone: bool = True):
        """
        Initialize the DriveSync.
        
        Args:
            base_dir: Base directory name on Google Drive
            use_rclone: Whether to use rclone (True) or Google Drive API (False)
        """
        self.base_dir = base_dir
        self.use_rclone = use_rclone
        self.drive_api = None
        self.folder_ids = None
        
        # Check if rclone is available when use_rclone is True
        if use_rclone:
            try:
                # Check if rclone is installed and configured
                result = subprocess.run(
                    ["rclone", "lsd", "gdrive:"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False
                )
                self.rclone_available = result.returncode == 0
                if not self.rclone_available:
                    logger.warning("rclone not properly configured. Error: " + result.stderr)
                else:
                    logger.info("rclone is properly configured")
                    # Check if base directory exists
                    self._ensure_base_dir_exists()
            except FileNotFoundError:
                logger.warning("rclone command not found. Please install rclone.")
                self.rclone_available = False
        else:
            # Initialize Google Drive API
            try:
                # Import the Drive API utilities
                sys.path.append(str(Path(__file__).parent.parent.parent))
                from src.utils.drive_api_utils import DriveAPI, setup_drive_directories
                
                # Initialize Drive API
                self.drive_api = DriveAPI()
                
                # Authenticate with Google Drive
                if self.drive_api.authenticate(headless=True):
                    logger.info("Successfully authenticated with Google Drive API")
                    
                    # Set up directory structure
                    self.folder_ids = setup_drive_directories(self.drive_api, self.base_dir)
                    if not self.folder_ids:
                        logger.error("Failed to set up Google Drive directory structure")
                else:
                    logger.warning("Failed to authenticate with Google Drive API")
            except ImportError:
                logger.warning("Google Drive API utilities not found. Please run setup.")
    
    def _ensure_base_dir_exists(self):
        """Ensure the base directory exists in Google Drive."""
        if self.use_rclone and self.rclone_available:
            # List all directories in the root
            result = subprocess.run(
                ["rclone", "lsd", "gdrive:"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            # Check if base directory exists
            base_dir_exists = False
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if self.base_dir in line:
                        base_dir_exists = True
                        break
            
            # Create base directory if it doesn't exist
            if not base_dir_exists:
                logger.info(f"Creating base directory '{self.base_dir}' in Google Drive")
                result = subprocess.run(
                    ["rclone", "mkdir", f"gdrive:{self.base_dir}"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False
                )
                if result.returncode != 0:
                    logger.error(f"Failed to create base directory: {result.stderr}")
            
            # Create the folder structure
            for folder_key, folder_path in DRIVE_FOLDERS.items():
                full_path = f"{self.base_dir}/{folder_path}"
                result = subprocess.run(
                    ["rclone", "mkdir", f"gdrive:{full_path}"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False
                )
                if result.returncode != 0:
                    logger.warning(f"Failed to create folder {full_path}: {result.stderr}")
    
    def sync_to_drive(self, local_path: str, drive_folder_key: str, 
                     delete_source: bool = False, update_only: bool = True) -> bool:
        """
        Sync local files/folders to Google Drive.
        
        Args:
            local_path: Path to local file or folder
            drive_folder_key: Key for the destination folder in DRIVE_FOLDERS
            delete_source: Whether to delete local files after sync
            update_only: Whether to only update existing files or sync everything
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(local_path):
            logger.error(f"Local path not found: {local_path}")
            return False
        
        if self.use_rclone and self.rclone_available:
            # Use rclone for syncing
            try:
                # Construct remote path
                folder_path = DRIVE_FOLDERS.get(drive_folder_key, drive_folder_key)
                remote_path = f"gdrive:{self.base_dir}/{folder_path}"
                
                # Ensure remote directory exists
                mkdir_result = subprocess.run(
                    ["rclone", "mkdir", remote_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False
                )
                
                # Prepare rclone command
                cmd = ["rclone"]
                
                if update_only:
                    # Only update existing files, skip new ones
                    cmd.extend(["copy", "--update"])
                else:
                    # Sync everything (create, update, delete)
                    cmd.extend(["sync"])
                
                # Add paths
                cmd.extend([local_path, remote_path])
                
                # Run the command
                logger.info(f"Syncing {local_path} to {remote_path}")
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False
                )
                
                if result.returncode != 0:
                    logger.error(f"Failed to sync to Drive: {result.stderr}")
                    return False
                
                logger.info(f"Successfully synced {local_path} to {remote_path}")
                
                # Delete source files if requested
                if delete_source:
                    if os.path.isdir(local_path):
                        import shutil
                        shutil.rmtree(local_path)
                        logger.info(f"Deleted local directory: {local_path}")
                    else:
                        os.remove(local_path)
                        logger.info(f"Deleted local file: {local_path}")
                
                return True
                
            except Exception as e:
                logger.error(f"Error syncing to Drive with rclone: {str(e)}")
                return False
        elif self.drive_api and self.drive_api.authenticated:
            # Use Google Drive API
            try:
                # Import the Drive API utilities if not already done
                from src.utils.drive_api_utils import save_to_drive
                
                folder_path = DRIVE_FOLDERS.get(drive_folder_key, drive_folder_key)
                
                # Find the folder ID
                folder_id = None
                components = folder_path.split('/')
                parent_id = None
                
                # Navigate through the folder structure to find the right ID
                for i, component in enumerate(components):
                    # For the first component, use the base directory
                    if i == 0:
                        base_id = self.drive_api.find_file_id(self.base_dir)
                        if not base_id:
                            base_id = self.drive_api.create_folder(self.base_dir)
                        parent_id = base_id
                    
                    # Find or create the folder
                    folder_id = self.drive_api.find_file_id(component, parent_id)
                    if not folder_id:
                        folder_id = self.drive_api.create_folder(component, parent_id)
                    
                    parent_id = folder_id
                
                if not folder_id:
                    logger.error(f"Failed to find or create folder {folder_path}")
                    return False
                
                # Upload the file/folder
                result = save_to_drive(self.drive_api, local_path, folder_id)
                
                if not result:
                    logger.error(f"Failed to save {local_path} to Drive")
                    return False
                
                logger.info(f"Successfully saved {local_path} to Drive")
                
                # Delete source files if requested
                if delete_source:
                    if os.path.isdir(local_path):
                        import shutil
                        shutil.rmtree(local_path)
                        logger.info(f"Deleted local directory: {local_path}")
                    else:
                        os.remove(local_path)
                        logger.info(f"Deleted local file: {local_path}")
                
                return True
                
            except Exception as e:
                logger.error(f"Error syncing to Drive with API: {str(e)}")
                return False
        else:
            logger.error("Neither rclone nor Drive API is available")
            return False
    
    def sync_from_drive(self, drive_folder_key: str, local_path: str) -> bool:
        """
        Sync files/folders from Google Drive to local.
        
        Args:
            drive_folder_key: Key for the source folder in DRIVE_FOLDERS
            local_path: Path to save locally
            
        Returns:
            True if successful, False otherwise
        """
        # Create local directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
        
        if self.use_rclone and self.rclone_available:
            # Use rclone for syncing
            try:
                # Construct remote path
                folder_path = DRIVE_FOLDERS.get(drive_folder_key, drive_folder_key)
                remote_path = f"gdrive:{self.base_dir}/{folder_path}"
                
                # Run the sync command
                logger.info(f"Syncing from {remote_path} to {local_path}")
                result = subprocess.run(
                    ["rclone", "copy", remote_path, local_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False
                )
                
                if result.returncode != 0:
                    logger.error(f"Failed to sync from Drive: {result.stderr}")
                    return False
                
                logger.info(f"Successfully synced from {remote_path} to {local_path}")
                return True
                
            except Exception as e:
                logger.error(f"Error syncing from Drive with rclone: {str(e)}")
                return False
        elif self.drive_api and self.drive_api.authenticated:
            # Use Google Drive API
            try:
                # Import the Drive API utilities if not already done
                from src.utils.drive_api_utils import load_from_drive
                
                folder_path = DRIVE_FOLDERS.get(drive_folder_key, drive_folder_key)
                
                # Find the folder ID
                folder_id = None
                components = folder_path.split('/')
                parent_id = None
                
                # Navigate through the folder structure to find the right ID
                for i, component in enumerate(components):
                    # For the first component, use the base directory
                    if i == 0:
                        base_id = self.drive_api.find_file_id(self.base_dir)
                        if not base_id:
                            logger.error(f"Base directory {self.base_dir} not found")
                            return False
                        parent_id = base_id
                    
                    # Find the folder
                    folder_id = self.drive_api.find_file_id(component, parent_id)
                    if not folder_id:
                        logger.error(f"Folder {component} not found in {folder_path}")
                        return False
                    
                    parent_id = folder_id
                
                if not folder_id:
                    logger.error(f"Failed to find folder {folder_path}")
                    return False
                
                # Download the file/folder
                result = load_from_drive(self.drive_api, folder_id, local_path)
                
                if not result:
                    logger.error(f"Failed to load {folder_path} from Drive")
                    return False
                
                logger.info(f"Successfully loaded {folder_path} from Drive to {local_path}")
                return True
                
            except Exception as e:
                logger.error(f"Error syncing from Drive with API: {str(e)}")
                return False
        else:
            logger.error("Neither rclone nor Drive API is available")
            return False

# Create global instance for easy imports
drive_sync = DriveSync(use_rclone=True)

def sync_to_drive(local_path: str, drive_folder_key: str, 
                delete_source: bool = False, update_only: bool = True) -> bool:
    """
    Sync local files/folders to Google Drive.
    
    Args:
        local_path: Path to local file or folder
        drive_folder_key: Key for the destination folder in DRIVE_FOLDERS or direct path
        delete_source: Whether to delete local files after sync
        update_only: Whether to only update existing files or sync everything
        
    Returns:
        True if successful, False otherwise
    """
    return drive_sync.sync_to_drive(local_path, drive_folder_key, delete_source, update_only)

def sync_from_drive(drive_folder_key: str, local_path: str) -> bool:
    """
    Sync files/folders from Google Drive to local.
    
    Args:
        drive_folder_key: Key for the source folder in DRIVE_FOLDERS or direct path
        local_path: Path to save locally
        
    Returns:
        True if successful, False otherwise
    """
    return drive_sync.sync_from_drive(drive_folder_key, local_path)

def configure_sync_method(use_rclone: bool = True, base_dir: str = "DeepseekCoder"):
    """
    Configure the global drive_sync instance.
    
    Args:
        use_rclone: Whether to use rclone (True) or Google Drive API (False)
        base_dir: Base directory name on Google Drive
    """
    global drive_sync
    drive_sync = DriveSync(base_dir=base_dir, use_rclone=use_rclone)
    
    return drive_sync 