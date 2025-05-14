"""
Drive Dataset Checker

A simple utility to check if datasets exist on Google Drive and download them if available.
This avoids redundant dataset processing by checking Drive before local processing.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import Google Drive manager
try:
    from src.utils.google_drive_manager import (
        drive_manager,
        sync_from_drive,
        test_authentication
    )
    DRIVE_AVAILABLE = True
except ImportError:
    logger.warning("Google Drive manager import failed. Drive functionality will be disabled.")
    DRIVE_AVAILABLE = False

def get_datasets_from_config(config_path: str) -> List[str]:
    """
    Extract dataset names from a configuration file.
    
    Args:
        config_path: Path to the dataset configuration file
        
    Returns:
        List of dataset names
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check for dataset_weights directly in config (dataset_config_text.json format)
        if "dataset_weights" in config:
            return list(config["dataset_weights"].keys())
        # Check for dataset_weights under dataset key (training_config_text.json format)
        elif "dataset" in config and "dataset_weights" in config["dataset"]:
            return list(config["dataset"]["dataset_weights"].keys())
        else:
            logger.warning(f"No dataset_weights found in {config_path}")
            return []
    except Exception as e:
        logger.error(f"Failed to load dataset config from {config_path}: {e}")
        return []

def check_drive_datasets(dataset_names: List[str], drive_folder: str = "preprocessed") -> Dict[str, bool]:
    """
    Check which datasets exist in Google Drive.
    
    Args:
        dataset_names: List of dataset names to check
        drive_folder: Folder key in Google Drive to check
        
    Returns:
        Dictionary mapping dataset names to existence status
    """
    if not DRIVE_AVAILABLE:
        logger.warning("Google Drive functionality not available")
        return {dataset: False for dataset in dataset_names}
        
    if not drive_manager.authenticated:
        if not test_authentication():
            logger.warning("Google Drive authentication failed. Drive functionality disabled.")
            return {dataset: False for dataset in dataset_names}
    
    # Status for each dataset
    dataset_status = {}
    
    # Find the drive folder
    folder_id = drive_manager.folder_ids.get(drive_folder)
    if not folder_id:
        # Try to find or create the folder
        folder_id = drive_manager.find_file_id(drive_folder)
        if not folder_id:
            logger.info(f"Drive folder '{drive_folder}' not found.")
            return {dataset: False for dataset in dataset_names}
    
    # Check each dataset
    for dataset in dataset_names:
        dataset_folder_name = f"{dataset}_processed"
        dataset_id = drive_manager.find_file_id(dataset_folder_name, folder_id)
        dataset_status[dataset] = dataset_id is not None
        
        if dataset_status[dataset]:
            logger.info(f"Dataset '{dataset}' found on Drive.")
        else:
            logger.info(f"Dataset '{dataset}' not found on Drive.")
    
    return dataset_status

def download_drive_datasets(dataset_names: List[str], 
                           output_dir: str = "data/processed", 
                           drive_folder: str = "preprocessed") -> Dict[str, bool]:
    """
    Download specified datasets from Google Drive.
    
    Args:
        dataset_names: List of dataset names to download
        output_dir: Directory to save datasets locally
        drive_folder: Folder key in Google Drive to download from
        
    Returns:
        Dictionary mapping dataset names to download success status
    """
    if not DRIVE_AVAILABLE:
        logger.warning("Google Drive functionality not available")
        return {dataset: False for dataset in dataset_names}
        
    if not drive_manager.authenticated:
        if not test_authentication():
            logger.warning("Google Drive authentication failed. Drive functionality disabled.")
            return {dataset: False for dataset in dataset_names}
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Download status for each dataset
    download_status = {}
    
    # Download each dataset
    for dataset in dataset_names:
        dataset_folder_name = f"{dataset}_processed"
        dataset_local_path = os.path.join(output_dir, dataset_folder_name)
        
        # Skip if already downloaded
        if os.path.exists(dataset_local_path):
            logger.info(f"Dataset '{dataset}' already exists locally.")
            download_status[dataset] = True
            continue
        
        logger.info(f"Downloading dataset '{dataset}' from Drive...")
        start_time = time.time()
        
        # Download from Drive
        success = sync_from_drive(f"{drive_folder}/{dataset_folder_name}", dataset_local_path)
        
        download_time = time.time() - start_time
        download_status[dataset] = success
        
        if success:
            logger.info(f"Successfully downloaded dataset '{dataset}' in {download_time:.2f} seconds.")
        else:
            logger.error(f"Failed to download dataset '{dataset}'.")
    
    return download_status

def check_local_datasets(dataset_names: List[str], output_dir: str = "data/processed") -> Dict[str, bool]:
    """
    Check which datasets are available locally.
    
    Args:
        dataset_names: List of dataset names to check
        output_dir: Directory containing processed datasets
        
    Returns:
        Dictionary mapping dataset names to existence status
    """
    local_status = {}
    
    for dataset in dataset_names:
        dataset_path = os.path.join(output_dir, f"{dataset}_processed")
        local_status[dataset] = os.path.exists(dataset_path)
    
    return local_status

def prepare_datasets(config_path: str, 
                     output_dir: str = "data/processed",
                     drive_folder: str = "preprocessed", 
                     skip_drive: bool = False) -> Tuple[Set[str], Set[str], float]:
    """
    Prepare datasets for training by checking Drive and downloading if available.
    
    Args:
        config_path: Path to the dataset configuration file
        output_dir: Directory to save datasets locally
        drive_folder: Folder key in Google Drive
        skip_drive: Whether to skip checking Drive and only use local datasets
        
    Returns:
        Tuple of (
            available_datasets: Set of dataset names available locally,
            needed_datasets: Set of dataset names that need to be processed locally,
            download_time: Time spent downloading datasets in seconds
        )
    """
    start_time = time.time()
    
    # Get list of datasets from config
    dataset_names = get_datasets_from_config(config_path)
    if not dataset_names:
        logger.warning("No datasets found in configuration.")
        return set(), set(), 0
    
    logger.info(f"Preparing datasets from config: {config_path}")
    logger.info(f"Datasets to prepare: {dataset_names}")
    
    # Check which datasets are already available locally
    local_status = check_local_datasets(dataset_names, output_dir)
    available_datasets = {ds for ds, exists in local_status.items() if exists}
    needed_datasets = {ds for ds, exists in local_status.items() if not exists}
    
    if not needed_datasets:
        logger.info("All datasets already available locally.")
        return set(dataset_names), set(), 0
    
    logger.info(f"Datasets already available locally: {available_datasets}")
    logger.info(f"Datasets still needed: {needed_datasets}")
    
    # If we're skipping Drive, just return the current status
    if skip_drive or not DRIVE_AVAILABLE:
        if skip_drive:
            logger.info("Skipping Google Drive check as requested.")
        return available_datasets, needed_datasets, 0
    
    # Check if any needed datasets exist on Drive
    drive_status = check_drive_datasets(list(needed_datasets), drive_folder)
    drive_available = {ds for ds, exists in drive_status.items() if exists}
    
    if not drive_available:
        logger.info("No needed datasets found on Drive.")
        return available_datasets, needed_datasets, 0
    
    logger.info(f"Datasets available on Drive: {drive_available}")
    
    # Download datasets from Drive
    download_start = time.time()
    download_status = download_drive_datasets(list(drive_available), output_dir, drive_folder)
    download_time = time.time() - download_start
    
    # Update available and needed datasets based on download results
    successfully_downloaded = {ds for ds, success in download_status.items() if success}
    available_datasets.update(successfully_downloaded)
    needed_datasets = needed_datasets - successfully_downloaded
    
    logger.info(f"Successfully downloaded datasets: {successfully_downloaded}")
    logger.info(f"Datasets now available locally: {available_datasets}")
    logger.info(f"Datasets still needed (require processing): {needed_datasets}")
    
    total_time = time.time() - start_time
    logger.info(f"Dataset preparation completed in {total_time:.2f} seconds.")
    logger.info(f"Download time: {download_time:.2f} seconds.")
    
    return available_datasets, needed_datasets, download_time 