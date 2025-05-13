"""
Dataset Drive Manager

This module manages dataset processing and synchronization with Google Drive.
It checks if processed datasets exist in Drive, downloads them if they do,
or processes them locally and uploads them if they don't.
"""

import os
import sys
import time
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import Google Drive manager
from src.utils.google_drive_manager import (
    drive_manager,
    sync_to_drive,
    sync_from_drive,
    test_authentication,
    DRIVE_FOLDERS
)

# Import dataset processing utilities
try:
    from src.data.process_datasets import process_datasets_from_config
except ImportError:
    logger.warning("Could not import process_datasets_from_config. Dataset processing functionality may be limited.")
    
    def process_datasets_from_config(*args, **kwargs):
        logger.error("Dataset processing function not available.")
        return False

class DatasetDriveManager:
    """
    Manages dataset processing and synchronization with Google Drive.
    """
    
    def __init__(self, 
                 config_path: str, 
                 output_dir: str = "data/processed",
                 drive_folder: str = "preprocessed"):
        """
        Initialize the Dataset Drive Manager.
        
        Args:
            config_path: Path to the dataset configuration file
            output_dir: Directory to save processed datasets locally
            drive_folder: Folder key in Google Drive to store datasets
        """
        self.config_path = config_path
        self.output_dir = output_dir
        self.drive_folder = drive_folder
        self.datasets_config = self._load_config()
        self.available_datasets = []
        self.processing_time = 0
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def _load_config(self) -> Dict:
        """Load dataset configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load dataset config from {self.config_path}: {e}")
            return {}
    
    def get_dataset_list(self) -> List[str]:
        """Get list of datasets from configuration."""
        if not self.datasets_config:
            return []
            
        datasets = []
        if "dataset_weights" in self.datasets_config:
            datasets = list(self.datasets_config["dataset_weights"].keys())
        
        return datasets
    
    def check_drive_datasets(self) -> Dict[str, bool]:
        """
        Check which datasets exist in Google Drive.
        
        Returns:
            Dictionary mapping dataset names to existence status
        """
        if not drive_manager.authenticated:
            if not test_authentication():
                logger.warning("Google Drive authentication failed. Cannot check for datasets.")
                return {}
        
        datasets = self.get_dataset_list()
        dataset_status = {}
        
        # First, try to find the preprocessed folder in Drive
        preprocessed_folder_id = drive_manager.folder_ids.get(self.drive_folder)
        if not preprocessed_folder_id:
            # Folder might exist but not be cached in folder_ids
            preprocessed_path = DRIVE_FOLDERS.get(self.drive_folder, self.drive_folder)
            parts = preprocessed_path.split('/')
            
            current_id = drive_manager.folder_ids.get('base')
            for part in parts:
                folder_id = drive_manager.find_file_id(part, current_id)
                if not folder_id:
                    logger.info(f"Drive folder '{part}' not found.")
                    return {dataset: False for dataset in datasets}
                current_id = folder_id
            
            preprocessed_folder_id = current_id
        
        # Check for each dataset
        for dataset in datasets:
            dataset_folder_name = f"{dataset}_processed"
            dataset_folder_id = drive_manager.find_file_id(dataset_folder_name, preprocessed_folder_id)
            dataset_status[dataset] = dataset_folder_id is not None
            
            if dataset_status[dataset]:
                logger.info(f"Dataset '{dataset}' found in Drive.")
            else:
                logger.info(f"Dataset '{dataset}' not found in Drive.")
        
        return dataset_status
    
    def download_from_drive(self, datasets: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Download processed datasets from Google Drive.
        
        Args:
            datasets: List of dataset names to download, or None for all
            
        Returns:
            Dictionary mapping dataset names to download success status
        """
        if not drive_manager.authenticated:
            if not test_authentication():
                logger.warning("Google Drive authentication failed. Cannot download datasets.")
                return {}
        
        # If no specific datasets provided, download all available ones
        if datasets is None:
            available_datasets = self.check_drive_datasets()
            datasets = [ds for ds, exists in available_datasets.items() if exists]
        
        download_status = {}
        
        for dataset in datasets:
            dataset_folder_name = f"{dataset}_processed"
            dataset_local_path = os.path.join(self.output_dir, dataset_folder_name)
            
            logger.info(f"Downloading dataset '{dataset}' from Drive...")
            start_time = time.time()
            
            # Download the dataset from Drive
            success = sync_from_drive(f"{self.drive_folder}/{dataset_folder_name}", dataset_local_path)
            
            download_time = time.time() - start_time
            download_status[dataset] = success
            
            if success:
                logger.info(f"Successfully downloaded dataset '{dataset}' in {download_time:.2f} seconds.")
            else:
                logger.error(f"Failed to download dataset '{dataset}'.")
        
        return download_status
    
    def upload_to_drive(self, datasets: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Upload processed datasets to Google Drive.
        
        Args:
            datasets: List of dataset names to upload, or None for all
            
        Returns:
            Dictionary mapping dataset names to upload success status
        """
        if not drive_manager.authenticated:
            if not test_authentication():
                logger.warning("Google Drive authentication failed. Cannot upload datasets.")
                return {}
        
        # If no specific datasets provided, upload all that exist locally
        if datasets is None:
            datasets = self.get_dataset_list()
        
        upload_status = {}
        
        for dataset in datasets:
            dataset_folder_name = f"{dataset}_processed"
            dataset_local_path = os.path.join(self.output_dir, dataset_folder_name)
            
            # Check if dataset exists locally
            if not os.path.exists(dataset_local_path):
                logger.warning(f"Dataset '{dataset}' not found locally at '{dataset_local_path}'.")
                upload_status[dataset] = False
                continue
            
            logger.info(f"Uploading dataset '{dataset}' to Drive...")
            start_time = time.time()
            
            # Upload the dataset to Drive
            success = sync_to_drive(dataset_local_path, self.drive_folder)
            
            upload_time = time.time() - start_time
            upload_status[dataset] = success
            
            if success:
                logger.info(f"Successfully uploaded dataset '{dataset}' in {upload_time:.2f} seconds.")
            else:
                logger.error(f"Failed to upload dataset '{dataset}'.")
        
        return upload_status
    
    def process_datasets(self, 
                         force_process: bool = False, 
                         upload_to_drive: bool = True) -> Tuple[Dict[str, bool], float]:
        """
        Process datasets according to configuration.
        
        Args:
            force_process: Whether to process datasets even if they exist in Drive
            upload_to_drive: Whether to upload processed datasets to Drive
            
        Returns:
            Tuple containing processing status dict and total processing time
        """
        start_time = time.time()
        datasets = self.get_dataset_list()
        
        if not datasets:
            logger.warning("No datasets found in configuration.")
            return {}, 0
        
        # If not forcing processing, check which datasets exist in Drive
        if not force_process and drive_manager.authenticated:
            drive_datasets = self.check_drive_datasets()
            
            # Download datasets that exist in Drive
            download_datasets = [ds for ds, exists in drive_datasets.items() if exists]
            if download_datasets:
                self.download_from_drive(download_datasets)
                
            # Only process datasets that don't exist in Drive
            datasets_to_process = [ds for ds, exists in drive_datasets.items() if not exists]
        else:
            datasets_to_process = datasets
        
        # Check which datasets already exist locally
        for dataset in list(datasets_to_process):
            dataset_path = os.path.join(self.output_dir, f"{dataset}_processed")
            if os.path.exists(dataset_path):
                logger.info(f"Dataset '{dataset}' already exists locally. Skipping processing.")
                datasets_to_process.remove(dataset)
        
        # If no datasets need processing, return early
        if not datasets_to_process:
            logger.info("All datasets are available. No processing needed.")
            self.processing_time = 0
            return {dataset: True for dataset in datasets}, 0
        
        # Process datasets
        logger.info(f"Processing datasets: {datasets_to_process}")
        process_start_time = time.time()
        
        processing_success = process_datasets_from_config(
            config_path=self.config_path,
            output_dir=self.output_dir,
            datasets=datasets_to_process
        )
        
        process_time = time.time() - process_start_time
        self.processing_time = process_time
        
        logger.info(f"Dataset processing completed in {process_time:.2f} seconds.")
        
        # Upload processed datasets to Drive if requested
        if upload_to_drive and processing_success and datasets_to_process:
            self.upload_to_drive(datasets_to_process)
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Return status for all datasets
        status = {}
        for dataset in datasets:
            dataset_path = os.path.join(self.output_dir, f"{dataset}_processed")
            status[dataset] = os.path.exists(dataset_path)
        
        return status, total_time

    def ensure_datasets_available(self, 
                                  force_process: bool = False,
                                  prefer_drive: bool = True) -> Tuple[bool, float]:
        """
        Ensure all required datasets are available, downloading from Drive or processing as needed.
        
        Args:
            force_process: Whether to force local processing even if datasets exist in Drive
            prefer_drive: Whether to prefer downloading from Drive over local processing
            
        Returns:
            Tuple of (success status, total processing time)
        """
        start_time = time.time()
        
        # Get list of required datasets
        datasets = self.get_dataset_list()
        if not datasets:
            logger.warning("No datasets found in configuration.")
            return False, 0
        
        logger.info(f"Ensuring availability of datasets: {datasets}")
        
        # Check which datasets already exist locally
        missing_locally = []
        for dataset in datasets:
            dataset_path = os.path.join(self.output_dir, f"{dataset}_processed")
            if not os.path.exists(dataset_path):
                missing_locally.append(dataset)
        
        # If all datasets exist locally and not forcing processing, we're done
        if not missing_locally and not force_process:
            logger.info("All required datasets already exist locally.")
            return True, 0
        
        # If forcing local processing, process all datasets
        if force_process:
            status, process_time = self.process_datasets(force_process=True, upload_to_drive=True)
            return all(status.values()), process_time
        
        # If preferring Drive and Drive is available, try to get datasets from there
        if prefer_drive and drive_manager.authenticated:
            # Check which datasets exist in Drive
            drive_datasets = self.check_drive_datasets()
            
            # Download datasets that exist in Drive
            download_datasets = [ds for ds in missing_locally if drive_datasets.get(ds, False)]
            if download_datasets:
                download_status = self.download_from_drive(download_datasets)
                
                # Remove successfully downloaded datasets from the missing list
                for ds, success in download_status.items():
                    if success and ds in missing_locally:
                        missing_locally.remove(ds)
        
        # Process any datasets still missing
        if missing_locally:
            logger.info(f"Processing missing datasets: {missing_locally}")
            process_start_time = time.time()
            
            processing_success = process_datasets_from_config(
                config_path=self.config_path,
                output_dir=self.output_dir,
                datasets=missing_locally
            )
            
            process_time = time.time() - process_start_time
            self.processing_time = process_time
            
            if processing_success:
                logger.info(f"Successfully processed missing datasets in {process_time:.2f} seconds.")
                
                # Upload newly processed datasets to Drive
                if drive_manager.authenticated:
                    self.upload_to_drive(missing_locally)
            else:
                logger.error("Failed to process some datasets.")
                return False, time.time() - start_time
        
        # Verify all datasets are now available locally
        all_available = True
        for dataset in datasets:
            dataset_path = os.path.join(self.output_dir, f"{dataset}_processed")
            if not os.path.exists(dataset_path):
                logger.error(f"Dataset '{dataset}' is still missing after processing attempts.")
                all_available = False
        
        total_time = time.time() - start_time
        return all_available, total_time

# Create a function to easily ensure datasets are available
def ensure_datasets_available(config_path: str, 
                              output_dir: str = "data/processed",
                              drive_folder: str = "preprocessed",
                              force_process: bool = False,
                              prefer_drive: bool = True) -> Tuple[bool, float]:
    """
    Ensure all required datasets are available for training.
    
    Args:
        config_path: Path to the dataset configuration file
        output_dir: Directory to save processed datasets locally
        drive_folder: Folder key in Google Drive to store datasets
        force_process: Whether to force local processing even if datasets exist in Drive
        prefer_drive: Whether to prefer downloading from Drive over local processing
        
    Returns:
        Tuple of (success status, total processing time)
    """
    manager = DatasetDriveManager(config_path, output_dir, drive_folder)
    return manager.ensure_datasets_available(force_process, prefer_drive) 