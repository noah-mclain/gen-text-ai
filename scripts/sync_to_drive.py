#!/usr/bin/env python3
"""
Sync to Drive Utility

Sync datasets, models, checkpoints, logs, and results to Google Drive.
Useful for backing up training runs or moving data between environments.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Try to import drive_sync
try:
    from src.utils.drive_sync import (
        sync_to_drive, 
        sync_from_drive, 
        configure_sync_method,
        DRIVE_FOLDERS
    )
except ImportError:
    logger.error("Failed to import drive_sync module. Make sure src/utils/drive_sync.py exists.")
    sys.exit(1)

def setup_sync(base_dir: str = "DeepseekCoder", use_rclone: bool = True):
    """
    Configure drive sync with the specified settings.
    
    Args:
        base_dir: Base directory on Google Drive
        use_rclone: Whether to use rclone (True) or Google Drive API (False)
    """
    try:
        # Setup drive sync
        configure_sync_method(use_rclone=use_rclone, base_dir=base_dir)
        logger.info(f"Configured drive sync to use {'rclone' if use_rclone else 'Google Drive API'}")
        logger.info(f"Using base directory: {base_dir}")
        return True
    except Exception as e:
        logger.error(f"Error setting up drive sync: {e}")
        return False

def sync_datasets_to_drive(delete_local: bool = False):
    """
    Sync all datasets to Google Drive.
    
    Args:
        delete_local: Whether to delete local datasets after syncing
    """
    datasets_dir = "data/processed"
    if not os.path.exists(datasets_dir):
        logger.warning(f"Datasets directory '{datasets_dir}' not found")
        return False
    
    logger.info("Syncing datasets to Drive...")
    try:
        # List all dataset directories
        dataset_dirs = [os.path.join(datasets_dir, d) for d in os.listdir(datasets_dir) 
                      if os.path.isdir(os.path.join(datasets_dir, d))]
        
        for dataset_dir in dataset_dirs:
            dataset_name = os.path.basename(dataset_dir)
            logger.info(f"Syncing dataset: {dataset_name}")
            
            success = sync_to_drive(
                dataset_dir,
                "preprocessed",
                delete_source=delete_local,
                update_only=False
            )
            
            if success:
                logger.info(f"Successfully synced dataset: {dataset_name}")
            else:
                logger.error(f"Failed to sync dataset: {dataset_name}")
        
        # Sync dataset metadata if any exist
        metadata_files = [os.path.join(datasets_dir, f) for f in os.listdir(datasets_dir)
                         if os.path.isfile(os.path.join(datasets_dir, f))]
        
        for metadata_file in metadata_files:
            file_name = os.path.basename(metadata_file)
            logger.info(f"Syncing metadata file: {file_name}")
            
            success = sync_to_drive(
                metadata_file,
                "preprocessed",
                delete_source=delete_local,
                update_only=False
            )
            
            if success:
                logger.info(f"Successfully synced metadata file: {file_name}")
            else:
                logger.error(f"Failed to sync metadata file: {file_name}")
        
        return True
    except Exception as e:
        logger.error(f"Error syncing datasets: {e}")
        return False

def sync_models_to_drive(model_dir: str, is_text_model: bool = False, delete_local: bool = False):
    """
    Sync model files to Google Drive.
    
    Args:
        model_dir: Directory containing model files
        is_text_model: Whether this is a text generation model
        delete_local: Whether to delete local files after syncing
    """
    if not os.path.exists(model_dir):
        logger.warning(f"Model directory '{model_dir}' not found")
        return False
    
    # Determine the target folder in Drive
    target_folder = "text_models" if is_text_model else "models"
    
    logger.info(f"Syncing model files from {model_dir} to Drive...")
    try:
        success = sync_to_drive(
            model_dir,
            target_folder,
            delete_source=delete_local,
            update_only=False
        )
        
        if success:
            logger.info(f"Successfully synced model files from {model_dir}")
        else:
            logger.error(f"Failed to sync model files from {model_dir}")
        
        return success
    except Exception as e:
        logger.error(f"Error syncing model files: {e}")
        return False

def sync_logs_to_drive(logs_dir: str = "logs", is_text_model: bool = False, delete_local: bool = False):
    """
    Sync log files to Google Drive.
    
    Args:
        logs_dir: Directory containing log files
        is_text_model: Whether these are logs for text generation
        delete_local: Whether to delete local files after syncing
    """
    if not os.path.exists(logs_dir):
        logger.warning(f"Logs directory '{logs_dir}' not found")
        return False
    
    # Determine the target folder in Drive
    target_folder = "text_logs" if is_text_model else "logs"
    
    logger.info(f"Syncing log files from {logs_dir} to Drive...")
    try:
        success = sync_to_drive(
            logs_dir,
            target_folder,
            delete_source=delete_local,
            update_only=False
        )
        
        if success:
            logger.info(f"Successfully synced log files from {logs_dir}")
        else:
            logger.error(f"Failed to sync log files from {logs_dir}")
        
        return success
    except Exception as e:
        logger.error(f"Error syncing log files: {e}")
        return False

def sync_results_to_drive(results_dir: str = "results", delete_local: bool = False):
    """
    Sync result files to Google Drive.
    
    Args:
        results_dir: Directory containing result files
        delete_local: Whether to delete local files after syncing
    """
    if not os.path.exists(results_dir):
        logger.warning(f"Results directory '{results_dir}' not found")
        return False
    
    logger.info(f"Syncing result files from {results_dir} to Drive...")
    try:
        success = sync_to_drive(
            results_dir,
            "results",
            delete_source=delete_local,
            update_only=False
        )
        
        if success:
            logger.info(f"Successfully synced result files from {results_dir}")
        else:
            logger.error(f"Failed to sync result files from {results_dir}")
        
        return success
    except Exception as e:
        logger.error(f"Error syncing result files: {e}")
        return False

def sync_checkpoints_to_drive(checkpoints_dir: str, model_name: str, is_text_model: bool = False, delete_local: bool = False):
    """
    Sync checkpoint files to Google Drive.
    
    Args:
        checkpoints_dir: Directory containing checkpoint files
        model_name: Name of the model (used for organizing in Drive)
        is_text_model: Whether these are checkpoints for text generation model
        delete_local: Whether to delete local files after syncing
    """
    if not os.path.exists(checkpoints_dir):
        logger.warning(f"Checkpoints directory '{checkpoints_dir}' not found")
        return False
    
    # Determine the target folder in Drive
    target_folder = "text_checkpoints" if is_text_model else "checkpoints"
    remote_path = f"{target_folder}/{model_name}"
    
    logger.info(f"Syncing checkpoint files from {checkpoints_dir} to Drive...")
    try:
        success = sync_to_drive(
            checkpoints_dir,
            remote_path,
            delete_source=delete_local,
            update_only=False
        )
        
        if success:
            logger.info(f"Successfully synced checkpoint files from {checkpoints_dir}")
        else:
            logger.error(f"Failed to sync checkpoint files from {checkpoints_dir}")
        
        return success
    except Exception as e:
        logger.error(f"Error syncing checkpoint files: {e}")
        return False

def download_from_drive(remote_folder: str, local_path: str):
    """
    Download files from Google Drive.
    
    Args:
        remote_folder: Folder key or path on Google Drive
        local_path: Path to save files locally
    """
    logger.info(f"Downloading files from {remote_folder} to {local_path}...")
    try:
        # Ensure local directory exists
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
        
        success = sync_from_drive(remote_folder, local_path)
        
        if success:
            logger.info(f"Successfully downloaded files from {remote_folder}")
        else:
            logger.error(f"Failed to download files from {remote_folder}")
        
        return success
    except Exception as e:
        logger.error(f"Error downloading files: {e}")
        return False

def list_drive_folders():
    """List available folder keys in Drive."""
    logger.info("Available folder keys in Drive:")
    for key, path in DRIVE_FOLDERS.items():
        logger.info(f"- {key}: {path}")

def main():
    parser = argparse.ArgumentParser(description="Sync datasets, models, and results to/from Google Drive")
    
    # Setup options
    parser.add_argument("--base-dir", type=str, default="DeepseekCoder",
                        help="Base directory name on Google Drive")
    parser.add_argument("--use-api", action="store_true", 
                        help="Use Google Drive API instead of rclone")
    
    # Actions
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--sync-datasets", action="store_true",
                            help="Sync datasets to Drive")
    action_group.add_argument("--sync-models", type=str, metavar="MODEL_DIR",
                            help="Sync model files to Drive")
    action_group.add_argument("--sync-logs", action="store_true",
                            help="Sync log files to Drive")
    action_group.add_argument("--sync-results", action="store_true",
                            help="Sync result files to Drive")
    action_group.add_argument("--sync-checkpoints", type=str, metavar="CHECKPOINTS_DIR",
                            help="Sync checkpoint files to Drive")
    action_group.add_argument("--download", type=str, metavar="REMOTE_FOLDER",
                            help="Download files from Drive")
    action_group.add_argument("--sync-all", action="store_true",
                            help="Sync everything to Drive")
    action_group.add_argument("--list-folders", action="store_true",
                            help="List available folder keys in Drive")
    
    # Additional options
    parser.add_argument("--model-name", type=str, default="model",
                        help="Model name for checkpoint syncing")
    parser.add_argument("--is-text-model", action="store_true",
                        help="Whether these are files for text generation model")
    parser.add_argument("--delete-local", action="store_true",
                        help="Delete local files after syncing to Drive")
    parser.add_argument("--local-path", type=str,
                        help="Local path for downloads")
    
    args = parser.parse_args()
    
    # Set up drive sync
    if not setup_sync(args.base_dir, not args.use_api):
        sys.exit(1)
    
    # Execute the requested action
    if args.list_folders:
        list_drive_folders()
        
    elif args.sync_datasets:
        success = sync_datasets_to_drive(args.delete_local)
        if not success:
            sys.exit(1)
        
    elif args.sync_models:
        success = sync_models_to_drive(args.sync_models, args.is_text_model, args.delete_local)
        if not success:
            sys.exit(1)
        
    elif args.sync_logs:
        success = sync_logs_to_drive("logs", args.is_text_model, args.delete_local)
        if not success:
            sys.exit(1)
        
    elif args.sync_results:
        success = sync_results_to_drive("results", args.delete_local)
        if not success:
            sys.exit(1)
        
    elif args.sync_checkpoints:
        success = sync_checkpoints_to_drive(args.sync_checkpoints, args.model_name, args.is_text_model, args.delete_local)
        if not success:
            sys.exit(1)
        
    elif args.download:
        if not args.local_path:
            logger.error("--local-path required for download operation")
            sys.exit(1)
            
        success = download_from_drive(args.download, args.local_path)
        if not success:
            sys.exit(1)
        
    elif args.sync_all:
        # Sync datasets
        if os.path.exists("data/processed"):
            sync_datasets_to_drive(args.delete_local)
        
        # Sync models
        if os.path.exists("models"):
            sync_models_to_drive("models", False, args.delete_local)
        
        # Sync text models if they exist
        text_models_dir = "text_models"
        if os.path.exists(text_models_dir):
            sync_models_to_drive(text_models_dir, True, args.delete_local)
        
        # Sync logs
        if os.path.exists("logs"):
            sync_logs_to_drive("logs", False, args.delete_local)
        
        # Sync results
        if os.path.exists("results"):
            sync_results_to_drive("results", args.delete_local)
    
    logger.info("Drive sync operation completed")

if __name__ == "__main__":
    main() 