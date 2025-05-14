#!/usr/bin/env python3
import os
import json
import argparse
import logging
from pathlib import Path
from .preprocessing import DataPreprocessor
from typing import List, Dict, Optional
import time
import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import drive utils
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# Try multiple import paths to handle reorganized structure
try:
    # First try direct import from reorganized path
    from scripts.google_drive.google_drive_manager import sync_to_drive, configure_sync_method
    logger.info("Successfully imported Drive manager from scripts.google_drive")
except ImportError:
    try:
        # Next try the src.utils path
        from src.utils.google_drive_manager import sync_to_drive, configure_sync_method
        logger.info("Successfully imported Drive manager from src.utils")
    except ImportError:
        # Lastly try direct import from utils dir
        try:
            utils_path = os.path.join(project_root, 'src', 'utils')
            if utils_path not in sys.path:
                sys.path.append(utils_path)
            from google_drive_manager import sync_to_drive, configure_sync_method
            logger.info("Successfully imported Drive manager from utils path")
        except ImportError:
            logger.error("CRITICAL: Could not import Google Drive functions. Syncing will not work.")
            # Provide dummy functions to prevent crashes
            def sync_to_drive(*args, **kwargs):
                logger.error("Drive sync function not available - no files will be synced")
                return False
                
            def configure_sync_method(*args, **kwargs):
                logger.error("Drive sync configuration not available")
                return None

# Try to set HF token from credentials
try:
    from scripts.utilities.set_hf_token import set_hf_token
    logging.info("Attempting to set HF_TOKEN from credentials")
    set_hf_token()
except ImportError:
    logging.warning("Could not import set_hf_token script. HF_TOKEN may not be set.")
except Exception as e:
    logging.warning(f"Error setting HF_TOKEN: {e}")

def process_datasets(config_path: str, datasets: Optional[List[str]] = None, 
                    streaming: bool = False, no_cache: bool = False, 
                    use_drive_api: bool = False, credentials_path: Optional[str] = None,
                    drive_base_dir: Optional[str] = None, headless: bool = False,
                    skip_local_storage: bool = False, verbose: bool = False):
    """Process datasets according to the configuration."""
    
    # Set logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check for existing HF_TOKEN environment variable first
    if os.environ.get("HF_TOKEN"):
        logger.info("Found existing HF_TOKEN in environment variables")
    # Only look in credentials if the token isn't already set
    elif credentials_path and os.path.exists(credentials_path):
        try:
            with open(credentials_path, 'r') as f:
                credentials = json.load(f)

                # Look for Hugging Face token in credentials
                if "huggingface" in credentials:
                    os.environ["HF_TOKEN"] = credentials["huggingface"].get("token", "")
                    logger.info("Set HF_TOKEN environment variable from credentials")
                elif "hf_token" in credentials:
                    os.environ["HF_TOKEN"] = credentials["hf_token"]
                    logger.info("Set HF_TOKEN environment variable from credentials")

                # Also look in other possible locations in the JSON structure
                if "api_keys" in credentials and "huggingface" in credentials["api_keys"]:
                    os.environ["HF_TOKEN"] = credentials["api_keys"]["huggingface"]
                    logger.info("Set HF_TOKEN environment variable from api_keys in credentials")
        except Exception as e:
            logger.error(f"Error loading credentials: {str(e)}")

    # Load dataset configuration
    try:
        with open(config_path, 'r') as f:
            dataset_config = json.load(f)
            logger.info(f"Loaded dataset configuration with {len(dataset_config)} datasets")
    except Exception as e:
        logger.error(f"Error loading dataset configuration: {str(e)}")
        return

    # Filter out explicitly disabled datasets
    filtered_config = {}
    for dataset_name, config in dataset_config.items():
        # Skip disabled datasets
        if isinstance(config, dict) and config.get("enabled") is False:
            logger.info(f"Skipping disabled dataset: {dataset_name}")
            continue
        filtered_config[dataset_name] = config
    dataset_config = filtered_config
    logger.info(f"Found {len(dataset_config)} enabled datasets")

    # Filter datasets if specified
    if datasets:
        filtered_config = {}
        for dataset_name in datasets:
            if dataset_name in dataset_config:
                filtered_config[dataset_name] = dataset_config[dataset_name]
                logger.info(f"Including specified dataset: {dataset_name}")
            else:
                logger.warning(f"Dataset {dataset_name} not found in configuration")
        dataset_config = filtered_config
        logger.info(f"Processing {len(dataset_config)} datasets specified by user")

    # Set streaming option for each dataset
    for dataset_name in dataset_config:
        dataset_config[dataset_name]["streaming"] = streaming
        dataset_config[dataset_name]["use_cache"] = not no_cache
        logger.debug(f"Dataset {dataset_name} configuration: streaming={streaming}, use_cache={not no_cache}")

    # Initialize preprocessor
    preprocessor = DataPreprocessor()

    # Process datasets
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/processed"))
    logger.info(f"Using output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Make sure the output directory has write permissions
    test_file_path = os.path.join(output_dir, ".write_test")
    try:
        with open(test_file_path, "w") as f:
            f.write("test")
        os.remove(test_file_path)
        logger.info(f"Confirmed write access to output directory: {output_dir}")
    except Exception as e:
        logger.error(f"Cannot write to output directory {output_dir}: {e}")
        logger.error("Please check directory permissions")
        return

    # Configure Google Drive syncing
    if use_drive_api:
        # Configure whether to use rclone or the Google Drive API
        # By default, use rclone if available since it's more reliable
        try:
            # Check if rclone is available
            import subprocess
            result = subprocess.run(
                ["rclone", "version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            use_rclone = result.returncode == 0
        except Exception:
            use_rclone = False

        # Ensure drive_base_dir is a string and not a boolean or None
        if drive_base_dir is None or drive_base_dir is True:
            # Set a default value if None or True
            drive_base_dir = "DeepseekCoder"
            logger.warning(f"Invalid drive_base_dir provided, using default: {drive_base_dir}")

        # Configure the drive sync method
        configure_sync_method(use_rclone=use_rclone, base_dir=drive_base_dir)
        logger.info(f"Configured Drive sync using {'rclone' if use_rclone else 'Google Drive API'}")

    # If skipping local storage, create a temporary directory for processing
    if skip_local_storage and use_drive_api:
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="dataset_processing_")
        logger.info(f"Using temporary directory for processing: {temp_dir}")
        output_dir = os.path.join(temp_dir, "processed")
        os.makedirs(output_dir, exist_ok=True)

    # Process the datasets - measure time for logging
    start_time = time.time()
    logger.info(f"Starting dataset processing at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    processed_datasets = preprocessor.load_and_process_all_datasets(dataset_config, output_dir)
    end_time = time.time()
    processing_time = end_time - start_time
    logger.info(f"Dataset processing completed in {processing_time:.2f} seconds")

    # Check if any datasets were processed
    if not processed_datasets:
        logger.warning("No datasets were processed successfully")
    else:
        logger.info(f"Successfully processed {len(processed_datasets)} datasets")
        for name in processed_datasets.keys():
            logger.info(f"  - {name}")

    # Upload processed datasets if using Drive API
    if use_drive_api:
        logger.info("Syncing processed datasets to Google Drive")
        try:
            # Sync the processed datasets to Drive
            for dataset_name in processed_datasets:
                dataset_path = os.path.join(output_dir, f"{dataset_name}_processed")
                if os.path.exists(dataset_path):
                    # Sync to the "preprocessed" directory in Drive
                    logger.info(f"Syncing dataset {dataset_name} to Drive")
                    success = sync_to_drive(
                        dataset_path, 
                        "preprocessed", 
                        delete_source=skip_local_storage,
                        update_only=False  # We want to ensure all files are synced
                    )

                    if success:
                        logger.info(f"Successfully synced dataset {dataset_name} to Drive")
                    else:
                        logger.error(f"Failed to sync dataset {dataset_name} to Drive")

            # If using a temporary directory, clean it up
            if skip_local_storage and 'temp_dir' in locals():
                import shutil
                try:
                    shutil.rmtree(temp_dir)
                    logger.info(f"Cleaned up temporary directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary directory: {e}")
        except Exception as e:
            logger.error(f"Error syncing datasets to Google Drive: {str(e)}")

    logger.info(f"Processed {len(processed_datasets)} datasets")
    return processed_datasets

def main():
    parser = argparse.ArgumentParser(description="Process datasets for fine-tuning deepseek-coder")
    parser.add_argument("--config", type=str, default="../../config/dataset_config.json", 
                        help="Path to dataset configuration file")
    parser.add_argument("--save_dir", type=str, default="../../data/processed",
                        help="Directory to save processed datasets")
    parser.add_argument("--tokenizer", type=str, default="deepseek-ai/deepseek-coder-6.7b-base",
                        help="Path or name of the tokenizer to use")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum token length for examples")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Optional list of dataset names to process (if not specified, all will be processed)")
    
    # Google Drive API options
    parser.add_argument("--use_drive_api", action="store_true", 
                        help="Use Google Drive API for storage (for Paperspace)")
    parser.add_argument("--use_drive", action="store_true", 
                        help="Use Google Drive with rclone (preferred)")
    parser.add_argument("--credentials_path", type=str, default="credentials.json",
                        help="Path to Google Drive API credentials JSON file")
    parser.add_argument("--drive_base_dir", type=str, default="DeepseekCoder",
                        help="Base directory on Google Drive")
    parser.add_argument("--headless", action="store_true",
                        help="Use headless authentication for environments without a browser")
    
    # Memory efficiency options
    parser.add_argument("--streaming", action="store_true",
                        help="Load datasets in streaming mode to save memory")
    parser.add_argument("--no_cache", action="store_true",
                        help="Disable caching for datasets to save disk space")
    parser.add_argument("--skip_local_storage", action="store_true",
                        help="Skip storing datasets locally, sync directly to Drive")
    
    args = parser.parse_args()
    
    process_datasets(
        args.config, 
        args.datasets, 
        args.streaming, 
        args.no_cache,
        args.use_drive_api or args.use_drive, 
        args.credentials_path,
        args.drive_base_dir, 
        args.headless,
        args.skip_local_storage
    )

if __name__ == "__main__":
    main() 