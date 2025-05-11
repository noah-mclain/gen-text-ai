#!/usr/bin/env python3
import os
import json
import argparse
import logging
from pathlib import Path
from .preprocessing import DataPreprocessor
from typing import List, Dict, Optional

# Import drive utils
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.drive_api_utils import DriveAPI, setup_drive_directories, save_to_drive, load_from_drive

# Try to set HF token from credentials
try:
    from scripts.set_hf_token import set_hf_token
    logging.info("Attempting to set HF_TOKEN from credentials")
    set_hf_token()
except ImportError:
    logging.warning("Could not import set_hf_token script. HF_TOKEN may not be set.")
except Exception as e:
    logging.warning(f"Error setting HF_TOKEN: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_datasets(config_path: str, datasets: Optional[List[str]] = None, 
                    streaming: bool = False, no_cache: bool = False, 
                    use_drive_api: bool = False, credentials_path: Optional[str] = None,
                    drive_base_dir: Optional[str] = None, headless: bool = False):
    """Process datasets according to the configuration."""
    
    # Check and update environment variable for credential-based authentication
    if credentials_path and os.path.exists(credentials_path):
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
    except Exception as e:
        logger.error(f"Error loading dataset configuration: {str(e)}")
        return
    
    # Filter datasets if specified
    if datasets:
        filtered_config = {}
        for dataset_name in datasets:
            if dataset_name in dataset_config:
                filtered_config[dataset_name] = dataset_config[dataset_name]
            else:
                logger.warning(f"Dataset {dataset_name} not found in configuration")
        dataset_config = filtered_config
    
    # Set streaming option for each dataset
    for dataset_name in dataset_config:
        dataset_config[dataset_name]["streaming"] = streaming
        dataset_config[dataset_name]["use_cache"] = not no_cache
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Process datasets
    output_dir = "data/processed"
    
    # Update output directory if using Drive API
    if use_drive_api:
        from utils.drive_api_utils import initialize_drive_api, setup_drive_directories
        
        logger.info("Initializing Google Drive API for dataset storage")
        drive_api = initialize_drive_api(credentials_path, headless=headless)
        
        if drive_api and drive_api.authenticated:
            logger.info(f"Setting up directories in Google Drive under {drive_base_dir}")
            directory_ids = setup_drive_directories(drive_api, drive_base_dir)
            
            if directory_ids and "data" in directory_ids and "processed" in directory_ids["data"]:
                logger.info("Will save processed datasets to Google Drive")
            else:
                logger.error("Failed to set up Google Drive directories for datasets")
    
    # Process the datasets
    processed_datasets = preprocessor.load_and_process_all_datasets(dataset_config, output_dir)
    
    # Upload processed datasets if using Drive API
    if use_drive_api and drive_api and drive_api.authenticated and directory_ids:
        from utils.drive_api_utils import save_to_drive
        
        logger.info("Uploading processed datasets to Google Drive")
        if "data" in directory_ids and "processed" in directory_ids["data"]:
            save_to_drive(drive_api, output_dir, directory_ids["data"]["processed"])
            logger.info("Successfully processed and uploaded datasets")
    
    logger.info(f"Processed {len(processed_datasets)} datasets")

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
    
    args = parser.parse_args()
    
    process_datasets(
        args.config, 
        args.datasets, 
        args.streaming, 
        args.no_cache,
        args.use_drive_api, 
        args.drive_base_dir, 
        args.headless
    )

if __name__ == "__main__":
    main() 