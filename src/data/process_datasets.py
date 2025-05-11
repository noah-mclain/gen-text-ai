import os
import json
import argparse
import logging
from pathlib import Path
from .preprocessing import DataPreprocessor

# Import drive utils
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.drive_api_utils import DriveAPI, setup_drive_directories, save_to_drive, load_from_drive

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
    # Setup Google Drive API if requested
    drive_api = None
    directory_ids = None
    
    if args.use_drive_api:
        logger.info(f"Initializing Google Drive API using credentials at {args.credentials_path}...")
        drive_api = DriveAPI(args.credentials_path)
        if drive_api.authenticate(headless=args.headless):
            logger.info(f"Setting up directories in Google Drive under {args.drive_base_dir}")
            directory_ids = drive_api.setup_directories(args.drive_base_dir)
            
            if not directory_ids:
                logger.error("Failed to set up Google Drive directories")
                args.use_drive_api = False
        else:
            logger.error("Failed to authenticate with Google Drive API")
            args.use_drive_api = False
    
    # Create local save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load dataset configuration
    with open(args.config, 'r') as f:
        dataset_config = json.load(f)
    
    # Filter datasets if specified
    if args.datasets:
        dataset_config = {name: config for name, config in dataset_config.items() if name in args.datasets}
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(tokenizer_path=args.tokenizer, max_length=args.max_length)
    
    # Update dataset config with streaming and caching options
    for name, config in dataset_config.items():
        config["streaming"] = args.streaming
        config["use_cache"] = not args.no_cache
    
    # Process datasets
    processed_datasets = preprocessor.load_and_process_all_datasets(dataset_config, args.save_dir)
    
    # If using Google Drive API, upload processed datasets
    if args.use_drive_api and drive_api and directory_ids and "preprocessed" in directory_ids:
        logger.info(f"Uploading processed datasets to Google Drive")
        for dataset_name in processed_datasets.keys():
            dataset_dir = os.path.join(args.save_dir, f"{dataset_name}_processed")
            if os.path.exists(dataset_dir):
                logger.info(f"Uploading {dataset_name} to Google Drive...")
                save_to_drive(drive_api, dataset_dir, directory_ids["preprocessed"])
    
    logger.info(f"Successfully processed {len(processed_datasets)} datasets")
    for name, dataset in processed_datasets.items():
        logger.info(f"{name}: {len(dataset)} examples")

if __name__ == "__main__":
    main() 