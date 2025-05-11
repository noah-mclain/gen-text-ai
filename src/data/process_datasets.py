import os
import json
import argparse
import logging
from pathlib import Path
from preprocessing import DataPreprocessor

# Import drive utils
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.drive_utils import mount_google_drive, setup_drive_directories, get_drive_path

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
    parser.add_argument("--use_drive", action="store_true", 
                        help="Use Google Drive for storage")
    parser.add_argument("--drive_base_dir", type=str, default="DeepseekCoder",
                        help="Base directory on Google Drive (if using Drive)")
    parser.add_argument("--streaming", action="store_true",
                        help="Load datasets in streaming mode to save memory")
    parser.add_argument("--no_cache", action="store_true",
                        help="Disable caching for datasets to save disk space")
    
    args = parser.parse_args()
    
    # Setup Google Drive if requested
    if args.use_drive:
        logger.info("Attempting to mount Google Drive...")
        if mount_google_drive():
            logger.info(f"Setting up directories in Google Drive under {args.drive_base_dir}")
            drive_base = os.path.join("/content/drive/MyDrive", args.drive_base_dir)
            drive_paths = setup_drive_directories(drive_base)
            
            # Update paths to use Google Drive
            args.save_dir = get_drive_path(args.save_dir, drive_paths["preprocessed"], args.save_dir)
            logger.info(f"Processed datasets will be saved to {args.save_dir}")
        else:
            logger.warning("Failed to mount Google Drive. Using local storage instead.")
            args.use_drive = False
    
    # Create save directory
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
    
    logger.info(f"Successfully processed {len(processed_datasets)} datasets")
    for name, dataset in processed_datasets.items():
        logger.info(f"{name}: {len(dataset)} examples")

if __name__ == "__main__":
    main() 