#!/usr/bin/env python3
"""
Prepare Datasets for Training

This script prepares preprocessed datasets from Google Drive for model training
by extracting features using the appropriate tokenizer for the target model.

Usage:
    python scripts/prepare_datasets_for_training.py --model_name <model_name> --config <config_file>
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import feature extractor
from src.data.processors.feature_extractor import FeatureExtractor

# Import Google Drive manager
try:
    from src.utils.google_drive_manager import DriveManager, sync_from_drive, test_authentication
except ImportError:
    logger.error("Failed to import Google Drive manager. Make sure the required dependencies are installed.")
    sys.exit(1)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare datasets for training")
    
    parser.add_argument("--model_name", type=str, default="deepseek-ai/deepseek-coder-6.7b-base",
                        help="Name of the model to use for tokenization")
    
    parser.add_argument("--config", type=str, default="config/training_config_text.json",
                        help="Path to the training configuration file")
    
    parser.add_argument("--dataset_config", type=str, default="config/dataset_config_text.json",
                        help="Path to the dataset configuration file")
    
    parser.add_argument("--output_dir", type=str, default="data/processed/features",
                        help="Directory to save the processed features")
    
    parser.add_argument("--from_google_drive", action="store_true",
                        help="Whether to load datasets from Google Drive")
    
    parser.add_argument("--is_encoder_decoder", action="store_true",
                        help="Whether the model is an encoder-decoder model")
    
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum sequence length for tokenization")
    
    parser.add_argument("--text_column", type=str, default="text",
                        help="Name of the text column in the dataset")
    
    parser.add_argument("--no_save", action="store_true",
                        help="Don't save the processed datasets to disk")
    
    parser.add_argument("--batch_size", type=int, default=1000,
                        help="Batch size for processing")
    
    parser.add_argument("--num_proc", type=int, default=4,
                        help="Number of processes for parallel processing")
    
    parser.add_argument("--drive_base_dir", type=str, default=None,
                        help="Base directory name on Google Drive")
    
    return parser.parse_args()

def load_configuration(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return {}

def get_dataset_paths(dataset_config: Dict[str, Any], data_dir: str) -> Dict[str, str]:
    """
    Get paths to datasets based on the configuration.
    
    Args:
        dataset_config: Dataset configuration
        data_dir: Base data directory
        
    Returns:
        Dictionary mapping dataset names to their paths
    """
    dataset_paths = {}
    
    for name, config in dataset_config.items():
        if not config.get("enabled", True):
            logger.info(f"Dataset {name} is disabled, skipping")
            continue
            
        # Get the processor type
        processor = config.get("processor", "standardize_text")
        
        # Construct the path
        path = os.path.join(data_dir, f"{name}_{processor}_processed")
        
        dataset_paths[name] = path
        logger.info(f"Added dataset {name} with path {path}")
    
    return dataset_paths

def setup_drive_manager(from_google_drive: bool, drive_base_dir: Optional[str] = None) -> Optional[DriveManager]:
    """
    Set up the Google Drive manager if needed.
    
    Args:
        from_google_drive: Whether to load from Google Drive
        drive_base_dir: Base directory name on Google Drive
        
    Returns:
        DriveManager instance or None if not using Google Drive or setup fails
    """
    if not from_google_drive:
        return None
        
    try:
        # Test authentication
        auth_success = test_authentication()
        if not auth_success:
            logger.error("Google Drive authentication failed")
            return None
            
        # Create drive manager
        drive_manager = DriveManager(base_dir=drive_base_dir or "DeepseekCoder")
        
        # Authenticate
        if not drive_manager.authenticate():
            logger.error("Failed to authenticate with Google Drive")
            return None
            
        return drive_manager
        
    except Exception as e:
        logger.error(f"Error setting up Google Drive manager: {str(e)}")
        return None

def main():
    """Main function to prepare datasets for training."""
    args = parse_arguments()
    
    # Load configurations
    training_config = load_configuration(args.config)
    dataset_config = load_configuration(args.dataset_config)
    
    if not training_config or not dataset_config:
        logger.error("Failed to load required configurations")
        sys.exit(1)
    
    # Get model name from config if not specified
    model_name = args.model_name
    if not model_name and "model" in training_config and "base_model" in training_config["model"]:
        model_name = training_config["model"]["base_model"]
    
    if not model_name:
        logger.error("Model name must be specified through arguments or configuration")
        sys.exit(1)
    
    # Get max_length from config if not specified
    max_length = args.max_length
    if "dataset" in training_config and "max_length" in training_config["dataset"]:
        max_length = training_config["dataset"]["max_length"]
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup Google Drive if needed
    drive_base_dir = args.drive_base_dir
    if not drive_base_dir and "google_drive" in training_config:
        drive_base_dir = training_config["google_drive"].get("base_dir")
    
    drive_manager = setup_drive_manager(args.from_google_drive, drive_base_dir)
    
    # Create feature extractor
    feature_extractor = FeatureExtractor(
        model_name=model_name,
        max_length=max_length,
        config_path=args.config
    )
    
    # Get dataset paths
    dataset_paths = get_dataset_paths(dataset_config, "data/processed")
    
    if not dataset_paths:
        logger.error("No datasets found in the configuration")
        sys.exit(1)
    
    # Extract features from datasets
    processed_dataset = feature_extractor.extract_features_from_drive_datasets(
        dataset_paths=dataset_paths,
        output_dir=output_dir,
        from_google_drive=args.from_google_drive,
        text_column=args.text_column,
        is_encoder_decoder=args.is_encoder_decoder,
        save_to_disk=not args.no_save,
        drive_manager=drive_manager
    )
    
    if processed_dataset is None:
        logger.error("Failed to process datasets")
        sys.exit(1)
    
    logger.info(f"Successfully processed {len(processed_dataset)} examples in total")
    logger.info(f"Feature extraction complete. Features saved to {output_dir}")

if __name__ == "__main__":
    main() 