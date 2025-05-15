#!/usr/bin/env python3
"""
Prepare Text Datasets for Training

This script prepares preprocessed text datasets for model training
by extracting features using the appropriate tokenizer for the target model.

It is specifically designed for text datasets listed in dataset_config_text.json,
handling their unique fields and requirements.

Usage:
    python scripts/datasets/prepare_text_datasets_for_training.py --model_name <model_name> 
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
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
    logger.warning("Failed to import Google Drive manager. Google Drive functionality will be disabled.")
    DriveManager = None
    sync_from_drive = None
    test_authentication = None

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare text datasets for training")
    
    parser.add_argument("--model_name", type=str, default="google/flan-ul2",
                        help="Name of the model to use for tokenization")
    
    parser.add_argument("--config", type=str, default="config/training_config_text.json",
                        help="Path to the training configuration file")
    
    parser.add_argument("--dataset_config", type=str, default="config/datasets/dataset_config_text.json",
                        help="Path to the dataset configuration file")
    
    parser.add_argument("--output_dir", type=str, default="data/processed/text_features",
                        help="Directory to save the processed features")
    
    parser.add_argument("--from_google_drive", action="store_true",
                        help="Whether to load datasets from Google Drive")
    
    parser.add_argument("--is_encoder_decoder", action="store_true",
                        help="Whether the model is an encoder-decoder model (default for text models)")
    
    parser.add_argument("--max_length", type=int, default=512,
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

def find_config_file(config_path: str) -> Optional[str]:
    """
    Find the configuration file by trying multiple possible locations.
    
    Args:
        config_path: The provided configuration path
        
    Returns:
        The path to the found configuration file or None if not found
    """
    # List of possible locations to check
    possible_locations = [
        config_path,  # Try the provided path first
        os.path.join("config", "datasets", os.path.basename(config_path)),  # Try in config/datasets/
        os.path.join("/notebooks", config_path),  # Try in /notebooks/ (for Paperspace)
        os.path.join("/notebooks/config", "datasets", os.path.basename(config_path)),  # Try in /notebooks/config/datasets/
    ]
    
    # Also check for the file without directory structure if it's just a filename
    base_name = os.path.basename(config_path)
    if base_name != config_path:
        possible_locations.append(base_name)
    
    logger.info(f"Searching for configuration file: {config_path}")
    logger.info(f"Checking these locations: {possible_locations}")
    
    for location in possible_locations:
        if os.path.exists(location):
            logger.info(f"Found configuration file at: {location}")
            return location
    
    # If not found in predefined locations, search recursively starting from current dir and /notebooks
    search_dirs = [".", "/notebooks"]
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        for root, _, files in os.walk(search_dir):
            if os.path.basename(config_path) in files:
                found_path = os.path.join(root, os.path.basename(config_path))
                logger.info(f"Found configuration file at: {found_path}")
                return found_path
    
    logger.error(f"Could not find configuration file at any of these locations: {possible_locations}")
    return None

def load_configuration(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        # Try to find the configuration file
        found_path = find_config_file(config_path)
        if not found_path:
            # Try to find possible files with similar names
            possible_files = []
            search_dirs = [".", "/notebooks"]
            for search_dir in search_dirs:
                if not os.path.exists(search_dir):
                    continue
                
                for root, _, files in os.walk(search_dir):
                    for file in files:
                        if os.path.basename(config_path) in file:
                            possible_files.append(os.path.join(root, file))
            
            if possible_files:
                logger.info(f"Found possible configuration files: {possible_files}")
                # Use the first found file
                found_path = possible_files[0]
            else:
                logger.error(f"No suitable configuration file found")
                return {}
        
        with open(found_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {found_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return {}

def get_dataset_paths(dataset_config: Dict[str, Any], data_dir: str) -> Dict[str, str]:
    """
    Get paths to text datasets based on the configuration.
    
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

def setup_drive_manager(from_google_drive: bool, drive_base_dir: Optional[str] = None) -> Optional[Any]:
    """
    Set up the Google Drive manager if needed.
    
    Args:
        from_google_drive: Whether to load from Google Drive
        drive_base_dir: Base directory name on Google Drive
        
    Returns:
        DriveManager instance or None if not using Google Drive or setup fails
    """
    if not from_google_drive or DriveManager is None:
        return None
        
    try:
        # Test authentication
        if test_authentication and not test_authentication():
            logger.error("Google Drive authentication failed")
            return None
            
        # Create drive manager
        drive_manager = DriveManager(base_dir=drive_base_dir or "TextModels")
        
        # Authenticate
        if not drive_manager.authenticate():
            logger.error("Failed to authenticate with Google Drive")
            return None
            
        return drive_manager
        
    except Exception as e:
        logger.error(f"Error setting up Google Drive manager: {str(e)}")
        return None

class TextFeatureExtractor(FeatureExtractor):
    """
    Extended feature extractor specifically for text datasets.
    """
    
    def prepare_dataset_for_training(
        self,
        dataset,
        text_column: str = "text",
        is_encoder_decoder: bool = True,
        batch_size: int = 1000,
        num_proc: int = 4
    ):
        """
        Prepare a text dataset for training by tokenizing and converting to tensors.
        
        This method extends the base feature extractor's functionality with special
        handling for text datasets, where columns may be different from code datasets.
        
        Args:
            dataset: Input dataset
            text_column: Name of the text column
            is_encoder_decoder: Whether the model is an encoder-decoder model
            batch_size: Batch size for processing
            num_proc: Number of processes for parallel processing
            
        Returns:
            Processed dataset ready for training
        """
        logger.info(f"Preparing text dataset for training with {len(dataset)} samples")
        
        # Check if the text column exists
        if text_column not in dataset.column_names:
            available_columns = ", ".join(dataset.column_names)
            logger.warning(f"Text column '{text_column}' not found in dataset. Available columns: {available_columns}")
            
            # For text datasets, check for common text-related columns
            text_related_columns = ["instruction", "response", "text", "content", "prompt"]
            alt_column = next((col for col in text_related_columns if col in dataset.column_names), None)
            
            if alt_column:
                logger.info(f"Using alternative column '{alt_column}' as text column")
                text_column = alt_column
            else:
                # Create a text column by combining available columns that might contain text
                logger.info(f"Creating text column by combining available columns")
                
                def combine_text_columns(example):
                    # Try common patterns for text datasets
                    if "instruction" in dataset.column_names and "response" in dataset.column_names:
                        return {"text": f"User: {example.get('instruction', '')}\nAssistant: {example.get('response', '')}"}
                    elif "prompt" in dataset.column_names and "completion" in dataset.column_names:
                        return {"text": f"{example.get('prompt', '')}{example.get('completion', '')}"}
                    elif "question" in dataset.column_names and "answer" in dataset.column_names:
                        return {"text": f"Question: {example.get('question', '')}\nAnswer: {example.get('answer', '')}"}
                    else:
                        # Fall back to combining all string columns
                        combined = " ".join([str(example[col]) for col in dataset.column_names 
                                            if isinstance(example[col], (str)) and len(str(example[col])) > 0])
                        return {"text": combined}
                
                dataset = dataset.map(combine_text_columns)
                text_column = "text"
        
        # Continue with the parent class method
        return super().prepare_dataset_for_training(
            dataset=dataset,
            text_column=text_column,
            is_encoder_decoder=is_encoder_decoder,
            batch_size=batch_size,
            num_proc=num_proc
        )

def main():
    """Main function to prepare text datasets for training."""
    args = parse_arguments()
    
    # Load configurations
    training_config = load_configuration(args.config)
    dataset_config = load_configuration(args.dataset_config)
    
    if not dataset_config:
        logger.error("Failed to load dataset configuration")
        sys.exit(1)
    
    # Get model name from config if not specified
    model_name = args.model_name
    if not model_name and training_config and "model" in training_config and "base_model" in training_config["model"]:
        model_name = training_config["model"]["base_model"]
    
    if not model_name:
        logger.error("Model name must be specified through arguments or configuration")
        sys.exit(1)
    
    # Get max_length from config if not specified and config is available
    max_length = args.max_length
    if training_config and "dataset" in training_config and "max_length" in training_config["dataset"]:
        max_length = training_config["dataset"]["max_length"]
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup Google Drive if needed
    drive_base_dir = args.drive_base_dir
    if not drive_base_dir and training_config and "google_drive" in training_config:
        drive_base_dir = training_config["google_drive"].get("base_dir")
    
    drive_manager = setup_drive_manager(args.from_google_drive, drive_base_dir)
    
    # Create feature extractor specifically for text datasets
    feature_extractor = TextFeatureExtractor(
        model_name=model_name,
        max_length=max_length,
        config_path=args.config
    )
    
    # Get dataset paths
    dataset_paths = get_dataset_paths(dataset_config, "data/processed")
    
    if not dataset_paths:
        logger.error("No datasets found in the configuration or all datasets are disabled")
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