#!/usr/bin/env python3
"""
Ensure Paperspace Feature Extractor

This script creates the feature_extractor.py file in the Paperspace environment
to fix the 'feature extractor not found' error during training.

This script is specifically designed to run in the Paperspace environment and
will create the feature extractor file in the correct location.
"""

import os
import sys
import logging
from pathlib import Path
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
notebooks_path = Path("/notebooks")
feature_extractor_content = """
""Feature Extractor for Preprocessed Datasets

This module provides utilities for extracting features from preprocessed datasets 
stored on Google Drive, preparing them for training models like DeepSeek-Coder.

It handles Arrow-type datasets and converts them into the appropriate format
for model training.
"""

import os
import logging
import json
from typing import Dict, List, Any, Union, Optional, Tuple
from datasets import Dataset, load_from_disk, concatenate_datasets
from transformers import AutoTokenizer
import torch
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Handles feature extraction from preprocessed datasets for training models.
    """
    
    def __init__(
        self,
        model_name: str,
        max_length: int = 1024,
        padding: str = "max_length",
        truncation: bool = True,
        config_path: Optional[str] = None
    ):
        """
        Initialize the feature extractor.
        
        Args:
            model_name: Base model name (e.g., 'deepseek-ai/deepseek-coder-6.7b-base')
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate sequences longer than max_length
            config_path: Path to the training configuration file
        """
        self.model_name = model_name
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.config = self._load_config(config_path) if config_path else {}
        
        # Load tokenizer
        logger.info(f"Loading tokenizer for {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side="right",
                use_fast=True
            )
            # Set padding token if not already set
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    logger.warning("No padding token or EOS token available, using a default")
                    self.tokenizer.pad_token = self.tokenizer.eos_token = "</s>"
                    
            logger.info(f"Tokenizer loaded successfully with vocabulary size: {len(self.tokenizer)}")
        except Exception as e:
            logger.error(f"Error loading tokenizer: {str(e)}")
            raise
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
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
        
    def load_dataset_from_drive(
        self,
        dataset_path: str, 
        from_google_drive: bool = False,
        drive_manager = None
    ) -> Optional[Dataset]:
        """
        Load a preprocessed dataset from a local path or Google Drive.
        
        Args:
            dataset_path: Path to the dataset
            from_google_drive: Whether to load from Google Drive
            drive_manager: Google Drive manager instance
            
        Returns:
            Loaded dataset or None if loading fails
        """
        try:
            if from_google_drive and drive_manager:
                try:
                    # Try to import directly
                    from src.utils.google_drive_manager import sync_from_drive
                except ImportError:
                    try:
                        # Try relative import
                        from utils.google_drive_manager import sync_from_drive
                    except ImportError:
                        # Last resort, check if we can find it in sys.path
                        import sys
                        import importlib.util
                        
                        # Try to find the module in various places
                        found = False
                        for path in sys.path:
                            module_path = os.path.join(path, 'utils', 'google_drive_manager.py')
                            if os.path.exists(module_path):
                                spec = importlib.util.spec_from_file_location("google_drive_manager", module_path)
                                google_drive_manager = importlib.util.module_from_spec(spec)
                                spec.loader.exec_module(google_drive_manager)
                                sync_from_drive = google_drive_manager.sync_from_drive
                                found = True
                                break
                            
                            # Try src/utils path
                            module_path = os.path.join(path, 'src', 'utils', 'google_drive_manager.py')
                            if os.path.exists(module_path):
                                spec = importlib.util.spec_from_file_location("google_drive_manager", module_path)
                                google_drive_manager = importlib.util.module_from_spec(spec)
                                spec.loader.exec_module(google_drive_manager)
                                sync_from_drive = google_drive_manager.sync_from_drive
                                found = True
                                break
                        
                        if not found:
                            raise ImportError("Could not find google_drive_manager module")
                            
                # First determine the drive folder key
                if "preprocessed" in dataset_path:
                    drive_folder_key = "preprocessed"
                elif "raw" in dataset_path:
                    drive_folder_key = "raw"
                else:
                    drive_folder_key = "data"
                
                # Create local directory if it doesn't exist
                local_dir = Path(dataset_path).parent
                os.makedirs(local_dir, exist_ok=True)
                
                # Sync from drive
                logger.info(f"Syncing dataset from Google Drive ({drive_folder_key}) to {dataset_path}")
                sync_success = sync_from_drive(drive_folder_key, str(local_dir))
                
                if not sync_success:
                    logger.error(f"Failed to sync dataset from Google Drive")
                    return None
                
                logger.info(f"Dataset synced successfully from Google Drive")
            
            # Load the dataset from disk
            logger.info(f"Loading dataset from {dataset_path}")
            
            # Check if path exists
            if not os.path.exists(dataset_path):
                logger.error(f"Dataset path {dataset_path} does not exist")
                return None
                
            # Check if it's an arrow directory by looking for a dataset_info.json file
            if os.path.isfile(os.path.join(dataset_path, "dataset_info.json")):
                dataset = load_from_disk(dataset_path)
                logger.info(f"Loaded Arrow dataset with {len(dataset)} samples")
            else:
                # Look for dataset in the directory structure
                for root, dirs, files in os.walk(dataset_path):
                    if "dataset_info.json" in files:
                        dataset = load_from_disk(root)
                        logger.info(f"Found and loaded dataset from {root} with {len(dataset)} samples")
                        break
                else:
                    logger.error(f"No valid dataset found in {dataset_path}")
                    return None
            
            return dataset
        
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return None
    
    def extract_features(
        self,
        dataset: Dataset,
        text_column: str = "text",
        metadata_column: str = "metadata",
        output_dir: Optional[str] = None,
        batch_size: int = 32,
        num_proc: int = 4,
        save_to_disk: bool = True
    ) -> Optional[Dataset]:
        """
        Extract features from a dataset.
        
        Args:
            dataset: Input dataset
            text_column: Column containing the text data
            metadata_column: Column containing metadata
            output_dir: Directory to save the processed dataset
            batch_size: Batch size for processing
            num_proc: Number of processes for parallel processing
            save_to_disk: Whether to save the processed dataset to disk
            
        Returns:
            Processed dataset with extracted features
        """
        try:
            logger.info(f"Extracting features from dataset with {len(dataset)} samples")
            
            # Define tokenization function
            def tokenize_function(examples):
                # Get the text data
                texts = examples[text_column]
                
                # Tokenize the texts
                tokenized = self.tokenizer(
                    texts,
                    padding=self.padding,
                    truncation=self.truncation,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                # Convert to lists for dataset storage
                result = {
                    "input_ids": tokenized["input_ids"].tolist(),
                    "attention_mask": tokenized["attention_mask"].tolist(),
                }
                
                # Add metadata if available
                if metadata_column in examples and examples[metadata_column] is not None:
                    result["metadata"] = examples[metadata_column]
                
                return result
            
            # Apply tokenization
            try:
                # First try with multiprocessing
                processed_dataset = dataset.map(
                    tokenize_function,
                    batched=True,
                    batch_size=batch_size,
                    num_proc=num_proc,
                    remove_columns=[col for col in dataset.column_names if col != metadata_column],
                    desc="Tokenizing texts"
                )
            except Exception as e:
                logger.warning(f"Multiprocessing failed, falling back to single process: {e}")
                # Fall back to single process
                processed_dataset = dataset.map(
                    tokenize_function,
                    batched=True,
                    batch_size=batch_size,
                    remove_columns=[col for col in dataset.column_names if col != metadata_column],
                    desc="Tokenizing texts (single process)"
                )
            
            logger.info(f"Feature extraction complete. Processed {len(processed_dataset)} samples")
            
            # Save to disk if requested
            if save_to_disk and output_dir:
                os.makedirs(output_dir, exist_ok=True)
                processed_dataset.save_to_disk(output_dir)
                logger.info(f"Saved processed dataset to {output_dir}")
            
            return processed_dataset
        
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return None

def create_feature_extractor():
    """Create the feature_extractor.py file in the Paperspace environment."""
    # Check if we're in Paperspace
    if not notebooks_path.exists():
        logger.info("Not running in Paperspace. No need to create feature extractor.")
        return True
    
    # Define target locations
    target_locations = [
        notebooks_path / "src" / "data" / "processors" / "feature_extractor.py",
    ]
    
    success = True
    for target_file in target_locations:
        # Create parent directory if it doesn't exist
        target_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Write the feature extractor content to the file
            with open(target_file, 'w') as f:
                f.write(feature_extractor_content)
            logger.info(f"Created feature extractor at {target_file}")
        except Exception as e:
            logger.error(f"Failed to create feature extractor at {target_file}: {e}")
            success = False
    
    return success

def main():
    """Main function to create the feature extractor."""
    logger.info("Starting feature extractor creation...")
    
    # Create feature extractor
    success = create_feature_extractor()
    
    if success:
        logger.info("✅ Feature extractor created successfully.")
        return 0
    else:
        logger.warning("⚠️ Feature extractor could not be created.")
        return 1

if __name__ == "__main__":
    sys.exit(main())