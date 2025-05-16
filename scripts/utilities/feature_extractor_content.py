"""
Feature Extractor for Preprocessed Datasets

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
    
    def tokenize_examples(
        self, 
        examples: Dict[str, List],
        text_column: str = "text",
        is_encoder_decoder: bool = False
    ) -> Dict[str, List]:
        """
        Tokenize a batch of examples.
        
        Args:
            examples: Dictionary of examples
            text_column: Name of the text column
            is_encoder_decoder: Whether the model is an encoder-decoder model
            
        Returns:
            Dictionary of tokenized examples
        """
        # Check if we have data to tokenize
        if not examples or text_column not in examples or not examples[text_column]:
            logger.warning(f"Empty examples or missing text column {text_column}")
            return {"input_ids": [], "attention_mask": []}
            
        texts = examples[text_column]
        
        # Tokenize
        try:
            tokenized = self.tokenizer(
                texts,
                max_length=self.max_length,
                padding=self.padding,
                truncation=self.truncation,
                return_tensors="pt"
            )
            
            result = {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
            }
            
            # For encoder-decoder models, also set the labels to be the same as inputs
            if is_encoder_decoder:
                result["labels"] = tokenized["input_ids"].clone()
            else:
                # For causal language models, shift the inputs to create labels
                labels = tokenized["input_ids"].clone()
                # Shift right and pad with -100 (ignored in loss calculation)
                labels[:, :-1] = labels[:, 1:].clone()
                labels[:, -1] = -100
                result["labels"] = labels
                
            return result
            
        except Exception as e:
            logger.error(f"Error tokenizing examples: {e}")
            # Return empty tensors as fallback
            return {
                "input_ids": torch.zeros((len(texts), 1), dtype=torch.long) if texts else torch.zeros((0, 1), dtype=torch.long),
                "attention_mask": torch.ones((len(texts), 1), dtype=torch.long) if texts else torch.zeros((0, 1), dtype=torch.long),
                "labels": torch.zeros((len(texts), 1), dtype=torch.long) if texts else torch.zeros((0, 1), dtype=torch.long)
            }
    
    def prepare_dataset_for_training(
        self,
        dataset: Dataset,
        text_column: str = "text",
        is_encoder_decoder: bool = False,
        batch_size: int = 1000,
        num_proc: int = 4
    ) -> Dataset:
        """
        Prepare a dataset for training by tokenizing and converting to tensors.
        
        Args:
            dataset: Input dataset
            text_column: Name of the text column
            is_encoder_decoder: Whether the model is an encoder-decoder model
            batch_size: Batch size for processing
            num_proc: Number of processes for parallel processing
            
        Returns:
            Processed dataset ready for training
        """
        if dataset is None:
            logger.error("Cannot prepare None dataset for training")
            return None
            
        logger.info(f"Preparing dataset for training with {len(dataset)} samples")
        
        # Check if the text column exists
        if text_column not in dataset.column_names:
            available_columns = ", ".join(dataset.column_names)
            logger.error(f"Text column '{text_column}' not found in dataset. Available columns: {available_columns}")
            
            # Try to find an alternative text column
            alt_columns = ["instruction", "response", "content", "code", "source"]
            alt_column = next((col for col in alt_columns if col in dataset.column_names), None)
            
            if alt_column:
                logger.info(f"Using alternative column '{alt_column}' as text column")
                text_column = alt_column
            else:
                # Create a text column by combining available columns
                logger.info(f"Creating text column by combining available columns")
                
                def combine_text_columns(example):
                    combined = " ".join([str(example[col]) for col in dataset.column_names 
                                        if isinstance(example[col], (str, int, float))])
                    return {"text": combined}
                
                dataset = dataset.map(combine_text_columns)
                text_column = "text"
        
        # Define the tokenization function
        def tokenize_function(examples):
            return self.tokenize_examples(examples, text_column, is_encoder_decoder)
        
        try:
            # Process the dataset with error handling
            logger.info(f"Tokenizing dataset using {text_column} column")
            
            # Reduce batch size for very large examples to avoid OOM issues
            avg_length = 0
            try:
                # Sample a few examples to estimate typical length
                sample_size = min(100, len(dataset))
                sample_indices = np.random.choice(len(dataset), sample_size, replace=False)
                sample = dataset.select(sample_indices)
                
                total_chars = sum(len(str(ex[text_column])) for ex in sample)
                avg_length = total_chars / sample_size
                
                # Adjust batch size based on average text length
                if avg_length > 10000:
                    batch_size = max(10, batch_size // 10)
                    logger.info(f"Texts are very long (avg {avg_length:.0f} chars), reducing batch size to {batch_size}")
                elif avg_length > 5000:
                    batch_size = max(50, batch_size // 5)
                    logger.info(f"Texts are long (avg {avg_length:.0f} chars), reducing batch size to {batch_size}")
            except Exception as e:
                logger.warning(f"Error estimating text length: {e}")
            
            # Process the dataset
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                batch_size=batch_size,
                num_proc=num_proc,
                remove_columns=dataset.column_names,  # Remove original columns to save memory
                desc="Tokenizing examples"
            )
            
            # Set the format to PyTorch tensors
            tokenized_dataset.set_format("torch")
            
            logger.info(f"Dataset preparation completed with {len(tokenized_dataset)} samples")
            return tokenized_dataset
            
        except Exception as e:
            logger.error(f"Error preparing dataset for training: {e}")
            return None
    
    def extract_features_from_drive_datasets(
        self,
        dataset_paths: Dict[str, str],
        output_dir: str,
        from_google_drive: bool = True,
        text_column: str = "text",
        is_encoder_decoder: bool = False,
        save_to_disk: bool = True,
        drive_manager = None
    ) -> Optional[Dataset]:
        """
        Extract features from multiple datasets stored on Google Drive.
        
        Args:
            dataset_paths: Dictionary mapping dataset names to paths
            output_dir: Directory to save the processed dataset
            from_google_drive: Whether to load from Google Drive
            text_column: Name of the text column
            is_encoder_decoder: Whether the model is an encoder-decoder model
            save_to_disk: Whether to save the processed dataset to disk
            drive_manager: Google Drive manager instance
            
        Returns:
            Combined processed dataset or None if processing fails
        """
        processed_datasets = []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        for name, path in dataset_paths.items():
            logger.info(f"Processing dataset {name} from {path}")
            
            # Load the dataset
            dataset = self.load_dataset_from_drive(path, from_google_drive, drive_manager)
            if dataset is None:
                logger.warning(f"Failed to load dataset {name}, skipping")
                continue
                
            # Prepare for training
            try:
                processed = self.prepare_dataset_for_training(
                    dataset,
                    text_column=text_column,
                    is_encoder_decoder=is_encoder_decoder
                )
                
                if processed is None:
                    logger.warning(f"Failed to process dataset {name}, skipping")
                    continue
                    
                processed_datasets.append(processed)
                logger.info(f"Successfully processed dataset {name} with {len(processed)} samples")
                
                # Save individual dataset if requested
                if save_to_disk:
                    dataset_output_dir = os.path.join(output_dir, f"{name}_features")
                    os.makedirs(dataset_output_dir, exist_ok=True)
                    processed.save_to_disk(dataset_output_dir)
                    logger.info(f"Saved processed features for {name} to {dataset_output_dir}")
                    
            except Exception as e:
                logger.error(f"Error processing dataset {name}: {str(e)}")
        
        if not processed_datasets:
            logger.error("No datasets were successfully processed")
            return None
            
        # Combine all processed datasets
        logger.info(f"Combining {len(processed_datasets)} processed datasets")
        try:
            combined_dataset = concatenate_datasets(processed_datasets)
            
            # Save the combined dataset
            if save_to_disk:
                combined_output_dir = os.path.join(output_dir, "combined_features")
                os.makedirs(combined_output_dir, exist_ok=True)
                combined_dataset.save_to_disk(combined_output_dir)
                logger.info(f"Saved combined features to {combined_output_dir}")
                
            return combined_dataset
        except Exception as e:
            logger.error(f"Error combining datasets: {e}")
            
            # If combination fails but we have at least one dataset, return the first one
            if processed_datasets:
                logger.warning("Returning only the first processed dataset as fallback")
                return processed_datasets[0]
            return None 