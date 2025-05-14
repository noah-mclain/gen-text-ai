"""
Streaming dataset utilities for handling large datasets efficiently.
This module provides functionality for loading and processing datasets in a streaming fashion.
"""

import os
import logging
import json
from typing import Dict, Any, Optional
from datasets import load_dataset, IterableDataset, Dataset, load_from_disk

logger = logging.getLogger(__name__)

# Import the new dataset mapping module
try:
    from src.training.dataset_mapping import get_dataset_paths
except ImportError:
    # Define a simple function in case the import fails
    def get_dataset_paths(dataset_names):
        return {}

def load_streaming_dataset(
    dataset_name: str,
    tokenizer,
    config: Dict[str, Any] = None,
    num_workers: int = 1,
    data_dir: str = "data/processed"
) -> IterableDataset:
    """
    Load a dataset in streaming mode.
    
    Args:
        dataset_name: Name of the dataset
        tokenizer: Tokenizer to use for preprocessing
        config: Configuration settings for streaming
        num_workers: Number of workers for parallel processing
        data_dir: Directory containing processed datasets
        
    Returns:
        An IterableDataset for streaming
    """
    config = config or {}
    logger.info(f"Loading streaming dataset: {dataset_name}")
    
    # Default configuration for streaming
    default_config = {
        "buffer_size": 10000,
        "batch_size": 1000,
    }
    
    # Update with user-provided config
    streaming_config = {**default_config, **config}
    
    try:
        # First try to find the dataset using our new mapping
        dataset_paths = get_dataset_paths([dataset_name])
        local_dataset_path = dataset_paths.get(dataset_name)
        
        # If not found with the mapping, try the old direct path
        if not local_dataset_path:
            # Generate conventional path
            local_dataset_path = os.path.join(data_dir, f"{dataset_name}_processed")
        
        if os.path.exists(local_dataset_path):
            try:
                # Load the dataset from disk
                logger.info(f"Loading dataset {dataset_name} from local path: {local_dataset_path}")
                dataset = load_from_disk(local_dataset_path)
                
                # Convert to iterable dataset for streaming if needed
                if not isinstance(dataset, IterableDataset):
                    logger.info(f"Converting local dataset {dataset_name} to streaming format")
                    dataset = dataset.to_iterable_dataset(num_shards=num_workers)
                
                # Standardize dataset schema to ensure compatibility with other datasets
                dataset = standardize_dataset_features(dataset, tokenizer, config)
                
                logger.info(f"✅ Successfully loaded dataset {dataset_name} from local storage")
                return dataset
            except Exception as local_err:
                logger.warning(f"Failed to load local dataset {dataset_name}: {local_err}. Falling back to Hub.")
        else:
            logger.info(f"Local dataset not found at {local_dataset_path}, trying Hub...")
            
        # Check if we have a dataset configuration available
        dataset_path = dataset_name
        split = "train"
        try:
            # Try to load the dataset config to get the correct path
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "dataset_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    dataset_configs = json.load(f)
                if dataset_name in dataset_configs and "path" in dataset_configs[dataset_name]:
                    dataset_path = dataset_configs[dataset_name]["path"]
                    if "split" in dataset_configs[dataset_name]:
                        split = dataset_configs[dataset_name]["split"]
                    logger.info(f"Using path '{dataset_path}' from config for dataset '{dataset_name}'")
        except Exception as config_err:
            logger.warning(f"Error loading dataset config: {config_err}. Using dataset name as path.")
        
        # Special case for HumanEval which only has a test split
        if dataset_name == "humaneval" or dataset_path == "openai/openai_humaneval":
            logger.info("HumanEval dataset detected, using 'test' split instead of 'train'")
            split = "test"
            
        # Try to load from Hugging Face datasets
        logger.info(f"Loading dataset from Hugging Face: {dataset_path} (split: {split})")
        try:
            dataset = load_dataset(
                dataset_path, 
                streaming=True,
                split=split,
                token=os.environ.get("HF_TOKEN") if "HF_TOKEN" in os.environ else None
            )
            logger.info(f"✅ Successfully loaded {dataset_name} from Hugging Face Hub")
            
            # Standardize dataset schema to ensure compatibility
            dataset = standardize_dataset_features(dataset, tokenizer, config)
            
        except Exception as e:
            logger.error(f"Error loading streaming dataset {dataset_name} from Hugging Face: {e}")
            # Return empty dataset as fallback
            logger.warning(f"⚠️ Returning empty dataset for {dataset_name} due to loading error")
            return Dataset.from_dict({
                "input_ids": [],
                "attention_mask": []
            }).to_iterable_dataset()
        
        return dataset
        
    except Exception as e:
        logger.error(f"Error loading streaming dataset {dataset_name}: {e}")
        
        # Return empty dataset as fallback
        return Dataset.from_dict({
            "input_ids": [],
            "attention_mask": []
        }).to_iterable_dataset()

def standardize_dataset_features(dataset, tokenizer, config):
    """
    Standardize dataset features to ensure compatibility when combining datasets.
    
    Args:
        dataset: Dataset to standardize
        tokenizer: Tokenizer to use for tokenization
        config: Configuration settings
        
    Returns:
        Standardized dataset with consistent features
    """
    logger.info(f"Standardizing dataset features to ensure compatibility")
    
    max_length = config.get("max_length", 2048)
    
    # Function to process and tokenize a single example
    def process_example(example):
        # Determine the text field in the example
        text_field = None
        
        # Check if processed_text exists
        if "processed_text" in example:
            text_field = "processed_text"
        # Check if text exists
        elif "text" in example:
            text_field = "text"
        # Fall back to checking common alternatives
        else:
            for field in ["content", "code", "source", "input", "prompt", "instruction"]:
                if field in example:
                    text_field = field
                    break
        
        # If we couldn't find a text field, generate an empty example
        if text_field is None or example[text_field] is None:
            # Return a minimal valid example with empty content
            return {
                "input_ids": [tokenizer.pad_token_id],
                "attention_mask": [1],
                "labels": [tokenizer.pad_token_id],
            }
        
        # Get the text and ensure it's a string
        text = example[text_field]
        if not isinstance(text, str):
            # If it's not a string, try to convert it
            if isinstance(text, (list, tuple)) and len(text) > 0:
                # If it's a list/tuple, join with newlines
                text = "\n".join(str(item) for item in text if item is not None)
            else:
                # Otherwise just convert to string
                text = str(text) if text is not None else ""
        
        # Tokenize the text
        try:
            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="np"
            )
            
            # Convert to lists for Arrow compatibility
            return {
                "input_ids": tokenized["input_ids"][0].tolist(),
                "attention_mask": tokenized["attention_mask"][0].tolist(),
                "labels": tokenized["input_ids"][0].tolist(),  # For causal language modeling, labels = input_ids
            }
        except Exception as e:
            logger.warning(f"Error tokenizing text: {e}")
            # Return a minimal valid example with pad tokens
            return {
                "input_ids": [tokenizer.pad_token_id],
                "attention_mask": [1],
                "labels": [tokenizer.pad_token_id],
            }
    
    # Map the processing function over the dataset
    return dataset.map(
        process_example,
        remove_columns=dataset.column_names  # Remove original columns to avoid conflicts
    ) 