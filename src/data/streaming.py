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
        except Exception as e:
            logger.error(f"Error loading streaming dataset {dataset_name} from Hugging Face: {e}")
            # Return empty dataset as fallback
            logger.warning(f"⚠️ Returning empty dataset for {dataset_name} due to loading error")
            return Dataset.from_dict({
                "input_ids": [],
                "attention_mask": []
            }).to_iterable_dataset()
        
        # Apply tokenization function
        def tokenize_function(examples):
            # Get the column names from the first example
            example_keys = list(examples.keys())
            if not example_keys:
                logger.warning(f"Empty examples for dataset {dataset_name}")
                return {"input_ids": [], "attention_mask": []}
                
            # Determine the text column based on common field names
            potential_text_columns = ["text", "content", "code", "source", "input", "prompt", "instruction"]
            text_column = None
            
            for col in potential_text_columns:
                if col in example_keys:
                    text_column = col
                    break
            
            if text_column is None:
                # Use the first column as fallback
                text_column = example_keys[0]
                
            # Extract the text, ensuring it's in the right format
            texts = examples[text_column]
            
            # Handle different input formats
            if not texts:
                logger.warning(f"Empty texts for dataset {dataset_name}")
                return {"input_ids": [], "attention_mask": []}
                
            # Convert to list of strings if needed
            if isinstance(texts, dict) and "text" in texts:
                # Some datasets nest text in a dict
                texts = texts["text"]
            
            # Ensure we have a list of strings
            if not isinstance(texts, list):
                texts = [texts]
                
            # Ensure each item is a string
            texts = [str(item) if item is not None else "" for item in texts]
            
            # Tokenize the texts
            try:
                tokenized = tokenizer(
                    texts,
                    padding="max_length",
                    truncation=True,
                    max_length=config.get("max_length", 2048),
                    return_tensors="np"  # Use numpy instead of torch tensors
                )
                
                # Convert to lists for Arrow compatibility
                return {
                    "input_ids": tokenized["input_ids"].tolist(),
                    "attention_mask": tokenized["attention_mask"].tolist()
                }
            except Exception as e:
                logger.error(f"Tokenization error for dataset {dataset_name}: {e}")
                # Return empty tensors as fallback
                return {
                    "input_ids": [[]],
                    "attention_mask": [[]]
                }
        
        # Map tokenization to the dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=streaming_config["batch_size"],
            remove_columns=dataset.column_names  # Remove original columns to avoid conflicts
        )
        
        return tokenized_dataset
        
    except Exception as e:
        logger.error(f"Error loading streaming dataset {dataset_name}: {e}")
        
        # Return empty dataset as fallback
        return Dataset.from_dict({
            "input_ids": [],
            "attention_mask": []
        }).to_iterable_dataset() 