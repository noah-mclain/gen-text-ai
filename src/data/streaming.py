"""
Streaming dataset utilities for handling large datasets efficiently.
This module provides functionality for loading and processing datasets in a streaming fashion.
"""

import os
import logging
from typing import Dict, Any, Optional
from datasets import load_dataset, IterableDataset, Dataset

logger = logging.getLogger(__name__)

def load_streaming_dataset(
    dataset_name: str,
    tokenizer,
    config: Dict[str, Any] = None,
    num_workers: int = 1
) -> IterableDataset:
    """
    Load a dataset in streaming mode.
    
    Args:
        dataset_name: Name of the dataset
        tokenizer: Tokenizer to use for preprocessing
        config: Configuration settings for streaming
        num_workers: Number of workers for parallel processing
        
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
        # Try to load from Hugging Face datasets
        dataset = load_dataset(
            dataset_name, 
            streaming=True,
            split="train"
        )
        
        # Apply tokenization function
        def tokenize_function(examples):
            # Get the column names from the first example
            text_column = next(iter(examples.keys())) if len(examples) > 0 else "text"
            
            # Tokenize the texts
            tokenized = tokenizer(
                examples[text_column],
                padding="max_length",
                truncation=True,
                max_length=config.get("max_length", 2048),
                return_tensors="pt"
            )
            
            return tokenized
        
        # Map tokenization to the dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=streaming_config["batch_size"],
            num_proc=num_workers
        )
        
        return tokenized_dataset
        
    except Exception as e:
        logger.error(f"Error loading streaming dataset {dataset_name}: {e}")
        
        # Return empty dataset as fallback
        return Dataset.from_dict({
            "input_ids": [],
            "attention_mask": []
        }).to_iterable_dataset() 