import os
import logging
import random
from typing import Dict, List, Optional, Union
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk, load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_processed_datasets(data_dir: str, dataset_names: Optional[List[str]] = None, 
                           streaming: bool = False, use_cache: bool = True) -> Dict[str, Dataset]:
    """
    Load processed datasets from disk.
    
    Args:
        data_dir: Directory containing processed datasets
        dataset_names: Optional list of dataset names to load (if None, load all datasets in the directory)
        streaming: Whether to load datasets in streaming mode to save memory
        use_cache: Whether to use the cache for datasets
    
    Returns:
        Dictionary mapping dataset names to loaded datasets
    """
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory {data_dir} does not exist")
    
    # Get all dataset directories
    dataset_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    # Filter by dataset names if specified
    if dataset_names:
        dataset_dirs = [d for d in dataset_dirs if any(d.startswith(name) for name in dataset_names)]
    
    if not dataset_dirs:
        logger.warning(f"No datasets found in {data_dir}")
        return {}
    
    # Set cache directory if needed
    if not use_cache:
        os.environ["HF_DATASETS_CACHE"] = "no"
    
    # Load each dataset
    datasets = {}
    for dir_name in dataset_dirs:
        try:
            dataset_path = os.path.join(data_dir, dir_name)
            dataset = load_from_disk(dataset_path, keep_in_memory=not streaming)
            name = dir_name.replace("_processed", "")
            datasets[name] = dataset
            logger.info(f"Loaded dataset {name} with {len(dataset)} examples" + 
                      f" (streaming={streaming}, cached={use_cache})")
        except Exception as e:
            logger.error(f"Error loading dataset {dir_name}: {str(e)}")
    
    return datasets

def load_raw_dataset(dataset_path: str, name: Optional[str] = None, split: Optional[str] = None, 
                    streaming: bool = False, use_cache: bool = True) -> Dataset:
    """
    Load a raw dataset from Hugging Face in streaming mode.
    
    Args:
        dataset_path: Path to the dataset on HuggingFace
        name: Optional dataset configuration name
        split: Optional dataset split to load
        streaming: Whether to load the dataset in streaming mode
        use_cache: Whether to use caching for the dataset

    Returns:
        Loaded dataset
    """
    # Set cache directory if needed
    if not use_cache:
        os.environ["HF_DATASETS_CACHE"] = "no"
    
    # Load dataset
    logger.info(f"Loading dataset {dataset_path} (streaming={streaming}, cached={use_cache})")
    return load_dataset(dataset_path, name=name, split=split, streaming=streaming)

def combine_datasets(datasets: Dict[str, Dataset], weights: Optional[Dict[str, float]] = None,
                   interleave_prob: Optional[float] = None, seed: int = 42) -> Dataset:
    """
    Combine multiple datasets into a single dataset, optionally with weighted sampling.
    
    Args:
        datasets: Dictionary mapping dataset names to datasets
        weights: Optional dictionary mapping dataset names to weights (if None, equal weights will be used)
        interleave_prob: If provided, interleave datasets with this probability instead of concatenating
        seed: Random seed for interleaving
    
    Returns:
        Combined dataset
    """
    if not datasets:
        raise ValueError("No datasets provided")
    
    # If weights not provided, use equal weights
    if weights is None:
        weights = {name: 1.0 for name in datasets.keys()}
    
    # Ensure all datasets have weights
    for name in datasets.keys():
        if name not in weights:
            logger.warning(f"No weight specified for dataset {name}, using default weight of 1.0")
            weights[name] = 1.0
    
    # Calculate the number of examples to sample from each dataset
    total_weight = sum(weights.values())
    normalized_weights = {name: weight / total_weight for name, weight in weights.items()}
    
    # Interleave or concatenate datasets
    if interleave_prob is not None:
        # Prepare datasets for interleaving
        dataset_dict = {}
        probabilities = []
        
        for name, dataset in datasets.items():
            dataset_dict[name] = dataset
            probabilities.append(normalized_weights[name])
        
        # Interleave datasets
        return DatasetDict(dataset_dict).interleave_datasets(
            probabilities=probabilities,
            seed=seed,
            stopping_strategy="first_exhausted"
        )
    else:
        # Combine datasets by concatenation
        combined_datasets = []
        for name, dataset in datasets.items():
            combined_datasets.append(dataset)
        
        return concatenate_datasets(combined_datasets)

def create_train_val_test_split(dataset: Dataset, train_size: float = 0.9, val_size: float = 0.05, 
                               test_size: float = 0.05, seed: int = 42,
                               streaming: bool = False) -> DatasetDict:
    """
    Split a dataset into train, validation, and test sets.
    
    Args:
        dataset: Dataset to split
        train_size: Proportion of data for training
        val_size: Proportion of data for validation
        test_size: Proportion of data for testing
        seed: Random seed for reproducibility
        streaming: Whether the dataset is in streaming mode
    
    Returns:
        DatasetDict with train, validation, and test splits
    """
    # Ensure sizes sum to 1
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        logger.warning(f"Sizes {train_size}, {val_size}, {test_size} do not sum to 1, normalizing")
        total = train_size + val_size + test_size
        train_size /= total
        val_size /= total
        test_size /= total
    
    if streaming:
        # For streaming datasets, we need to shuffle and then split
        # This is more memory-efficient but might be slower
        dataset = dataset.shuffle(seed=seed, buffer_size=10000)
        
        def split_generator(dataset, train_end, val_end):
            for i, example in enumerate(dataset):
                if i < train_end:
                    yield "train", example
                elif i < val_end:
                    yield "validation", example
                else:
                    yield "test", example
        
        # Estimate dataset size (this is approximate for streaming datasets)
        try:
            dataset_size = len(dataset)
        except TypeError:
            # For truly streaming datasets without known length, use a large default size
            logger.warning("Dataset size unknown, using 100,000 as default for splitting")
            dataset_size = 100000
        
        train_end = int(dataset_size * train_size)
        val_end = int(dataset_size * (train_size + val_size))
        
        return dataset.to_dict().add_column("split", split_generator(dataset, train_end, val_end))
    else:
        # For normal datasets, use the standard train_test_split method
        train_val_test = dataset.train_test_split(test_size=val_size + test_size, seed=seed)
        train_dataset = train_val_test["train"]
        
        # Split the test set into validation and test
        val_test_ratio = val_size / (val_size + test_size)
        val_test = train_val_test["test"].train_test_split(test_size=1-val_test_ratio, seed=seed)
        
        return DatasetDict({
            "train": train_dataset,
            "validation": val_test["train"],
            "test": val_test["test"]
        }) 