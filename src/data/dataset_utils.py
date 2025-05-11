import os
import logging
import random
import traceback
from typing import Dict, List, Optional, Union, Tuple, Any
from datasets import Dataset, DatasetDict, concatenate_datasets

# Try different import patterns for load_from_disk and load_dataset
try:
    from datasets import load_from_disk, load_dataset
except ImportError:
    # Fallback to alternative imports if needed
    try:
        from huggingface_datasets import load_from_disk, load_dataset
    except ImportError:
        # Define placeholders that will raise informative errors
        def load_from_disk(path, **kwargs):
            raise ImportError("Could not import load_from_disk from datasets or huggingface_datasets")
        
        def load_dataset(path, **kwargs):
            raise ImportError("Could not import load_dataset from datasets or huggingface_datasets")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_processed_datasets(data_dir: str, dataset_names: Optional[List[str]] = None, 
                           streaming: bool = False, use_cache: bool = True,
                           max_samples: Optional[Dict[str, int]] = None) -> Dict[str, Dataset]:
    """
    Load processed datasets from disk with robust error handling.
    
    Args:
        data_dir: Directory containing processed datasets
        dataset_names: Optional list of dataset names to load (if None, load all datasets in the directory)
        streaming: Whether to load datasets in streaming mode to save memory
        use_cache: Whether to use the cache for datasets
        max_samples: Optional dictionary mapping dataset names to maximum number of samples to load
    
    Returns:
        Dictionary mapping dataset names to loaded datasets
    """
    if not os.path.exists(data_dir):
        logger.warning(f"Data directory {data_dir} does not exist - creating it")
        try:
            os.makedirs(data_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create data directory: {e}")
        return {}
    
    # Get all dataset directories
    try:
        # Only look for directories that end with _processed
        dataset_dirs = [d for d in os.listdir(data_dir) 
                      if os.path.isdir(os.path.join(data_dir, d)) and d.endswith("_processed")]
    except Exception as e:
        logger.error(f"Error listing directory {data_dir}: {e}")
        dataset_dirs = []
    
    # Filter by dataset names if specified
    if dataset_names:
        # Match partial names - if the directory contains any of the names, include it
        dataset_dirs = [d for d in dataset_dirs 
                      if any(name in d for name in dataset_names)]
    
    if not dataset_dirs:
        logger.warning(f"No datasets found in {data_dir}" + 
                      (f" matching {dataset_names}" if dataset_names else ""))
        return {}
    
    # Set cache directory if needed
    if not use_cache:
        os.environ["HF_DATASETS_CACHE"] = "no"
    
    # Load each dataset
    datasets = {}
    for dir_name in dataset_dirs:
        try:
            dataset_path = os.path.join(data_dir, dir_name)
            
            # Extract the base name without _processed
            name = dir_name.replace("_processed", "")
            
            # Skip if the dataset is not requested (exact match)
            if dataset_names and name not in dataset_names:
                continue
                
            logger.info(f"Loading dataset {name} from {dataset_path}")
            
            # Load the dataset
            dataset = load_from_disk(dataset_path, keep_in_memory=not streaming)
            
            # Apply max_samples if specified
            if max_samples and name in max_samples and max_samples[name] > 0:
                max_to_take = min(max_samples[name], len(dataset))
                if max_to_take < len(dataset):
                    logger.info(f"Limiting {name} dataset to {max_to_take} samples (from {len(dataset)})")
                    # Select a random sample if not streaming
                    if not streaming:
                        dataset = dataset.shuffle(seed=42).select(range(max_to_take))
            
            datasets[name] = dataset
            logger.info(f"Successfully loaded dataset {name} with {len(dataset) if hasattr(dataset, '__len__') else 'unknown'} examples")
            
        except Exception as e:
            logger.error(f"Error loading dataset {dir_name}: {str(e)}")
            logger.debug(traceback.format_exc())
    
    # Log stats about loaded datasets
    if datasets:
        logger.info(f"Successfully loaded {len(datasets)} datasets: {', '.join(datasets.keys())}")
    else:
        logger.warning("No datasets could be loaded")
    
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
    
    # Load dataset with error handling
    try:
        logger.info(f"Loading dataset {dataset_path} (streaming={streaming}, cached={use_cache})")
        return load_dataset(dataset_path, name=name, split=split, streaming=streaming)
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_path}: {str(e)}")
        logger.debug(traceback.format_exc())
        
        # Return an empty dataset as a fallback
        if not streaming:
            return Dataset.from_dict({"text": []})
        else:
            # For streaming, return an empty generator
            def empty_gen():
                if False:  # Will never yield
                    yield {"text": ""}
            return empty_gen()

def combine_datasets(datasets: Dict[str, Dataset], weights: Optional[Dict[str, float]] = None,
                   interleave_prob: Optional[float] = None, seed: int = 42) -> Optional[Dataset]:
    """
    Combine multiple datasets into a single dataset, optionally with weighted sampling.
    
    Args:
        datasets: Dictionary mapping dataset names to datasets
        weights: Optional dictionary mapping dataset names to weights (if None, equal weights will be used)
        interleave_prob: If provided, interleave datasets with this probability instead of concatenating
        seed: Random seed for interleaving
    
    Returns:
        Combined dataset or None if no datasets are provided
    """
    if not datasets:
        logger.warning("No datasets provided to combine")
        return None
    
    # If only one dataset, return it directly
    if len(datasets) == 1:
        name, dataset = next(iter(datasets.items()))
        logger.info(f"Only one dataset ({name}) provided, returning it directly")
        return dataset
    
    # If weights not provided, use equal weights
    if weights is None:
        weights = {name: 1.0 for name in datasets.keys()}
    
    # Ensure all datasets have weights
    for name in datasets.keys():
        if name not in weights:
            logger.warning(f"No weight specified for dataset {name}, using default weight of 1.0")
            weights[name] = 1.0
    
    # Filter out datasets with zero weight
    valid_datasets = {name: dataset for name, dataset in datasets.items() 
                    if name in weights and weights[name] > 0}
    
    if not valid_datasets:
        logger.warning("No datasets with positive weights to combine")
        return None
    
    # Calculate the number of examples to sample from each dataset
    valid_weights = {name: weights[name] for name in valid_datasets.keys()}
    total_weight = sum(valid_weights.values())
    normalized_weights = {name: weight / total_weight for name, weight in valid_weights.items()}
    
    try:
        # Interleave or concatenate datasets
        if interleave_prob is not None and interleave_prob > 0:
            # Prepare datasets for interleaving
            dataset_dict = {}
            probabilities = []
            
            for name, dataset in valid_datasets.items():
                dataset_dict[name] = dataset
                probabilities.append(normalized_weights[name])
            
            logger.info(f"Interleaving {len(dataset_dict)} datasets with probabilities: {dict(zip(dataset_dict.keys(), probabilities))}")
            
            # Create a DatasetDict and interleave
            try:
                from datasets import interleave_datasets
                combined = interleave_datasets(
                    list(dataset_dict.values()),
                    probabilities=probabilities,
                    seed=seed,
                    stopping_strategy="first_exhausted"
                )
                return combined
            except ImportError:
                # Fallback to alternative implementation if interleave_datasets not available
                logger.warning("interleave_datasets not available, falling back to concatenation")
                combined_datasets = list(valid_datasets.values())
                return concatenate_datasets(combined_datasets)
        else:
            # Combine datasets by concatenation
            logger.info(f"Concatenating {len(valid_datasets)} datasets")
            combined_datasets = list(valid_datasets.values())
            return concatenate_datasets(combined_datasets)
    except Exception as e:
        logger.error(f"Error combining datasets: {str(e)}")
        logger.debug(traceback.format_exc())
        
        # Return the largest dataset as a fallback
        largest_dataset = max(valid_datasets.items(), key=lambda x: len(x[1]) if hasattr(x[1], '__len__') else 0)
        logger.warning(f"Returning largest dataset ({largest_dataset[0]}) as fallback")
        return largest_dataset[1]

def create_train_val_test_split(dataset: Dataset, train_size: float = 0.8, val_size: float = 0.1, 
                               test_size: float = 0.1, seed: int = 42,
                               streaming: bool = False) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split a dataset into train, validation, and test sets with robust error handling.
    
    Args:
        dataset: Dataset to split
        train_size: Proportion of data for training
        val_size: Proportion of data for validation
        test_size: Proportion of data for testing
        seed: Random seed for reproducibility
        streaming: Whether the dataset is in streaming mode
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    if dataset is None:
        logger.error("Cannot split None dataset")
        # Return empty datasets instead of None to prevent errors
        empty_dataset = Dataset.from_dict({"text": []})
        return empty_dataset, empty_dataset, empty_dataset
        
    try:
        # Check if dataset is empty
        if hasattr(dataset, '__len__') and len(dataset) == 0:
            logger.warning("Cannot split empty dataset")
            return dataset, dataset, dataset
    except Exception as e:
        logger.warning(f"Could not check dataset length: {e}")
    
    # Ensure sizes sum to 1
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        logger.warning(f"Sizes {train_size}, {val_size}, {test_size} do not sum to 1, normalizing")
        total = train_size + val_size + test_size
        train_size /= total
        val_size /= total
        test_size /= total
    
    try:
        if streaming:
            logger.warning("Splitting streaming datasets is approximate and may not be reproducible")
            
            # Try to use the datasets library's streaming split functionality if available
            try:
                from datasets import IterableDataset
                
                # Check if the dataset is already an IterableDataset
                if isinstance(dataset, IterableDataset):
                    # Use built-in streaming split for iterable datasets if available
                    splits = dataset.train_test_split(
                        test_size=val_size + test_size,
                        seed=seed,
                        shuffle=True
                    )
                    train_dataset = splits["train"]
                    
                    # Further split the test into validation and test
                    remaining_ratio = test_size / (val_size + test_size)
                    remaining_splits = splits["test"].train_test_split(
                        test_size=remaining_ratio,
                        seed=seed,
                        shuffle=True
                    )
                    
                    return train_dataset, remaining_splits["train"], remaining_splits["test"]
            except (ImportError, AttributeError) as e:
                logger.warning(f"Could not use IterableDataset split functionality: {e}")
            
            # Fallback: Create a buffered generator for each split
            def create_stream_generator(gen_function):
                """Create a dataset-like object from a generator function"""
                try:
                    from datasets import IterableDataset
                    # Convert generator to IterableDataset if possible
                    return IterableDataset.from_generator(gen_function)
                except (ImportError, AttributeError):
                    # Fallback to just returning the generator function
                    return gen_function()
            
            # Generator function to apply random split
            def split_generator(dataset, train_prop, val_prop, test_prop, seed=42):
                random.seed(seed)
                for example in dataset:
                    r = random.random()
                    if r < train_prop:
                        yield ("train", example)
                    elif r < train_prop + val_prop:
                        yield ("validation", example)
                    else:
                        yield ("test", example)
            
            # Generator functions for each split
            def train_gen():
                for split, ex in split_generator(dataset, train_size, val_size, test_size, seed):
                    if split == "train":
                        yield ex
                        
            def val_gen():
                for split, ex in split_generator(dataset, train_size, val_size, test_size, seed):
                    if split == "validation":
                        yield ex
                        
            def test_gen():
                for split, ex in split_generator(dataset, train_size, val_size, test_size, seed):
                    if split == "test":
                        yield ex
            
            # Create dataset objects
            return create_stream_generator(train_gen), create_stream_generator(val_gen), create_stream_generator(test_gen)
        else:
            # For normal datasets, use the standard train_test_split method
            train_val_test = dataset.train_test_split(test_size=val_size + test_size, seed=seed)
            train_dataset = train_val_test["train"]
            
            # Split the test set into validation and test
            val_test_ratio = val_size / (val_size + test_size)
            val_test = train_val_test["test"].train_test_split(test_size=1-val_test_ratio, seed=seed)
            
            return train_dataset, val_test["train"], val_test["test"]
    except Exception as e:
        logger.error(f"Error splitting dataset: {str(e)}")
        logger.debug(traceback.format_exc())
        
        # Return empty datasets to prevent further errors
        try:
            # Try to create an empty dataset with the same structure
            if hasattr(dataset, 'features'):
                empty_dict = {k: [] for k in dataset.features.keys()}
                empty_dataset = Dataset.from_dict(empty_dict)
            else:
                # Fallback to a simple empty dataset
                empty_dataset = Dataset.from_dict({"text": []})
                
            logger.warning("Returning empty datasets for all splits as fallback")
            return empty_dataset, empty_dataset, empty_dataset
        except Exception:
            # Last resort fallback
            logger.warning("Returning the original dataset for all splits as fallback")
            return dataset, dataset, dataset