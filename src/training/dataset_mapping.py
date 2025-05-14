#!/usr/bin/env python
import os
import glob
import logging
from typing import Dict, List, Optional, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mapping between dataset config names and actual directory patterns
DATASET_MAPPING = {
    # Dataset name from config -> list of possible directory prefixes in descending order of preference
    "codesearchnet_all": ["codesearchnet_all_processed", "codesearchnet_all_*_processed", "codesearchnet_all_*_processed_interim_*"],
    "codesearchnet_python": ["codesearchnet_python_processed", "codesearchnet_all_python_processed", "codesearchnet_python_processed_interim_*"],
    "codesearchnet_java": ["codesearchnet_java_processed", "codesearchnet_all_java_processed", "codesearchnet_java_processed_interim_*"],
    "codesearchnet_javascript": ["codesearchnet_javascript_processed", "codesearchnet_all_javascript_processed", "codesearchnet_javascript_processed_interim_*"],
    "codesearchnet_php": ["codesearchnet_php_processed", "codesearchnet_all_php_processed", "codesearchnet_php_processed_interim_*"],
    "codesearchnet_ruby": ["codesearchnet_ruby_processed", "codesearchnet_all_ruby_processed", "codesearchnet_ruby_processed_interim_*"],
    "codesearchnet_go": ["codesearchnet_go_processed", "codesearchnet_all_go_processed", "codesearchnet_go_processed_interim_*"],
    "code_alpaca": ["code_alpaca_processed", "code_alpaca_processed_interim_*"],
    "instruct_code": ["instruct_code_processed", "instruct_code_processed_interim_*"],
    "mbpp": ["mbpp_processed"],
    "humaneval": ["humaneval_processed"],
    "codeparrot": ["codeparrot_processed"],
}

def find_matching_directories(data_dir: str, pattern: str) -> List[str]:
    """
    Find all directories in data_dir that match the given pattern.
    
    Args:
        data_dir: Directory to search in
        pattern: Pattern to match (supports * wildcard)
        
    Returns:
        List of directory names (not full paths) that match the pattern
    """
    if not os.path.exists(data_dir):
        return []
        
    # If pattern has wildcards, use glob
    if "*" in pattern:
        # Convert to glob pattern by replacing specific characters
        glob_pattern = os.path.join(data_dir, pattern)
        matching_paths = glob.glob(glob_pattern)
        return [os.path.basename(path) for path in matching_paths if os.path.isdir(path)]
    else:
        # Direct path check
        direct_path = os.path.join(data_dir, pattern)
        if os.path.isdir(direct_path):
            return [pattern]
        return []

def prioritize_directories(directories: List[str]) -> List[str]:
    """
    Sort directories by priority: final > highest number > earliest in list.
    
    Args:
        directories: List of directory names
        
    Returns:
        Sorted list of directory names
    """
    def priority_key(directory):
        # Final versions have highest priority
        if "final" in directory:
            return (0, "")
            
        # For numbered versions, extract the number and sort by it
        import re
        numbers = re.findall(r'\d+', directory)
        if numbers:
            # Use the highest number found
            max_number = max(int(n) for n in numbers)
            # Negative because we want higher numbers first
            return (1, -max_number)
        
        # Default priority based on position in list
        return (2, directory)
    
    return sorted(directories, key=priority_key)

def map_dataset_names(dataset_names: List[str], data_dirs: List[str]) -> Dict[str, str]:
    """
    Map dataset names to their full paths by searching in multiple data directories.
    
    Args:
        dataset_names: List of dataset names from config
        data_dirs: List of directories to search in (e.g., local dir, /notebooks dir)
        
    Returns:
        Dictionary mapping dataset names to their full paths
    """
    result = {}
    
    for dataset_name in dataset_names:
        if dataset_name not in DATASET_MAPPING:
            logger.warning(f"No mapping defined for dataset {dataset_name}")
            continue
            
        # Get potential directory patterns for this dataset
        patterns = DATASET_MAPPING[dataset_name]
        
        # Search each data directory
        for data_dir in data_dirs:
            if not data_dir or not os.path.exists(data_dir):
                continue
                
            all_matching = []
            
            # Try each pattern
            for pattern in patterns:
                matching = find_matching_directories(data_dir, pattern)
                if matching:
                    all_matching.extend(matching)
            
            # If we found matches, prioritize them
            if all_matching:
                prioritized = prioritize_directories(all_matching)
                best_match = prioritized[0]
                result[dataset_name] = os.path.join(data_dir, best_match)
                logger.info(f"Mapped dataset {dataset_name} to {result[dataset_name]}")
                break
    
    return result

def get_dataset_paths(dataset_names: List[str]) -> Dict[str, str]:
    """
    Get the full paths for the given dataset names, checking in both local and notebooks directories.
    
    Args:
        dataset_names: List of dataset names from config
        
    Returns:
        Dictionary mapping dataset names to their full paths
    """
    # Check in multiple directories
    data_dirs = [
        "data/processed",  # Local directory
        "/notebooks/data/processed"  # Notebooks directory on Paperspace
    ]
    
    # Only include directories that exist
    data_dirs = [d for d in data_dirs if d and os.path.exists(d)]
    
    if not data_dirs:
        logger.warning("No data directories found")
        return {}
        
    return map_dataset_names(dataset_names, data_dirs)

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        dataset_names = sys.argv[1:]
        paths = get_dataset_paths(dataset_names)
        for name, path in paths.items():
            print(f"{name}: {path}")
    else:
        print("Usage: python dataset_mapping.py dataset1 dataset2 ...") 