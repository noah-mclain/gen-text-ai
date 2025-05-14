#!/usr/bin/env python3
"""
Fix Dataset Mapping

This script creates symbolic links between the actual dataset directory names and 
the expected dataset directory names to make the datasets discoverable by the training code.
"""

import os
import glob
import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dataset name mappings
DATASET_MAPPINGS = {
    # Expected name -> list of possible actual names
    "codesearchnet_all_processed": ["codesearchnet_all_*_processed", "codesearchnet_all_*_processed_interim_*"],
    "codesearchnet_python_processed": ["codesearchnet_python_processed_*", "codesearchnet_all_python_processed"],
    "codesearchnet_java_processed": ["codesearchnet_java_processed_*", "codesearchnet_all_java_processed"],
    "codesearchnet_javascript_processed": ["codesearchnet_javascript_processed_*", "codesearchnet_all_javascript_processed"],
    "codesearchnet_php_processed": ["codesearchnet_php_processed_*", "codesearchnet_all_php_processed"],
    "codesearchnet_ruby_processed": ["codesearchnet_ruby_processed_*", "codesearchnet_all_ruby_processed"],
    "codesearchnet_go_processed": ["codesearchnet_go_processed_*", "codesearchnet_all_go_processed"],
    "code_alpaca_processed": ["code_alpaca_processed_interim_*"],
    "instruct_code_processed": ["instruct_code_processed_interim_*"],
}

def find_best_match(data_dir: str, patterns: list) -> str:
    """
    Find the best matching directory given the patterns.
    
    Args:
        data_dir: The directory to search in
        patterns: List of glob patterns to search
        
    Returns:
        The best matching directory name or None if not found
    """
    all_matches = []
    for pattern in patterns:
        matches = glob.glob(os.path.join(data_dir, pattern))
        matches = [os.path.basename(m) for m in matches if os.path.isdir(m)]
        all_matches.extend(matches)
    
    if not all_matches:
        return None
    
    # Sort by priority: final > numbered versions > others
    def priority_key(directory):
        if "final" in directory:
            return (0, 0)  # Highest priority
        
        # Extract any numbers for secondary sorting
        import re
        numbers = re.findall(r'\d+', directory)
        if numbers:
            max_number = max(int(n) for n in numbers)
            return (1, -max_number)  # Higher numbers first
        
        return (2, directory)  # Default lowest priority
    
    sorted_matches = sorted(all_matches, key=priority_key)
    return sorted_matches[0] if sorted_matches else None

def create_symlinks(data_dir: str, dry_run: bool = False) -> dict:
    """
    Create symbolic links from expected dataset names to actual dataset directories.
    
    Args:
        data_dir: Directory containing datasets
        dry_run: If True, don't actually create links, just print what would be done
        
    Returns:
        Dictionary mapping expected names to actual directories
    """
    if not os.path.exists(data_dir):
        logger.error(f"Data directory {data_dir} does not exist")
        return {}
    
    symlinks_created = {}
    
    for expected_name, patterns in DATASET_MAPPINGS.items():
        # Skip if the expected name already exists
        expected_path = os.path.join(data_dir, expected_name)
        if os.path.exists(expected_path):
            logger.info(f"Directory {expected_name} already exists, skipping")
            continue
        
        # Find the best match
        best_match = find_best_match(data_dir, patterns)
        if best_match:
            source_path = os.path.join(data_dir, best_match)
            
            # Create the symlink
            if not dry_run:
                try:
                    os.symlink(source_path, expected_path)
                    logger.info(f"Created symlink {expected_name} -> {best_match}")
                    symlinks_created[expected_name] = best_match
                except Exception as e:
                    logger.error(f"Error creating symlink {expected_name} -> {best_match}: {e}")
            else:
                logger.info(f"Would create symlink {expected_name} -> {best_match}")
                symlinks_created[expected_name] = best_match
        else:
            logger.warning(f"No match found for {expected_name}")
    
    return symlinks_created

def list_available_datasets(data_dir: str) -> None:
    """
    List all available dataset directories.
    
    Args:
        data_dir: Directory containing datasets
    """
    if not os.path.exists(data_dir):
        logger.error(f"Data directory {data_dir} does not exist")
        return
    
    # Find all dataset directories
    all_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    # Filter to likely dataset directories
    dataset_dirs = [d for d in all_dirs if "_processed" in d or "_interim_" in d]
    
    if dataset_dirs:
        logger.info(f"Found {len(dataset_dirs)} dataset directories:")
        for i, d in enumerate(sorted(dataset_dirs), 1):
            logger.info(f"{i}. {d}")
    else:
        logger.warning("No dataset directories found")

def main():
    parser = argparse.ArgumentParser(description="Create symlinks for expected dataset names")
    parser.add_argument("--data-dir", default="/notebooks/data/processed", 
                        help="Directory containing datasets (default: /notebooks/data/processed)")
    parser.add_argument("--local-dir", default="data/processed",
                        help="Local directory containing datasets (default: data/processed)")
    parser.add_argument("--list", action="store_true", 
                        help="List available datasets instead of creating symlinks")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't actually create symlinks, just print what would be done")
    parser.add_argument("--fix-both", action="store_true",
                        help="Fix both local and notebook directories")
    
    args = parser.parse_args()
    
    data_dirs = [args.data_dir]
    if args.fix_both:
        data_dirs.append(args.local_dir)
    
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            logger.warning(f"Data directory {data_dir} does not exist, skipping")
            continue
            
        logger.info(f"Working with directory: {data_dir}")
        
        if args.list:
            list_available_datasets(data_dir)
        else:
            symlinks = create_symlinks(data_dir, args.dry_run)
            if symlinks:
                logger.info(f"Created {len(symlinks)} symlinks in {data_dir}")
            else:
                logger.warning(f"No symlinks created in {data_dir}")

if __name__ == "__main__":
    main() 