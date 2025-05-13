#!/usr/bin/env python3
"""
Comprehensive Codebase Cleanup Tool

This script identifies and removes redundant, obsolete or unnecessary files
from the codebase to improve maintainability and reduce clutter.
"""

import os
import sys
import shutil
import json
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Files in root directory to be removed or archived
ROOT_REDUNDANT_FILES = [
    "quick_fix.py"                # One-time fix script that's no longer needed
]

# Files in scripts directory to be removed or archived
SCRIPT_REDUNDANT_FILES = [
]

# Configuration files to be consolidated
CONFIG_REDUNDANT_FILES = [
]

def clean_root_directory(dry_run=False, archive=False):
    """Clean up redundant files in the root directory."""
    logger.info("Cleaning root directory...")
    
    archive_dir = Path("archived")
    if archive and not archive_dir.exists():
        if not dry_run:
            archive_dir.mkdir(exist_ok=True)
        logger.info(f"Created archive directory: {archive_dir}")
    
    for filename in ROOT_REDUNDANT_FILES:
        file_path = Path(filename)
        if file_path.exists():
            if dry_run:
                logger.info(f"Would {'archive' if archive else 'remove'}: {file_path}")
            else:
                try:
                    if archive:
                        # Move to archive directory
                        shutil.move(str(file_path), str(archive_dir / file_path.name))
                        logger.info(f"Archived: {file_path} -> {archive_dir / file_path.name}")
                    else:
                        # Delete file
                        os.remove(str(file_path))
                        logger.info(f"Removed: {file_path}")
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")

def clean_scripts_directory(dry_run=False, archive=False):
    """Clean up redundant files in the scripts directory."""
    logger.info("Cleaning scripts directory...")
    
    archive_dir = Path("scripts/archived")
    if archive and not archive_dir.exists():
        if not dry_run:
            archive_dir.mkdir(exist_ok=True)
        logger.info(f"Created archive directory: {archive_dir}")
    
    for filename in SCRIPT_REDUNDANT_FILES:
        file_path = Path("scripts") / filename
        if file_path.exists():
            if dry_run:
                logger.info(f"Would {'archive' if archive else 'remove'}: {file_path}")
            else:
                try:
                    if archive:
                        # Move to archive directory
                        shutil.move(str(file_path), str(archive_dir / Path(filename).name))
                        logger.info(f"Archived: {file_path} -> {archive_dir / Path(filename).name}")
                    else:
                        # Delete file
                        os.remove(str(file_path))
                        logger.info(f"Removed: {file_path}")
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")

def merge_config_files(dry_run=False):
    """Merge redundant config files."""
    logger.info("Consolidating configuration files...")
    
    # Consolidate dataset configs
    if os.path.exists("config/dataset_config.json") and os.path.exists("config/dataset_config_updated.json"):
        try:
            # Load both configs
            with open("config/dataset_config.json", "r") as f:
                main_config = json.load(f)
            
            with open("config/dataset_config_updated.json", "r") as f:
                updated_config = json.load(f)
            
            # Back up the main config
            if not dry_run:
                shutil.copy("config/dataset_config.json", "config/dataset_config.json.bak")
                logger.info("Created backup: config/dataset_config.json.bak")
            
            # Update any missing entries in main config
            for key, value in updated_config.items():
                if key not in main_config:
                    main_config[key] = value
                    logger.info(f"Added {key} from updated config to main config")
            
            # Save the merged config
            if not dry_run:
                with open("config/dataset_config.json", "w") as f:
                    json.dump(main_config, f, indent=2)
                logger.info("Merged configurations saved to config/dataset_config.json")
                
                # Remove the redundant config file
                os.remove("config/dataset_config_updated.json")
                logger.info("Removed redundant config: config/dataset_config_updated.json")
            else:
                logger.info("Would merge configurations and remove redundant config file")
                
        except Exception as e:
            logger.error(f"Error merging config files: {e}")

def remove_stack_references(dry_run=False):
    """Remove remaining Stack references from training scripts."""
    logger.info("Removing Stack references from training scripts...")
    
    training_scripts = [
        "train_deepseek_coder.sh",
        "train_optimized.sh",
        "train_a6000_optimized.sh",
        "train_local_test.sh"
    ]
    
    for script in training_scripts:
        if os.path.exists(script):
            try:
                with open(script, "r") as f:
                    content = f.read()
                
                # Remove any lines that reference The Stack dataset
                if "stack" in content.lower():
                    modified = False
                    lines = content.split("\n")
                    filtered_lines = []
                    
                    for line in lines:
                        if "stack" in line.lower() and not line.strip().startswith("#"):
                            logger.info(f"Would remove line from {script}: {line.strip()}")
                            modified = True
                        else:
                            filtered_lines.append(line)
                    
                    if modified and not dry_run:
                        with open(script, "w") as f:
                            f.write("\n".join(filtered_lines))
                        logger.info(f"Removed Stack references from {script}")
                    elif modified:
                        logger.info(f"Would remove Stack references from {script}")
            
            except Exception as e:
                logger.error(f"Error processing {script}: {e}")

def clean_cache_files(dry_run=False):
    """Clean up Python cache files."""
    logger.info("Cleaning Python cache files...")
    
    # Remove __pycache__ directories
    for root, dirs, files in os.walk("."):
        for dir in dirs:
            if dir == "__pycache__":
                cache_dir = os.path.join(root, dir)
                if dry_run:
                    logger.info(f"Would remove cache directory: {cache_dir}")
                else:
                    try:
                        shutil.rmtree(cache_dir)
                        logger.info(f"Removed cache directory: {cache_dir}")
                    except Exception as e:
                        logger.error(f"Error removing {cache_dir}: {e}")
    
    # Remove .pyc files
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".pyc"):
                pyc_file = os.path.join(root, file)
                if dry_run:
                    logger.info(f"Would remove .pyc file: {pyc_file}")
                else:
                    try:
                        os.remove(pyc_file)
                        logger.info(f"Removed .pyc file: {pyc_file}")
                    except Exception as e:
                        logger.error(f"Error removing {pyc_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Clean up redundant files in the codebase")
    
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be changed without making actual changes")
    parser.add_argument("--archive", action="store_true",
                        help="Archive files instead of deleting them")
    parser.add_argument("--no-cache", action="store_true",
                        help="Skip cleaning cache files")
    parser.add_argument("--no-config", action="store_true",
                        help="Skip consolidating config files")
    parser.add_argument("--no-stack", action="store_true",
                        help="Skip removing Stack references")
    
    args = parser.parse_args()
    
    print("\nCleaning up Gen-Text-AI codebase...")
    print(f"Mode: {'Dry run (no changes will be made)' if args.dry_run else 'Live run'}")
    print(f"File handling: {'Archive' if args.archive else 'Delete'} redundant files\n")
    
    # Execute cleaning operations
    clean_root_directory(args.dry_run, args.archive)
    clean_scripts_directory(args.dry_run, args.archive)
    
    if not args.no_config:
        merge_config_files(args.dry_run)
    
    if not args.no_stack:
        remove_stack_references(args.dry_run)
    
    if not args.no_cache:
        clean_cache_files(args.dry_run)
    
    print("\nCleanup completed!")
    if args.dry_run:
        print("This was a dry run. Run without --dry-run to make actual changes.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 