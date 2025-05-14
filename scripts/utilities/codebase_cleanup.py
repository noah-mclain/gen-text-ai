#!/usr/bin/env python3
"""
Codebase Cleanup Script

This script organizes the codebase by moving files from the root directory 
to appropriate subdirectories based on their purpose.
"""

import os
import shutil
import logging
from pathlib import Path
import json
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define file mappings
FILE_MAPPINGS = {
    "scripts": [
        # Training scripts
        "train_a6000_optimized.sh",
        "train_deepseek_coder.sh",
        "train_local_test.sh",
        "train_optimized.sh",
        "train_text_flan_a6000.sh",
        "train_text_flan.sh",
        "train_without_wandb.sh",
        
        # Utility scripts
        "cleanup_stack_datasets.sh",
        "cuda_env_setup.sh",
        "fix_before_training.sh",
        "fix_dataset_features.sh",
        "fix_wandb.sh",
        "install_language_dependencies.sh",
        "paperspace_setup.sh",
    ],
    "src/data": [
        # Data processing scripts
        "fix_dataset_mapping.py",
    ],
    "src/utils": [
        # Utility scripts
        "setup_google_drive.py",
    ],
    "docs": [
        # Documentation files
        "GOOGLE_DRIVE_SETUP.md",
    ],
    "tests": [
        # Test files
        "test_flan_ul2.py",
    ],
    "src": [
        # Main application files
        "main_api.py",
        "main.py",
    ],
    "config": [
        # Configuration files that don't contain sensitive data
        # credentials.json is omitted as it might contain sensitive data
    ],
    "archived": [
        # Files that should be archived rather than organized
    ]
}

# Files that should be left in the root directory
ROOT_FILES = [
    ".gitattributes",
    ".gitignore",
    "LICENSE",
    "README.md",
    "requirements.txt",
    "credentials.json",  # Kept in root for now due to potential sensitivity
]

def create_directories():
    """Ensure all necessary directories exist."""
    for directory in FILE_MAPPINGS.keys():
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

def move_file(src_path, dest_dir):
    """Move a file to the specified directory."""
    if not os.path.exists(src_path):
        logger.warning(f"Source file not found: {src_path}")
        return False
        
    dest_path = os.path.join(dest_dir, os.path.basename(src_path))
    
    # Check if destination already exists
    if os.path.exists(dest_path):
        logger.warning(f"Destination file already exists: {dest_path}")
        
        # Check if files are identical
        import filecmp
        if filecmp.cmp(src_path, dest_path, shallow=False):
            logger.info(f"Files are identical. Removing source: {src_path}")
            os.remove(src_path)
            return True
            
        # Create a backup of the existing file
        backup_path = f"{dest_path}.bak"
        shutil.copy2(dest_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
    
    # Move the file
    try:
        shutil.move(src_path, dest_path)
        logger.info(f"Moved {src_path} to {dest_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to move {src_path} to {dest_path}: {e}")
        return False

def update_imports(moved_files):
    """Update import statements in Python files to reflect the new file organization."""
    python_files = []
    
    # Find Python files only in project directories, skipping environment and venv dirs
    skip_dirs = ['.env', 'venv', '.git', '.vscode', '.cursor', '__pycache__']
    
    for root, dirs, files in os.walk("."):
        # Skip directories we want to exclude
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            if file.endswith(".py"):
                # Only process files in our project directories
                if not any(excluded in root for excluded in skip_dirs):
                    python_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(python_files)} Python files to check for import updates")
    
    # Update import statements
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            modified = False
            
            # Update import statements for each moved file
            for src_file, dest_dir in moved_files.items():
                if not src_file.endswith(".py"):
                    continue
                    
                src_module = os.path.splitext(os.path.basename(src_file))[0]
                dest_module = os.path.join(dest_dir, src_module).replace("/", ".")
                
                # Simple import
                if f"import {src_module}" in content and f"import {dest_module}" not in content:
                    content = content.replace(f"import {src_module}", f"import {dest_module}")
                    modified = True
                
                # From import
                if f"from {src_module} import" in content and f"from {dest_module} import" not in content:
                    content = content.replace(f"from {src_module} import", f"from {dest_module} import")
                    modified = True
            
            # Write back the modified content
            if modified:
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"Updated imports in {py_file}")
                
        except UnicodeDecodeError:
            logger.warning(f"Skipping {py_file} due to encoding issues")
        except Exception as e:
            logger.error(f"Error processing {py_file}: {e}")

def create_init_files():
    """Create __init__.py files in directories that don't have them."""
    for directory in FILE_MAPPINGS.keys():
        if directory.startswith("src/") or directory == "src":
            init_file = os.path.join(directory, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write('"""Module for {0}."""\n'.format(directory.split("/")[-1]))
                logger.info(f"Created {init_file}")

def organize_codebase(dry_run=False):
    """Organize the codebase by moving files to appropriate directories."""
    create_directories()
    
    moved_files = {}
    
    for dest_dir, files in FILE_MAPPINGS.items():
        for file in files:
            if dry_run:
                logger.info(f"Would move {file} to {dest_dir}")
            else:
                if move_file(file, dest_dir):
                    moved_files[file] = dest_dir
    
    if not dry_run:
        # Create necessary __init__.py files
        create_init_files()
        
        # Update import statements to reflect new file locations
        update_imports(moved_files)
    
    return moved_files

def main():
    parser = argparse.ArgumentParser(description="Organize codebase by moving files to appropriate directories")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually move files, just print what would be done")
    parser.add_argument("--skip-imports", action="store_true", help="Skip updating import statements")
    
    args = parser.parse_args()
    
    logger.info("Starting codebase organization...")
    
    if args.dry_run:
        logger.info("Performing dry run - no files will be moved")
    
    moved_files = organize_codebase(dry_run=args.dry_run)
    
    if args.dry_run:
        logger.info(f"Would move {len(moved_files)} files")
    else:
        logger.info(f"Successfully moved {len(moved_files)} files")
    
    logger.info("Codebase organization complete")

if __name__ == "__main__":
    main() 