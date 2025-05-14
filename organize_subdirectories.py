#!/usr/bin/env python3
"""
Subdirectory Organization Script

This script further organizes files within the major directories (src, scripts, docs, config, tests)
by moving them to more specific subdirectories based on their purpose.
"""

import os
import shutil
import logging
from pathlib import Path
import argparse
import importlib.util
import re
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define file mappings for each major directory
FILE_MAPPINGS = {
    # Scripts directory organization
    "scripts": {
        "scripts/training": [
            "train_a6000_optimized.sh",
            "train_deepseek_coder.sh",
            "train_text_flan_a6000.sh",
            "train_text_flan.sh",
            "train_without_wandb.sh",
            "train_optimized.sh",
            "train_local_test.sh",
            "train_text_flan.py"
        ],
        "scripts/environment": [
            "fix_environment.py",
            "fix_cuda.py",
            "cuda_env_setup.sh",
            "paperspace_setup.sh",
            "install_language_dependencies.sh",
            "check_paperspace_env.py",
            "fix_xformers_env.py",
            "fix_before_training.sh"
        ],
        "scripts/deepspeed": [
            "clean_deepspeed_env.py",
            "fix_deepspeed.py",
            "purge_deepspeed.py",
            "diagnose_deepspeed.sh"
        ],
        "scripts/datasets": [
            "prepare_datasets_for_training.py",
            "prepare_drive_datasets.sh",
            "process_datasets.sh",
            "save_datasets.py",
            "fix_dataset_features.sh",
            "process_updated.sh",
            "cleanup_stack_datasets.sh"
        ],
        "scripts/google_drive": [
            "setup_google_drive.py",
            "sync_to_drive.py",
            "google_drive_manager.py"
        ],
        "scripts/utilities": [
            "codebase_cleanup.py",
            "fix_wandb.sh",
            "fix_token.py",
            "set_hf_token.py",
            "cleanup.sh",
            "optimized_commands.sh",
            "ensure_feature_extractor.py",
            "run_paperspace.sh",
            # Keep README.md in the main scripts directory
        ]
    },

    # Docs directory organization
    "docs": {
        "docs/setup": [
            "GOOGLE_DRIVE_SETUP.md",
            "SERVICE_ACCOUNT_GUIDE.md",
            "PAPERSPACE.md",
            "PAPERSPACE_DRIVE_GUIDE.md"
        ],
        "docs/training": [
            "FEATURE_EXTRACTION_GUIDE.md",
            "STACK_TRAINING.md",
            "TEXT_GENERATION_GUIDE.md"
        ],
        "docs/implementation": [
            "IMPLEMENTATION_SUMMARY.md",
            "PREPROCESSING_IMPROVEMENTS.md"
        ],
        "docs/datasets": [
            "DATASET_CHECKING.md",
            "DIRECT_STACK_GUIDE.md"
        ]
        # Keep README.md in the main docs directory
    },

    # Config directory organization
    "config": {
        "config/training": [
            "training_config.json",
            "training_config_text.json",
            "training_config_low_memory.json"
        ],
        "config/datasets": [
            "dataset_config.json",
            "dataset_config_text.json",
            "dataset_config.json.bak"
        ]
    },

    # Tests directory organization
    "tests": {
        "tests/data": [
            "test_preprocessing.py",
            "test_stack_processing.py"
        ],
        "tests/training": [
            "test_flan_ul2.py",
            "test_flan_setup.py",
            "test_deepspeed_config.py"
        ],
        "tests/utils": [
            "test_google_drive.py",
            "test_feature_extractor.py"
        ],
        "tests/evaluation": [
            "test_evaluation.sh"
        ],
        "tests/general": [
            "test_script.py"
        ]
    }
}

def ensure_subdirectories():
    """Create all required subdirectories."""
    for parent_dir, subdirs in FILE_MAPPINGS.items():
        for subdir in subdirs:
            os.makedirs(subdir, exist_ok=True)
            logger.info(f"Created/ensured subdirectory exists: {subdir}")

def move_file(src_path, dest_dir):
    """Move a file to the specified directory."""
    if not os.path.exists(src_path):
        logger.warning(f"Source file not found: {src_path}")
        return False
        
    os.makedirs(dest_dir, exist_ok=True)
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

def fix_python_imports():
    """Update import statements in Python files to reflect the new file organization."""
    # Get all Python files in the project
    python_files = []
    skip_dirs = ['.env', 'venv', '.git', '.vscode', '.cursor', '__pycache__']
    
    for root, dirs, files in os.walk("."):
        # Skip directories we want to exclude
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            if file.endswith(".py"):
                if not any(excluded in root for excluded in skip_dirs):
                    python_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(python_files)} Python files to check for import fixes")
    
    # Create a mapping of old import paths to new import paths
    import_mappings = {}
    
    # Build the import mapping dictionary based on file moves
    for parent_dir, subdirs in FILE_MAPPINGS.items():
        for target_dir, files in subdirs.items():
            for filename in files:
                if filename.endswith('.py'):
                    src_path = os.path.join(os.path.dirname(target_dir), filename)
                    dest_path = os.path.join(target_dir, filename)
                    
                    src_module = src_path.replace('/', '.').replace('.py', '')
                    dest_module = dest_path.replace('/', '.').replace('.py', '')
                    
                    import_mappings[src_module] = dest_module
    
    # Update imports in each file
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            modified = False
            
            # Check for each import pattern and update if needed
            for old_import, new_import in import_mappings.items():
                # Match for direct imports: import scripts.module
                pattern = r'import\s+' + old_import.replace('.', r'\.')
                replacement = 'import ' + new_import
                new_content = re.sub(pattern, replacement, content)
                if new_content != content:
                    content = new_content
                    modified = True
                
                # Match for from imports: from scripts.module import ...
                pattern = r'from\s+' + old_import.replace('.', r'\.') + r'\s+import'
                replacement = 'from ' + new_import + ' import'
                new_content = re.sub(pattern, replacement, content)
                if new_content != content:
                    content = new_content
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

def create_index_files():
    """Create index files (README.md, __init__.py) in subdirectories."""
    # Create __init__.py files for Python package directories
    for parent_dir, subdirs in FILE_MAPPINGS.items():
        if parent_dir.startswith('src') or parent_dir.startswith('tests') or parent_dir == 'scripts':
            for subdir in subdirs:
                init_file = os.path.join(subdir, "__init__.py")
                if not os.path.exists(init_file):
                    with open(init_file, 'w') as f:
                        module_name = os.path.basename(subdir)
                        f.write(f'"""Module for {module_name}."""\n')
                    logger.info(f"Created {init_file}")
    
    # Create README.md files for each subdirectory
    for parent_dir, subdirs in FILE_MAPPINGS.items():
        for subdir in subdirs:
            dir_name = os.path.basename(subdir)
            readme_file = os.path.join(subdir, "README.md")
            if not os.path.exists(readme_file):
                with open(readme_file, 'w') as f:
                    title = ' '.join(word.capitalize() for word in dir_name.split('_'))
                    f.write(f"# {title}\n\n")
                    f.write(f"This directory contains {dir_name.replace('_', ' ')} files for the Gen-Text-AI project.\n")
                logger.info(f"Created {readme_file}")

def organize_files(dry_run=False):
    """Organize files into their respective subdirectories."""
    if not dry_run:
        ensure_subdirectories()
    
    moved_files = {}
    
    for parent_dir, subdirs in FILE_MAPPINGS.items():
        for target_dir, files in subdirs.items():
            for file in files:
                src_path = os.path.join(os.path.dirname(target_dir), file)
                
                if dry_run:
                    logger.info(f"Would move {src_path} to {target_dir}")
                else:
                    if move_file(src_path, target_dir):
                        moved_files[src_path] = target_dir
    
    if not dry_run:
        create_index_files()
        fix_python_imports()
    
    return moved_files

def main():
    parser = argparse.ArgumentParser(description="Organize files into more specific subdirectories")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually move files, just print what would be done")
    
    args = parser.parse_args()
    
    logger.info("Starting subdirectory organization...")
    
    if args.dry_run:
        logger.info("Performing dry run - no files will be moved")
    
    moved_files = organize_files(dry_run=args.dry_run)
    
    if args.dry_run:
        logger.info(f"Would move {len(moved_files)} files")
    else:
        logger.info(f"Successfully moved {len(moved_files)} files")
        logger.info("Updated import statements to reflect new file structure")
    
    logger.info("Subdirectory organization complete")

if __name__ == "__main__":
    main() 