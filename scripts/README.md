# Scripts Directory

This directory contains utility scripts for the Gen-Text-AI project.

## Training Scripts

- `train_a6000_optimized.sh`: Optimized training script for A6000 GPUs with 48GB VRAM. Includes advanced optimizations for DeepSeek-Coder model with feature extraction.
- `train_deepseek_coder.sh`: Script specifically for training DeepSeek-Coder models.
- `train_local_test.sh`: Simplified training script for testing on local machines.
- `train_optimized.sh`: General optimized training script with memory efficiency improvements.
- `train_text_flan_a6000.sh`: Optimized training script for Flan-UL2 text models on A6000 GPUs.
- `train_text_flan.sh`: Training script for Flan-UL2 text models.
- `train_without_wandb.sh`: Training script that skips Weights & Biases logging.

## Utility Scripts

- `cleanup_stack_datasets.sh`: Cleans up all The Stack dataset files and directories to free up space.
- `cuda_env_setup.sh`: Sets up CUDA environment variables for optimal GPU usage.
- `fix_before_training.sh`: Fixes common issues that might occur before training starts.
- `fix_dataset_features.sh`: Fixes dataset feature inconsistencies to ensure compatibility with training.
- `fix_wandb.sh`: Fixes Weights & Biases integration issues.
- `fix_xformers_env.py`: Configures xformers and ensures proper setup for memory-efficient attention.
- `ensure_feature_extractor.py`: Verifies the feature extractor setup before training.
- `install_language_dependencies.sh`: Installs necessary language-specific dependencies.
- `paperspace_setup.sh`: Sets up the environment for Paperspace cloud instances.
- `setup_google_drive.py`: Configures Google Drive integration for storing and retrieving datasets.
- `sync_to_drive.py`: Syncs training results and datasets to Google Drive.
- `codebase_cleanup.py`: Organizes the codebase by moving files to appropriate directories.

## Dataset Processing Scripts

- `process_datasets.sh`: Processes raw datasets for training.
- `prepare_datasets_for_training.py`: Extracts features from preprocessed datasets before training.

## Environment Management Scripts

- `fix_deepspeed.py`: Fixes DeepSpeed configuration issues.
- `purge_deepspeed.py`: Completely removes DeepSpeed from the environment.
- `clean_deepspeed_env.py`: Cleans up DeepSpeed-related environment variables.
- `fix_environment.py`: Fixes common environment issues.
- `fix_cuda.py`: Resolves CUDA-related problems.

## Diagnostics

- `check_paperspace_env.py`: Checks if the code is running in a Paperspace environment.
- `diagnose_deepspeed.sh`: Diagnoses DeepSpeed configuration issues.

## Usage

Most scripts can be run directly from the command line. For example:

```bash
# Run optimized training on A6000 GPU
bash train_a6000_optimized.sh

# Fix dataset features before training
bash fix_dataset_features.sh

# Set up Google Drive integration
python setup_google_drive.py
```

For specific usage instructions, refer to the documentation at the beginning of each script.
