#!/bin/bash
# Optimized commands for preprocessing and training with GPU optimizations

# Set bash to exit on error
set -e

# Step 1: Authenticate with Google Drive in headless mode
echo "==== Setting up Google Drive Authentication ===="
python scripts/google_drive_manager.py --action setup --headless

# Step 2: Create folder structure in Google Drive
echo "==== Creating folder structure in Google Drive ===="
python scripts/google_drive_manager.py --action create_folders --base_dir DeepseekCoder --headless

# Step 3: Preprocess all datasets EXCEPT the Stack (with memory optimizations)
echo "==== Preprocessing datasets with memory optimizations ===="
python main_api.py --mode process \
  --dataset_config config/dataset_config.json \
  --datasets codesearchnet_all code_alpaca mbpp humaneval codeparrot \
  --streaming --no_cache \
  --use_drive_api \
  --credentials_path credentials.json \
  --drive_base_dir DeepseekCoder \
  --headless

# Step 4: Train with ALL datasets (including the Stack) with GPU optimizations
echo "==== Training with all datasets including The Stack with GPU optimizations ===="
python main_api.py --mode quick-stack \
  --auto-time \
  --datasets codesearchnet_all code_alpaca mbpp humaneval the_stack_filtered \
  --use_drive_api \
  --credentials_path credentials.json \
  --drive_base_dir DeepseekCoder \
  --headless

echo "==== Pipeline completed! ====" 