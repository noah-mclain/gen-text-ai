#!/bin/bash

# Script to run the deepseek-coder fine-tuning pipeline on Paperspace
# Uses the Google Drive API instead of mounting (which is not supported on Paperspace)
# Also applies memory efficiency optimizations

# Set environment variables for Hugging Face token (replace with your token)
# export HF_TOKEN=your_token_here

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib

# Ensure credentials file exists
if [ ! -f "credentials.json" ]; then
    echo "ERROR: credentials.json file not found."
    echo "Please download your credentials.json file from the Google Cloud Console:"
    echo "1. Go to https://console.cloud.google.com/"
    echo "2. Create a new project if you don't have one"
    echo "3. Enable the Google Drive API"
    echo "4. Create OAuth 2.0 credentials"
    echo "5. Download the credentials JSON file and rename it to credentials.json"
    exit 1
fi

# Create necessary directories
mkdir -p data/raw data/processed models results visualizations logs

# Set dataset and model parameters
DATASETS="code_alpaca mbpp humaneval"
BASE_MODEL="deepseek-ai/deepseek-coder-6.7b-base"
DRIVE_BASE_DIR="DeepseekCoder"

# Process datasets with memory efficiency
echo "Processing datasets with memory efficiency and Google Drive API..."
python main_api.py \
    --mode process \
    --dataset_config config/dataset_config.json \
    --datasets $DATASETS \
    --streaming \
    --no_cache \
    --use_drive_api \
    --credentials_path credentials.json \
    --drive_base_dir $DRIVE_BASE_DIR

# Train the model
echo "Training model with Google Drive API integration..."
python main_api.py \
    --mode train \
    --training_config config/training_config.json \
    --use_drive_api \
    --credentials_path credentials.json \
    --drive_base_dir $DRIVE_BASE_DIR

# Evaluate the model
echo "Evaluating model..."
python main_api.py \
    --mode evaluate \
    --model_path models/deepseek-coder-finetune \
    --base_model $BASE_MODEL \
    --use_drive_api \
    --credentials_path credentials.json \
    --drive_base_dir $DRIVE_BASE_DIR

# Visualize results
echo "Generating visualizations..."
python main_api.py \
    --mode visualize \
    --use_drive_api \
    --credentials_path credentials.json \
    --drive_base_dir $DRIVE_BASE_DIR

echo "Pipeline completed! All data has been saved to Google Drive under $DRIVE_BASE_DIR."
echo "You can access your model, results and visualizations there." 