#!/bin/bash

# Paperspace Setup Script for Flan-UL2 Text Generation Training
# This script prepares the environment for training on Paperspace A6000 machines

echo "===== SETTING UP ENVIRONMENT FOR PAPERSPACE ====="

# Install required dependencies
pip install -q unsloth transformers datasets accelerate peft bitsandbytes zstandard langid nltk xformers

# Install optional dependencies for visualization and monitoring
pip install -q tensorboard wandb

# Download necessary NLTK resources
python -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    print('Successfully downloaded NLTK punkt')
except Exception as e:
    print(f'Could not download NLTK data: {e}')
"

# Create necessary directories
mkdir -p data/processed
mkdir -p logs
mkdir -p text_models/flan-ul2-fine-tuned

# Make scripts executable
chmod +x *.sh
chmod +x scripts/*.sh

# Set up HuggingFace token if provided
if [ -n "$HF_TOKEN" ]; then
    echo "HF_TOKEN environment variable found. Will use for Hugging Face Hub operations."
else
    echo "⚠️ Please set your HuggingFace token to access the model: export HF_TOKEN=your_token_here"
fi

# Apply fixes for dataset mapping if needed
if [ -f "fix_dataset_mapping.py" ]; then
    echo "Applying dataset mapping fixes..."
    python fix_dataset_mapping.py
fi

# Check if Unsloth is properly installed
python -c "
try:
    from unsloth import FastLanguageModel
    print('✅ Unsloth correctly installed and available!')
except (ImportError, NotImplementedError) as e:
    print(f'⚠️ Unsloth not available: {e}')
    print('Training will continue without Unsloth optimizations')
"

echo "===== ENVIRONMENT SETUP COMPLETE ====="
echo "You can now process datasets and train your model with:"
echo "1. python train_text_flan.py --process_only --data_dir data/processed"
echo "2. bash train_text_flan_a6000.sh"
echo "===== SETUP COMPLETE ====="
