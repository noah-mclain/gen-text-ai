#!/bin/bash
# Optimized training script for A6000 GPU with 48GB VRAM

# Set environment variables for better performance
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
  echo "Warning: HF_TOKEN is not set. Some datasets might be inaccessible."
  echo "You should set it using: export HF_TOKEN=your_huggingface_token"
fi

# Make script directory
mkdir -p logs

# Run main command with optimized settings
echo "Starting optimized training with all datasets..."
python main_api.py \
    --mode quick-stack \
    --datasets the_stack_filtered codesearchnet_all code_alpaca humaneval mbpp codeparrot instruct_code \
    --streaming \
    --no_cache \
    --auto-time \
    --use_drive_api \
    --credentials_path credentials.json \
    --headless \
    2>&1 | tee logs/train_optimized_$(date +%Y%m%d_%H%M%S).log

# Check if command succeeded
if [ $? -eq 0 ]; then
  echo "Training completed successfully!"
else
  echo "Training failed. Check the logs for details."
fi 