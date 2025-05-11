#!/bin/bash
# Fully optimized training script for A6000 GPU with 48GB VRAM
# This script is configured for maximum performance with multi-language support

# Set environment variables for better performance
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Set DeepSpeed environment variables
export DS_ACCELERATOR=cuda
export DS_OFFLOAD_PARAM=cpu
export DS_OFFLOAD_OPTIMIZER=cpu
export ACCELERATE_USE_DEEPSPEED=true
export ACCELERATE_DEEPSPEED_CONFIG_FILE=ds_config_a6000.json

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
  echo "Warning: HF_TOKEN is not set. Some datasets might be inaccessible."
  echo "You should set it using: export HF_TOKEN=your_huggingface_token"
fi

# Make directories
mkdir -p logs
mkdir -p data/processed

# Clean any cached files for better memory management
echo "Cleaning cache directories..."
rm -rf ~/.cache/huggingface/datasets/downloads/completed.lock
rm -rf ~/.cache/huggingface/transformers/models--deepseek-ai--deepseek-coder-6.7b-base/snapshots/*/*.safetensors.tmp*

# Set max training time based on system time (default to 10 hours if not specified)
if [ -z "$MAX_HOURS" ]; then
  # Calculate hours until midnight
  CURRENT_HOUR=$(date +%H)
  MAX_HOURS=$((24 - CURRENT_HOUR - 1))
  
  # Ensure we have at least 2 hours of training
  if [ $MAX_HOURS -lt 2 ]; then
    MAX_HOURS=10
  fi
  
  echo "Auto-calculated training time: $MAX_HOURS hours"
else
  echo "Using specified training time: $MAX_HOURS hours"
fi

# Pre-download datasets with retry mechanism
echo "Pre-downloading datasets to ensure availability..."
python -c "from datasets import load_dataset; load_dataset('code-search-net/code_search_net', split='train', streaming=True, trust_remote_code=True)"
python -c "from datasets import load_dataset; load_dataset('sahil2801/CodeAlpaca-20k', split='train', streaming=True, trust_remote_code=True)"
python -c "from datasets import load_dataset; load_dataset('mbpp', split='train', streaming=True, trust_remote_code=True)"
python -c "from datasets import load_dataset; load_dataset('codeparrot/codeparrot-clean', split='train', streaming=True, trust_remote_code=True)"
python -c "from datasets import load_dataset; load_dataset('openai/openai_humaneval', streaming=True, trust_remote_code=True)"
python -c "from datasets import load_dataset; load_dataset('ise-uiuc/Magicoder-OSS-Instruct-75K', split='train', streaming=True, trust_remote_code=True)"

# Run the optimized training command
echo "Starting optimized training with all datasets..."
python -m torch.distributed.run --nproc_per_node=1 main_api.py \
    --mode quick-stack \
    --datasets the_stack_filtered codesearchnet_all code_alpaca humaneval mbpp codeparrot instruct_code \
    --streaming \
    --no_cache \
    --auto-time \
    --use_drive_api \
    --credentials_path credentials.json \
    --headless \
    --dataset_config config/dataset_config.json \
    --training_config config/training_config.json \
    2>&1 | tee logs/train_a6000_optimized_$(date +%Y%m%d_%H%M%S).log

# Check exit status
EXIT_STATUS=$?

# Report completion
if [ $EXIT_STATUS -eq 0 ]; then
  echo "Training completed successfully!"
  
  # Create completion marker file with timestamp for tracking
  echo "Training completed at $(date)" > logs/training_complete_$(date +%Y%m%d_%H%M%S).txt
else
  echo "Training failed with exit code $EXIT_STATUS. Check the logs for details."
fi

# Calculate training statistics if log file exists
LOG_FILE=$(find logs -name "train_a6000_optimized_*.log" -type f -exec ls -t {} \; | head -1)
if [ -f "$LOG_FILE" ]; then
  echo "Analyzing training log for statistics..."
  echo "Training duration: $(grep -o "Training runtime.*" $LOG_FILE | tail -1)"
  echo "Samples processed: $(grep -o "trained on [0-9]* samples" $LOG_FILE | tail -1)"
  echo "Final loss: $(grep -o "loss=.*" $LOG_FILE | tail -1)"
fi

echo "Training process complete at $(date)" 