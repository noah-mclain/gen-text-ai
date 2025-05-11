#!/bin/bash
# Optimized training script for FLAN-UL2 text generation fine-tuning

# Set environment variables for better performance
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
  echo "Warning: HF_TOKEN is not set. Some datasets might be inaccessible."
  echo "You should set it using: export HF_TOKEN=your_huggingface_token"
fi

# Make directories
mkdir -p logs
mkdir -p data/processed
mkdir -p models/flan-ul2-finetune

# Process datasets first
echo "Processing datasets for FLAN-UL2 text generation fine-tuning..."
python train_text_flan.py \
    --config config/training_config_text.json \
    --data_dir data/processed \
    --process_only \
    --debug \
    2>&1 | tee logs/process_datasets_$(date +%Y%m%d_%H%M%S).log

# Check if dataset processing was successful
if [ $? -ne 0 ]; then
    echo "Dataset processing failed. Check the logs for details."
    exit 1
fi

# Run the training with optimized settings
echo "Starting FLAN-UL2 fine-tuning for text generation..."
python train_text_flan.py \
    --config config/training_config_text.json \
    --data_dir data/processed \
    --use_drive \
    --drive_base_dir "FlanUL2Text" \
    2>&1 | tee logs/train_flan_ul2_$(date +%Y%m%d_%H%M%S).log

# Check if command succeeded
if [ $? -eq 0 ]; then
  echo "Training completed successfully!"
else
  echo "Training failed. Check the logs for details."
  exit 1
fi 