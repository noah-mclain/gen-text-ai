#!/bin/bash
# Optimized training script for DeepSeek-Coder fine-tuning

# Set environment variables for better performance
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
  echo "Warning: HF_TOKEN is not set. Some datasets might be inaccessible and you won't be able to push to Hugging Face Hub."
  echo "You should set it using: export HF_TOKEN=your_huggingface_token"
else
  echo "HF_TOKEN is set. Will use for authentication with Hugging Face Hub."
fi

# Make directories
mkdir -p logs
mkdir -p data/processed
mkdir -p models/deepseek-coder-finetune

# Set up Google Drive authentication first
echo "Setting up Google Drive authentication..."
python scripts/setup_google_drive.py
if [ $? -ne 0 ]; then
    echo "Google Drive authentication setup failed. Will proceed without Drive integration."
    USE_DRIVE_FLAG=""
else
    echo "Google Drive authentication successful. Will use Drive for storage."
    USE_DRIVE_FLAG="--use_drive"
fi

# Process datasets first (if needed)
if [ ! -d "data/processed/humaneval_processed" ] || [ ! -d "data/processed/mbpp_processed" ]; then
    echo "Processing datasets for DeepSeek-Coder fine-tuning..."
    python src/data/process_datasets.py \
        --config config/dataset_config.json \
        --output_dir data/processed \
        2>&1 | tee logs/process_datasets_$(date +%Y%m%d_%H%M%S).log
    
    # Check if dataset processing was successful
    if [ $? -ne 0 ]; then
        echo "Dataset processing failed. Check the logs for details."
        exit 1
    fi
else
    echo "Datasets already processed. Skipping dataset processing step."
fi

# Run the training with optimized settings
echo "Starting DeepSeek-Coder fine-tuning..."
python -m src.training.train \
    --config config/training_config.json \
    --data_dir data/processed \
    $USE_DRIVE_FLAG \
    --drive_base_dir "DeepseekCoder" \
    --push_to_hub \
    --hub_model_id "your-username/deepseek-coder-fine-tuned" \
    2>&1 | tee logs/train_deepseek_coder_$(date +%Y%m%d_%H%M%S).log

# Check if command succeeded
if [ $? -eq 0 ]; then
  echo "Training completed successfully!"
  
  # Sync results to Drive if available
  if [ -n "$USE_DRIVE_FLAG" ]; then
    echo "Syncing results to Google Drive..."
    python scripts/sync_to_drive.py --sync-all
  fi
  
  # Run evaluation
  echo "Running evaluation on trained model..."
  python -m src.evaluation.evaluate \
      --model_path models/deepseek-coder-finetune/final \
      --benchmark humaneval,mbpp \
      2>&1 | tee logs/eval_deepseek_coder_$(date +%Y%m%d_%H%M%S).log
  
  # Check if evaluation was successful
  if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully!"
  else
    echo "Evaluation failed. Check the logs for details."
  fi
else
  echo "Training failed. Check the logs for details."
  exit 1
fi 