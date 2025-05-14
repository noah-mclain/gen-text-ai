#!/bin/bash
# Script to run training with Weights & Biases enabled

echo "===== RUNNING TRAINING WITH WANDB ====="

# Set your wandb API key
export WANDB_API_KEY="636def25145822bffada9359d3c3b3ace65380a4"

# Remove the wandb disabled flag
unset WANDB_DISABLED

# Run the training script with wandb enabled
python -m src.training.train \
    --config config/training_config.json \
    --data_dir data/processed \
    --no_deepspeed \
    --estimate_time

echo "===== TRAINING COMPLETE =====" 