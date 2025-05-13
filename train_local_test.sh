#!/bin/bash
# Simplified training script for testing on local machine
# This script disables GPU-specific optimizations

# Set environment variables
export CUDA_VISIBLE_DEVICES=""  # Disable CUDA
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=true

# Ensure the DeepSpeed config can be used on CPU
cp ds_config_a6000.json ds_config_cpu.json

# Disable GPU-specific DeepSpeed settings
cat > ds_config_cpu.json << EOL
{
  "fp16": {
    "enabled": false
  },
  "bf16": {
    "enabled": false
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "allgather_bucket_size": 5e8
  },
  "gradient_accumulation_steps": 4,
  "gradient_clipping": 1.0,
  "steps_per_print": 10,
  "train_batch_size": 2,
  "train_micro_batch_size_per_gpu": 1,
  "wall_clock_breakdown": false
}
EOL

# Set DeepSpeed environment variables for CPU
export DS_ACCELERATOR="cpu"
export DS_OFFLOAD_PARAM=cpu
export DS_OFFLOAD_OPTIMIZER=cpu
export ACCELERATE_USE_DEEPSPEED=true
export ACCELERATE_DEEPSPEED_CONFIG_FILE=ds_config_cpu.json

# Make directories
mkdir -p logs
mkdir -p data/processed

# Skip Google Drive authentication for local testing
SKIP_DRIVE="--skip_drive"
echo "Skipping Google Drive authentication for local testing"

# Process sample datasets for testing
echo "Processing minimal datasets for testing..."
python main_api.py \
    --mode process \
    --datasets mbpp \
    --max_samples 20 \
    --streaming \
    --no_cache \
    --dataset_config config/dataset_config.json \
    2>&1 | tee logs/dataset_processing_$(date +%Y%m%d_%H%M%S).log

# Train with direct module call (with CPU-compatible settings)
echo "Starting training with direct module call..."
python -m src.training.train \
    --config config/training_config_cpu.json \
    --data_dir data/processed \
    --debug \
    2>&1 | tee logs/train_local_test_$(date +%Y%m%d_%H%M%S).log

# Check exit status
EXIT_STATUS=$?

# Report completion
if [ $EXIT_STATUS -eq 0 ]; then
  echo "Training completed successfully!"
else
  echo "Training failed with exit code $EXIT_STATUS. Check the logs for details."
fi 