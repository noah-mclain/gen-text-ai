#!/bin/bash
# Optimized training script for FLAN-UL2 text generation fine-tuning on A6000 GPUs

# Set environment variables for better performance
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true

# Enable autotuning for cuDNN
export CUDNN_FRONTEND_ENABLE_HEURISTIC_MODE=1

# Set PyTorch settings for better performance
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export NCCL_P2P_DISABLE=1

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
    2>&1 | tee logs/process_datasets_a6000_$(date +%Y%m%d_%H%M%S).log

# Check if dataset processing was successful
if [ $? -ne 0 ]; then
    echo "Dataset processing failed. Check the logs for details."
    exit 1
fi

# Generate a temporary deepspeed config with A6000 optimized settings
cat > ds_config_a6000_tmp.json << EOL
{
    "zero_stage": 3,
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6,
        "sub_group_size": 1e9,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "gather_16bit_weights_on_model_save": true
    },
    "gradient_accumulation_steps": 16,
    "gradient_clipping": 1.0,
    "steps_per_print": 10,
    "train_micro_batch_size_per_gpu": 2,
    "wall_clock_breakdown": false
}
EOL

# Run the training with optimized settings for A6000 GPU
echo "Starting FLAN-UL2 fine-tuning for text generation on A6000..."
python train_text_flan.py \
    --config config/training_config_text.json \
    --data_dir data/processed \
    --use_drive \
    --drive_base_dir "FlanUL2Text" \
    2>&1 | tee logs/train_flan_ul2_a6000_$(date +%Y%m%d_%H%M%S).log

# Check if command succeeded
if [ $? -eq 0 ]; then
  echo "Training completed successfully!"
  # Clean up temporary file
  rm ds_config_a6000_tmp.json
else
  echo "Training failed. Check the logs for details."
  # Clean up temporary file
  rm ds_config_a6000_tmp.json
  exit 1
fi 