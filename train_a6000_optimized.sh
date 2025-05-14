#!/bin/bash
# Fully optimized training script for A6000 GPU with 48GB VRAM using advanced optimizations
# This script is configured for maximum performance with multi-language support

# Set environment variables for better performance
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Explicitly unset all DeepSpeed-related variables to ensure clean environment
unset ACCELERATE_USE_DEEPSPEED
unset ACCELERATE_DEEPSPEED_CONFIG_FILE
unset ACCELERATE_DEEPSPEED_PLUGIN_TYPE
unset HF_DS_CONFIG
unset DEEPSPEED_CONFIG_FILE
unset DS_ACCELERATOR
unset DS_OFFLOAD_PARAM
unset DS_OFFLOAD_OPTIMIZER
unset TRANSFORMERS_ZeRO_2_FORCE_INVALIDATE_CHECKPOINT
unset DEEPSPEED_OVERRIDE_DISABLE

# Set environment variables for mixed precision training (instead of command-line args)
export ACCELERATE_MIXED_PRECISION="bf16"

# Remove any existing DeepSpeed config files to prevent conflicts
if [ -f "ds_config.json" ]; then
    echo "Removing existing DeepSpeed config file: ds_config.json"
    rm ds_config.json
fi
if [ -f "ds_config_a6000.json" ]; then
    echo "Removing existing DeepSpeed config file: ds_config_a6000.json"
    rm ds_config_a6000.json
fi

# Create function to calculate expected training time
calculate_training_time() {
    # Inputs to function
    local num_epochs=$1
    local num_datasets=$2
    local model_name=$(grep -o '"model_name_or_path": "[^"]*"' config/training_config.json | cut -d'"' -f4)
    local batch_size=$(grep -o '"per_device_train_batch_size": [0-9]*' config/training_config.json | grep -o '[0-9]*')
    local grad_accum=$(grep -o '"gradient_accumulation_steps": [0-9]*' config/training_config.json | grep -o '[0-9]*')

    # Default values if not found
    if [ -z "$batch_size" ]; then batch_size=1; fi
    if [ -z "$grad_accum" ]; then grad_accum=16; fi
    
    # Use Python instead of bc for calculations
    python -c "
import sys
# Inputs
num_epochs = $num_epochs
num_datasets = $num_datasets
batch_size = $batch_size
grad_accum = $grad_accum
model_name = '$model_name'

# Base time calculation - minutes per epoch per dataset
base_time_per_epoch = 30  # Base minutes per epoch for a single dataset

# Adjust for model size - larger models take longer
model_scale = 1.0
if 'deepseek-coder-6.7b' in model_name:
    model_scale = 1.5
elif 'deepseek-coder-33b' in model_name:
    model_scale = 4.0

# Adjust for effective batch size
batch_factor = 16 / (batch_size * grad_accum)
batch_factor = max(0.5, min(2.0, batch_factor))

# Calculate total hours
total_minutes = base_time_per_epoch * num_epochs * num_datasets * model_scale * batch_factor
total_hours = total_minutes / 60

# Add buffer time (15%)
total_hours = total_hours * 1.15

# Round up to nearest integer
total_hours = max(1, round(total_hours))

# Output the result
print(int(total_hours))
"
}

# Read number of epochs from training config
NUM_EPOCHS=$(grep -o '"num_train_epochs": [0-9]*' config/training_config.json | grep -o '[0-9]*')
if [ -z "$NUM_EPOCHS" ]; then
  NUM_EPOCHS=1  # Default to 1 if not found
fi
echo "Detected $NUM_EPOCHS epochs in training configuration"

# Count number of enabled datasets - force it to include all datasets
NUM_DATASETS=$(python -c "
import json, os
try:
    with open('config/dataset_config.json', 'r') as f:
        config = json.load(f)
    # Count all datasets in the config, regardless of enabled status
    print(len(config.keys()))
except:
    # In case of error, assume a reasonable number
    print(10)
")
echo "Training on approximately $NUM_DATASETS datasets for $NUM_EPOCHS epochs"

# Calculate estimated training time
MAX_HOURS=$(calculate_training_time $NUM_EPOCHS $NUM_DATASETS)
echo "Estimated training time: $MAX_HOURS hours"

# Calculate expected completion time using portable date command
START_TIME=$(date +%s)
END_TIME=$((START_TIME + MAX_HOURS * 3600))
# Use portable date format that works across systems
COMPLETION_TIME=$(python -c "import time; print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime($END_TIME)))")
echo "Expected completion time: $COMPLETION_TIME"

# Override MAX_HOURS if provided as a command-line argument
if [ "$1" != "" ] && [[ "$1" =~ ^[0-9]+$ ]]; then
  MAX_HOURS=$1
  echo "Using command-line specified training time: $MAX_HOURS hours"
fi

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
  echo "Warning: HF_TOKEN is not set. Some datasets might be inaccessible and you won't be able to push to Hugging Face Hub."
  echo "You should set it using: export HF_TOKEN=your_huggingface_token"
else
  echo "HF_TOKEN is set. Will use for authentication with Hugging Face Hub."
fi

# Fix Unsloth import order warning
export PYTHONPATH=$PYTHONPATH:.

# Make directories
mkdir -p logs
mkdir -p data/processed

# Set up Google Drive authentication first
echo "Setting up Google Drive authentication..."
python scripts/setup_google_drive.py
if [ $? -ne 0 ]; then
    echo "Google Drive authentication setup failed. Will proceed without Drive integration."
    USE_DRIVE_FLAG=""
    SKIP_DRIVE="--skip_drive"
else
    echo "Google Drive authentication successful. Will use Drive for storage."
    USE_DRIVE_FLAG="--use_drive --drive_base_dir DeepseekCoder"
    SKIP_DRIVE=""
fi

# Clean any cached files for better memory management
echo "Cleaning cache directories..."
rm -rf ~/.cache/huggingface/datasets/downloads/completed.lock

# Pre-download datasets with retry mechanism (fallback option)
echo "Pre-warming dataset cache for faster loading..."
python -c "from datasets import load_dataset; load_dataset('code-search-net/code_search_net', split='train', streaming=True, trust_remote_code=True)"
python -c "from datasets import load_dataset; load_dataset('sahil2801/CodeAlpaca-20k', split='train', streaming=True, trust_remote_code=True)"
python -c "from datasets import load_dataset; load_dataset('mbpp', split='train', streaming=True, trust_remote_code=True)"
python -c "from datasets import load_dataset; load_dataset('codeparrot/codeparrot-clean', split='train', streaming=True, trust_remote_code=True)"
python -c "from datasets import load_dataset; load_dataset('openai/openai_humaneval', split='test', streaming=True, trust_remote_code=True)"
python -c "from datasets import load_dataset; load_dataset('ise-uiuc/Magicoder-OSS-Instruct-75K', split='train', streaming=True, trust_remote_code=True)"

# ===== DATASET PREPARATION FLOW =====
# 1. Check Google Drive for preprocessed datasets
# 2. Download available preprocessed datasets
# 3. Process only datasets not found on Drive
# 4. Train on all datasets

# Check if datasets are already available on Google Drive
echo "===== CHECKING FOR PRE-PROCESSED DATASETS ====="
if [ "${SKIP_DRIVE:-False}" = "False" ]; then
    echo "Google Drive integration is ENABLED. Looking for pre-processed datasets on Drive..."
else
    echo "Google Drive integration is DISABLED (SKIP_DRIVE=$SKIP_DRIVE). Using local datasets only."
fi

# Make sure all datasets are enabled in the config
echo "===== ENABLING ALL DATASETS IN CONFIG ====="
python -c "
import json
import sys
try:
    with open('config/dataset_config.json', 'r') as f:
        config = json.load(f)
    
    # Enable all datasets
    for dataset_name in config:
        config[dataset_name]['enabled'] = True
        print(f'Enabled dataset: {dataset_name}')
    
    with open('config/dataset_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print('All datasets enabled in config')
except Exception as e:
    print(f'Error enabling datasets: {e}', file=sys.stderr)
"

# Get list of datasets that need processing vs ones available on Drive
echo "Running dataset availability check..."
DATASET_STATUS=$(python -c "
from src.utils.drive_dataset_checker import prepare_datasets
import json
import os
import sys
import glob

config_path = 'config/dataset_config.json'
skip_drive = ${SKIP_DRIVE:-False}

# This gets datasets available locally or on Drive, and those still needed
try:
    # Check what datasets are directly available on disk first
    processed_dir = 'data/processed'
    locally_available = []
    
    # Create a map between dataset processing names and config names
    dataset_name_mappings = {
        'code_alpaca': ['code_alpaca', 'alpaca'],
        'codeparrot': ['codeparrot'],
        'codesearchnet_all_go': ['codesearchnet_go'],
        'codesearchnet_go': ['codesearchnet_go'],
        'codesearchnet_all_java': ['codesearchnet_java'],
        'codesearchnet_java': ['codesearchnet_java'],
        'codesearchnet_all_javascript': ['codesearchnet_javascript'],
        'codesearchnet_javascript': ['codesearchnet_javascript'],
        'codesearchnet_all_php': ['codesearchnet_php'],
        'codesearchnet_php': ['codesearchnet_php'],
        'codesearchnet_all_python': ['codesearchnet_python'],
        'codesearchnet_python': ['codesearchnet_python'],
        'codesearchnet_all_ruby': ['codesearchnet_ruby'],
        'codesearchnet_ruby': ['codesearchnet_ruby'],
        'humaneval': ['humaneval'],
        'instruct_code': ['instruct_code'],
        'mbpp': ['mbpp']
    }
    
    # Find all processed dataset directories that exist
    all_processed_dirs = glob.glob(os.path.join(processed_dir, '*_processed*'))
    all_processed_dirs += glob.glob(os.path.join(processed_dir, '*_interim_*'))
    all_processed_dirs = [os.path.basename(d) for d in all_processed_dirs if os.path.isdir(os.path.join(processed_dir, d))]
    
    # Check for datasets from the mapping
    for processed_prefix, config_names in dataset_name_mappings.items():
        matching_dirs = [d for d in all_processed_dirs if d.startswith(processed_prefix)]
        if matching_dirs:
            # Add all associated config names as available
            locally_available.extend(config_names)
    
    print(f'Found locally available datasets: {locally_available}', file=sys.stderr)
    
    # Now run the normal prepare_datasets function with the predetected local datasets
    available, needed, download_time = prepare_datasets(
        config_path, 
        output_dir='data/processed',
        skip_drive=skip_drive,
        predetected_local=locally_available
    )

except Exception as e:
    print(f'Error checking local datasets: {e}', file=sys.stderr)
    # Fallback to standard prepare_datasets
    available, needed, download_time = prepare_datasets(
        config_path, 
        output_dir='data/processed',
        skip_drive=skip_drive
    )

print('AVAILABLE=' + ','.join(available), file=sys.stdout)
print('NEEDED=' + ','.join(needed), file=sys.stdout)
print('DOWNLOAD_TIME=' + str(download_time), file=sys.stdout)
")

# Parse results
AVAILABLE_DATASETS=$(echo "$DATASET_STATUS" | grep "AVAILABLE=" | cut -d'=' -f2)
DATASETS_TO_PROCESS=$(echo "$DATASET_STATUS" | grep "NEEDED=" | cut -d'=' -f2)
DOWNLOAD_TIME=$(echo "$DATASET_STATUS" | grep "DOWNLOAD_TIME=" | cut -d'=' -f2)

# Show status
echo "===== DATASET STATUS ====="
echo "Already available datasets: $AVAILABLE_DATASETS"
echo "Datasets that need processing: $DATASETS_TO_PROCESS"
echo "Download time from Drive: $DOWNLOAD_TIME seconds"

# Only process datasets if there are any that need processing
if [ -n "$DATASETS_TO_PROCESS" ]; then
    echo "===== PROCESSING DATASETS ====="
    echo "Processing datasets: $DATASETS_TO_PROCESS"
    
    python main_api.py \
        --mode process \
        --datasets $DATASETS_TO_PROCESS \
        --streaming \
        --no_cache \
        --dataset_config config/dataset_config.json \
        $USE_DRIVE_FLAG \
        2>&1 | tee logs/dataset_processing_$(date +%Y%m%d_%H%M%S).log
        
    if [ $? -ne 0 ]; then
        echo "⚠️ Dataset processing encountered errors. Training may use incomplete data."
    else
        echo "✅ Dataset processing completed successfully."
    fi
else
    echo "✅ All datasets already available. Skipping dataset processing step."
fi

# Update config to ensure no DeepSpeed and use proper optimizations
echo "===== UPDATING CONFIG FOR OPTIMIZED TRAINING ====="
python -c "
import json
import sys
import os
try:
    with open('config/training_config.json', 'r') as f:
        config = json.load(f)
    
    if 'training' not in config:
        config['training'] = {}
    
    # Thoroughly remove all DeepSpeed references
    deepspeed_keys = ['use_deepspeed', 'deepspeed_config', 'deepspeed']
    for key in deepspeed_keys:
        if key in config['training']:
            del config['training'][key]
            print(f'Removed {key} from config')
    
    # Fix mixed precision settings - choose ONLY ONE precision mode
    # BF16 is preferred for A6000 GPUs
    config['training']['bf16'] = True  # Use bfloat16 for better performance
    config['training']['fp16'] = False  # Disable fp16 completely
    
    # Make sure torch_dtype is set correctly if it exists
    if 'torch_dtype' in config['training']:
        config['training']['torch_dtype'] = 'bfloat16'
        print('Set torch_dtype to bfloat16')
    
    # Enable mixed precision in the config directly
    config['training']['mixed_precision'] = 'bf16'
    print('Set mixed_precision to bf16 in the config')
    
    # Make sure gradient checkpointing is enabled for memory efficiency
    config['training']['gradient_checkpointing'] = True  # Enable memory optimization
    
    # Also check for and disable deepspeed in accelerate config if it exists
    accelerate_config_paths = [
        './.accelerate/default_config.yaml',
        os.path.expanduser('~/.cache/huggingface/accelerate/default_config.yaml'),
        os.path.expanduser('~/.config/huggingface/accelerate/default_config.yaml')
    ]
    
    for config_path in accelerate_config_paths:
        if os.path.exists(config_path):
            try:
                import yaml
                with open(config_path, 'r') as f:
                    accel_config = yaml.safe_load(f)
                
                # Disable DeepSpeed in accelerate config
                if 'deepspeed_plugin' in accel_config:
                    del accel_config['deepspeed_plugin']
                    print(f'Removed deepspeed_plugin from {config_path}')
                
                if 'deepspeed' in accel_config:
                    del accel_config['deepspeed']
                    print(f'Removed deepspeed from {config_path}')
                
                # Set mixed precision in accelerate config
                accel_config['mixed_precision'] = 'bf16'
                
                with open(config_path, 'w') as f:
                    yaml.safe_dump(accel_config, f)
                    
                print(f'Updated accelerate config at {config_path}')
            except Exception as e:
                print(f'Error updating accelerate config at {config_path}: {e}', file=sys.stderr)
    
    # Make sure Unsloth is configured in the config rather than command line
    # This avoids command line argument errors
    model_name = config.get('training', {}).get('model_name_or_path', '').lower()
    is_causal_lm = not any(name in model_name for name in ['t5', 'ul2', 'flan'])
    
    if is_causal_lm:
        config['training']['use_unsloth'] = True
        print('Configured for causal LM with Unsloth optimizations')
    else:
        # For sequence-to-sequence models
        config['training']['use_unsloth'] = False
        print('Configured for sequence-to-sequence model (Unsloth not compatible)')
    
    # Update datasets weights to use all available datasets
    if 'dataset_weights' not in config:
        config['dataset_weights'] = {}
    
    # Get all available datasets from the dataset config
    try:
        with open('config/dataset_config.json', 'r') as f:
            dataset_config = json.load(f)
        
        # Add all datasets with equal weights
        for dataset_name in dataset_config:
            if dataset_name not in config['dataset_weights']:
                config['dataset_weights'][dataset_name] = 1.0
                print(f'Added dataset {dataset_name} to training with weight 1.0')
    except Exception as e:
        print(f'Error loading dataset_config.json: {e}', file=sys.stderr)
    
    # Add dataset paths configuration
    if 'dataset_paths' not in config:
        config['dataset_paths'] = {}
    
    # Update dataset paths to include more paths with naming variations
    import glob
    import os
    
    processed_dir = 'data/processed'
    dataset_paths = {}
    
    # Add mapping for each dataset type we want to support
    dataset_mapping = {
        'code_alpaca': ['code_alpaca'],
        'codeparrot': ['codeparrot'],
        'codesearchnet_go': ['codesearchnet_all_go', 'codesearchnet_go'],
        'codesearchnet_java': ['codesearchnet_all_java', 'codesearchnet_java'],
        'codesearchnet_javascript': ['codesearchnet_all_javascript', 'codesearchnet_javascript'],
        'codesearchnet_php': ['codesearchnet_all_php', 'codesearchnet_php'],
        'codesearchnet_python': ['codesearchnet_all_python', 'codesearchnet_python'],
        'codesearchnet_ruby': ['codesearchnet_all_ruby', 'codesearchnet_ruby'],
        'humaneval': ['humaneval'],
        'instruct_code': ['instruct_code'],
        'mbpp': ['mbpp']
    }
    
    # Find all processed dataset directories
    all_processed_dirs = []
    if os.path.exists(processed_dir):
        all_processed_dirs = os.listdir(processed_dir)
    processed_dirs = [d for d in all_processed_dirs if os.path.isdir(os.path.join(processed_dir, d)) and ('_processed' in d or '_interim_' in d)]
    
    # For each config dataset name, find matching processed directory
    for config_name, search_prefixes in dataset_mapping.items():
        if not isinstance(search_prefixes, list):
            search_prefixes = [search_prefixes]
            
        # Look for matching directories
        for prefix in search_prefixes:
            matches = [d for d in processed_dirs if d.startswith(prefix)]
            if matches:
                # Sort to prioritize final over interim when multiple exist
                matches.sort(key=lambda x: 0 if 'final' in x else 1)
                dataset_paths[config_name] = os.path.join(processed_dir, matches[0])
                print(f'Mapped dataset {config_name} to {matches[0]}')
                break
    
    # Update the config
    config['dataset_paths'] = dataset_paths
    
    with open('config/training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print('Config file updated successfully for optimized training with all datasets')
except Exception as e:
    print(f'Error updating config: {e}', file=sys.stderr)
"

# Create accelerate config if it doesn't exist
echo "===== CREATING ACCELERATE CONFIG ====="
mkdir -p ~/.cache/huggingface/accelerate/
cat > ~/.cache/huggingface/accelerate/default_config.yaml << EOF
compute_environment: LOCAL_MACHINE
distributed_type: NO
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
use_cpu: false
EOF
echo "Created accelerate config with bf16 precision"

echo "===== STARTING OPTIMIZED TRAINING ====="
TRAINING_START_TIME=$(date +%s)

# Train with direct module call without the problematic arguments
# Note: Removed --mixed_precision bf16 argument that was causing the error
echo "Starting training with optimizations..."
python -m src.training.train \
    --config config/training_config.json \
    --data_dir data/processed \
    $USE_DRIVE_FLAG \
    --push_to_hub \
    --no_deepspeed \
    --estimate_time \
    2>&1 | tee logs/train_a6000_optimized_$(date +%Y%m%d_%H%M%S).log

# Check exit status
EXIT_STATUS=$?

# Calculate actual training time
TRAINING_END_TIME=$(date +%s)
ACTUAL_TRAINING_TIME=$((TRAINING_END_TIME - TRAINING_START_TIME))
# Use Python instead of bc for floating point division
ACTUAL_HOURS=$(python -c "print('{:.2f}'.format($ACTUAL_TRAINING_TIME / 3600))")

# Report completion
if [ $EXIT_STATUS -eq 0 ]; then
  echo "Training completed successfully!"
  echo "Actual training time: $ACTUAL_HOURS hours (estimated: $MAX_HOURS hours)"
  
  # Create completion marker file with timestamp for tracking
  echo "Training completed at $(date)" > logs/training_complete_$(date +%Y%m%d_%H%M%S).txt
  echo "Actual training time: $ACTUAL_HOURS hours" >> logs/training_complete_$(date +%Y%m%d_%H%M%S).txt
  echo "Estimated training time: $MAX_HOURS hours" >> logs/training_complete_$(date +%Y%m%d_%H%M%S).txt
  
  # Sync results to Drive if available
  if [ -n "$USE_DRIVE_FLAG" ]; then
    echo "Syncing results to Google Drive..."
    python scripts/sync_to_drive.py --sync-all
  fi
else
  echo "Training failed with exit code $EXIT_STATUS. Check the logs for details."
  echo "Training ran for $ACTUAL_HOURS hours before failing (estimated: $MAX_HOURS hours)"
fi

# Calculate training statistics if log file exists
LOG_FILE=$(find logs -name "train_a6000_optimized_*.log" -type f -exec ls -t {} \; | head -1)
if [ -f "$LOG_FILE" ]; then
  echo "Analyzing training log for statistics..."
  echo "Training duration: $(grep -o "Training runtime.*" $LOG_FILE | tail -1)"
  echo "Samples processed: $(grep -o "trained on [0-9]* samples" $LOG_FILE | tail -1)"
  echo "Final loss: $(grep -o "loss=.*" $LOG_FILE | tail -1)"
  
  # Calculate samples per second using Python instead of bc
  SAMPLES=$(grep -o "trained on [0-9]* samples" $LOG_FILE | tail -1 | grep -o "[0-9]*")
  if [ -n "$SAMPLES" ] && [ "$ACTUAL_TRAINING_TIME" -gt 0 ]; then
    SAMPLES_PER_SEC=$(python -c "print('{:.2f}'.format($SAMPLES / $ACTUAL_TRAINING_TIME))")
    echo "Processing speed: $SAMPLES_PER_SEC samples/second"
  fi
  
  # Add training speed to completion file
  echo "Processing speed: $SAMPLES_PER_SEC samples/second" >> logs/training_complete_$(date +%Y%m%d_%H%M%S).txt
fi

echo "Training process complete at $(date)" 