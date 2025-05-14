#!/bin/bash
# Optimized training script for text generation with FLAN-UL2 on A6000 GPU
# This script includes Google Drive integration for cloud storage of datasets and models

# Set bash to exit on error
set -e

# Parse arguments
DRIVE_BASE_DIR="FlanUL2Text"
USE_DRIVE=true
SKIP_LOCAL=false
BATCH_SIZE=1
GRAD_ACCUMULATION=16
LR=1e-5
MAX_STEPS=10000
SAVE_STEPS=1000
WARMUP_STEPS=100

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --no-drive)
      USE_DRIVE=false
      shift
      ;;
    --delete-local)
      SKIP_LOCAL=true
      shift
      ;;
    --drive-dir)
      DRIVE_BASE_DIR="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --grad-accum)
      GRAD_ACCUMULATION="$2"
      shift 2
      ;;
    --lr)
      LR="$2"
      shift 2
      ;;
    --max-steps)
      MAX_STEPS="$2"
      shift 2
      ;;
    --save-steps)
      SAVE_STEPS="$2"
      shift 2
      ;;
    --warmup-steps)
      WARMUP_STEPS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Function to calculate expected training time for seq2seq models
calculate_training_time_seq2seq() {
    # Input parameters
    local max_steps=$1
    local batch_size=$2
    local grad_accum=$3
    local dataset_count=$4  # Number of text datasets being used
    
    # Use Python for calculations instead of bc
    python -c "
import sys
# Inputs
max_steps = $max_steps
batch_size = $batch_size
grad_accum = $grad_accum
dataset_count = $dataset_count

# Get model info from config - will be read within Python
import os, json
try:
    with open('config/training_config_text.json', 'r') as f:
        config = json.load(f)
    model_name = config.get('training', {}).get('model_name_or_path', 'flan-ul2')
except:
    model_name = 'flan-ul2'  # Default if can't read config

# Base time calculation
# For seq2seq models like FLAN-UL2, we calculate time differently than causal LMs
# Average time per step in seconds for FLAN-UL2 on A6000 with batch size 1
base_time_per_step = 1.2

# Adjust for model size factor
model_size_factor = 1.0
if 'flan-t5-xl' in model_name:
    model_size_factor = 0.7  # Smaller than UL2
elif 'flan-ul2' in model_name:
    model_size_factor = 1.0  # Base reference
elif 'flan-t5-xxl' in model_name:
    model_size_factor = 1.5  # Larger model

# Adjust for effective batch size
batch_factor = (16 / (batch_size * grad_accum)) ** 0.5  # sqrt
batch_factor = max(0.5, min(2.0, batch_factor))

# Dataset complexity factor - more datasets mean more varied data
dataset_factor = 1 + ((dataset_count - 1) * 0.1)
dataset_factor = min(1.5, dataset_factor)

# Calculate total training time in seconds
total_seconds = base_time_per_step * max_steps * model_size_factor * batch_factor * dataset_factor

# Convert to hours
total_hours = total_seconds / 3600

# Add 15% buffer for safety
total_hours = total_hours * 1.15

# Round up to nearest integer
total_hours = max(1, round(total_hours))

# Output result
print(int(total_hours))
"
}

# Get number of datasets from config
echo "===== COUNTING TEXT DATASETS ====="
TEXT_DATASET_COUNT=$(python -c "
import json
import sys
try:
    with open('config/dataset_config_text.json', 'r') as f:
        config = json.load(f)
        # Count only enabled datasets
        enabled_count = sum(1 for _, info in config.items() if info.get('enabled', True))
        print(enabled_count)
except Exception as e:
    print('3')  # Default count if file read fails
    sys.stderr.write(f'Error reading dataset config: {e}\\n')
")

# Calculate expected training time
ESTIMATED_HOURS=$(calculate_training_time_seq2seq $MAX_STEPS $BATCH_SIZE $GRAD_ACCUMULATION $TEXT_DATASET_COUNT)
echo "===== TRAINING TIME ESTIMATE ====="
echo "Model parameters:"
echo "- Max steps: $MAX_STEPS"
echo "- Batch size: $BATCH_SIZE"
echo "- Gradient accumulation: $GRAD_ACCUMULATION"
echo "- Effective batch size: $((BATCH_SIZE * GRAD_ACCUMULATION))"
echo "- Number of datasets: $TEXT_DATASET_COUNT"
echo "- Learning rate: $LR"
echo "Estimated training time: $ESTIMATED_HOURS hours"

# Calculate expected completion time using portable date command
START_TIME=$(date +%s)
END_TIME=$((START_TIME + ESTIMATED_HOURS * 3600))
# Use portable date format that works across systems
COMPLETION_TIME=$(python -c "import time; print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime($END_TIME)))")
echo "Expected completion time: $COMPLETION_TIME"

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
  echo "Warning: HF_TOKEN is not set. Some datasets might be inaccessible and you won't be able to push to Hugging Face Hub."
  echo "You should set it using: export HF_TOKEN=your_huggingface_token"
else
  echo "HF_TOKEN is set. Will use for authentication with Hugging Face Hub."
fi

# Check Paperspace environment
echo "===== CHECKING PAPERSPACE ENVIRONMENT ====="
python scripts/check_paperspace_env.py

# Set CUDA visible devices to control which GPU is used
export CUDA_VISIBLE_DEVICES=0

# Set environment variables for better performance
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Make directories
mkdir -p logs
mkdir -p data/processed
mkdir -p text_models/flan-ul2-fine-tuned

# Set up Google Drive authentication first (moved earlier in the script)
echo "===== SETTING UP GOOGLE DRIVE INTEGRATION ====="
# Set text model-specific drive base directory
export DRIVE_BASE_DIR="${DRIVE_BASE_DIR:-FlanUL2Text}"
echo "Using Google Drive base directory: $DRIVE_BASE_DIR for cloud storage"

# Force Google Drive integration to be enabled unless specifically disabled
if [ "${USE_DRIVE}" = "false" ]; then
  echo "Google Drive integration was explicitly disabled with --no-drive flag."
else
  # Default to enabled
  USE_DRIVE="true"
  echo "Google Drive integration is ENABLED by default."
fi

# Authenticate with Google Drive if enabled
if [ "${USE_DRIVE}" = "true" ]; then
  python scripts/setup_google_drive.py --base_dir "$DRIVE_BASE_DIR"
  if [ $? -ne 0 ]; then
    echo "⚠️ Google Drive authentication failed. Will try again with default settings."
    python scripts/setup_google_drive.py  # Try without arguments
    if [ $? -ne 0 ]; then
      echo "❌ Google Drive authentication failed again. Proceeding without Drive integration."
      USE_DRIVE="false"
      DRIVE_OPTS=""
    else
      echo "✅ Google Drive authentication successful with default settings."
      DRIVE_OPTS="--use_drive --drive_base_dir $DRIVE_BASE_DIR"
      if [ "$SKIP_LOCAL" = "true" ]; then
        DRIVE_OPTS="$DRIVE_OPTS --skip_local_storage"
      fi
    fi
  else
    echo "✅ Google Drive authentication successful."
    DRIVE_OPTS="--use_drive --drive_base_dir $DRIVE_BASE_DIR"
    if [ "$SKIP_LOCAL" = "true" ]; then
      DRIVE_OPTS="$DRIVE_OPTS --skip_local_storage"
    fi
    
    # Validate authentication by testing if we can access the drive
    python -c "
import sys
from src.utils.google_drive_manager import test_drive_mounting
if not test_drive_mounting():
    print('Failed to access Google Drive after authentication')
    sys.exit(1)
else:
    print('Successfully validated Google Drive access')
" 
    
    # If validation failed, disable Drive
    if [ $? -ne 0 ]; then
      echo "❌ Google Drive access validation failed. Proceeding without Drive integration."
      USE_DRIVE="false"
      DRIVE_OPTS=""
    fi
  fi
else
  USE_DRIVE="false"
  DRIVE_OPTS=""
  echo "Google Drive integration disabled by user request"
fi

# Check if models are compatible with Unsloth
echo "===== CHECKING MODEL ARCHITECTURE FOR OPTIMIZATIONS ====="
# Extract model name from config
MODEL_NAME=$(python -c "
import json
try:
    with open('config/training_config_text.json', 'r') as f:
        config = json.load(f)
    model_name = config.get('training', {}).get('model_name_or_path', '')
    print(model_name)
except:
    print('flan-ul2')  # Default
")

# Check if the model is a T5/UL2 model (seq2seq) vs a causal LM
IS_SEQ2SEQ=$(python -c "
import sys
model_name = '$MODEL_NAME'.lower()
# T5, UL2, and FLAN models are typically seq2seq
if any(name in model_name for name in ['t5', 'ul2', 'flan']):
    print('true')
else:
    print('false')
")

# Update config to use optimized training settings
echo "===== UPDATING CONFIG FOR OPTIMIZED TRAINING ====="
python -c "
import json
import sys
try:
    with open('config/training_config_text.json', 'r') as f:
        config = json.load(f)
    
    if 'training' not in config:
        config['training'] = {}
    
    # Set model architecture type - this is used by train_text_flan.py
    # Use proper Python booleans (True/False, not true/false)
    is_seq2seq = '$IS_SEQ2SEQ' == 'true'
    config['training']['is_seq2seq'] = is_seq2seq
    
    # Add optimization settings appropriate for the model type
    if is_seq2seq:
        print('Configuring for seq2seq model (T5/UL2/FLAN)')
        # Best settings for T5/UL2 models
        config['training']['bf16'] = True           # Use bfloat16 precision
        config['training']['fp16'] = False          # Don't use fp16 with bf16
        config['training']['optim'] = 'adamw_torch' # Best optimizer for T5
        
        # Ensure task_type is correct for seq2seq
        config['training']['task_type'] = 'SEQ_TO_SEQ_LM'
        
        # These settings improve T5/UL2 training
        config['training']['gradient_checkpointing'] = True
        config['training']['use_cache'] = False
    else:
        print('Configuring for causal language model')
        # For causal LMs, we can use Unsloth
        config['training']['use_unsloth'] = True
        config['training']['bf16'] = True
        config['training']['fp16'] = False
        config['training']['task_type'] = 'CAUSAL_LM'
    
    # Add memory optimizations
    config['training']['gradient_checkpointing'] = True
    config['training']['gradient_accumulation_steps'] = $GRAD_ACCUMULATION
    config['training']['per_device_train_batch_size'] = $BATCH_SIZE
    config['training']['learning_rate'] = $LR
    config['training']['max_steps'] = $MAX_STEPS
    config['training']['save_steps'] = $SAVE_STEPS
    config['training']['warmup_steps'] = $WARMUP_STEPS
    
    # Add Google Drive settings
    use_drive = '$USE_DRIVE' == 'true'
    config['training']['use_drive'] = use_drive
    if use_drive:
        config['training']['drive_base_dir'] = '$DRIVE_BASE_DIR'
        print(f'Configured Drive integration with base dir: {config.get(\"training\", {}).get(\"drive_base_dir\")}')
    
    # Update training configuration settings
    with open('config/training_config_text.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print('Config file updated successfully for optimized training')
except Exception as e:
    print(f'Error updating config: {e}', file=sys.stderr)
"

# Step 1: Get dataset names from the config file
echo "===== IDENTIFYING TEXT DATASETS FROM CONFIG ====="
TEXT_DATASETS=$(python -c "
import json
import sys
try:
    with open('config/dataset_config_text.json', 'r') as f:
        config = json.load(f)
        # Filter only enabled datasets
        enabled_datasets = [name for name, info in config.items() 
                           if info.get('enabled', True)]
        if enabled_datasets:
            print(' '.join(enabled_datasets))
        else:
            # Default datasets if none are enabled
            print('openassistant gpteacher_general pile synthetic_persona writingprompts')
except Exception as e:
    print('openassistant gpteacher_general pile')  # Fallback list
    sys.stderr.write(f'Error reading dataset config: {e}\\n')
")
echo "Datasets in configuration: $TEXT_DATASETS"

# Step 2: Check which datasets are already processed and available on Drive
echo "===== CHECKING FOR PRE-PROCESSED DATASETS ====="
if [ "${USE_DRIVE}" = "true" ]; then
    echo "Google Drive integration is ENABLED. Looking for pre-processed datasets in Drive folder: $DRIVE_BASE_DIR"
else
    echo "Google Drive integration is DISABLED. Using local datasets only."
fi

# Run the dataset checker to identify available vs needed datasets
echo "Running dataset availability check..."
DATASET_STATUS=$(python -c "
from src.utils.drive_dataset_checker import prepare_datasets
import json
import sys
import os

config_path = 'config/dataset_config_text.json'
skip_drive = '$USE_DRIVE' != 'true'
drive_folder = '$DRIVE_BASE_DIR/preprocessed' if not skip_drive else None

try:
    # This gets datasets available locally or on Drive, and those still needed
    available, needed, download_time = prepare_datasets(
        config_path, 
        output_dir='data/processed',
        drive_folder=drive_folder,
        skip_drive=skip_drive
    )
    
    print('AVAILABLE=' + ','.join(available), file=sys.stdout)
    print('NEEDED=' + ','.join(needed), file=sys.stdout)
    print('DOWNLOAD_TIME=' + str(download_time), file=sys.stdout)
except Exception as e:
    print(f'Error checking datasets: {e}', file=sys.stderr)
    # Fallback to processing all datasets
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        all_datasets = [name for name, info in config.items() if info.get('enabled', True)]
        print('AVAILABLE=', file=sys.stdout)
        print('NEEDED=' + ','.join(all_datasets), file=sys.stdout)
        print('DOWNLOAD_TIME=0', file=sys.stdout)
    except Exception as e2:
        print(f'Error reading config: {e2}', file=sys.stderr)
        print('AVAILABLE=', file=sys.stdout)
        print('NEEDED=openassistant gpteacher_general pile', file=sys.stdout)
        print('DOWNLOAD_TIME=0', file=sys.stdout)
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

# Step 3: Process only the datasets that need processing
if [ -n "$DATASETS_TO_PROCESS" ]; then
    echo "===== PROCESSING DATASETS ====="
    echo "Processing datasets: $DATASETS_TO_PROCESS"
    
    python -m src.data.process_datasets \
      --config config/dataset_config_text.json \
      --datasets $DATASETS_TO_PROCESS \
      --streaming \
      --no_cache \
      $DRIVE_OPTS \
      --verbose
      
    if [ $? -ne 0 ]; then
        echo "⚠️ Dataset processing encountered errors. Training may use incomplete data."
    else  
        echo "✅ Dataset processing completed successfully."
    fi
else
    echo "✅ All datasets already available. Skipping dataset processing step."
fi

# Step 4: Train the model with appropriate optimizations
if [ "$IS_SEQ2SEQ" = "true" ]; then
    echo "==== Training FLAN-UL2 with seq2seq optimizations ===="
else
    echo "==== Training with Unsloth optimizations ===="
fi

TRAINING_START_TIME=$(date +%s)

# Start training
python train_text_flan.py \
  --config config/training_config_text.json \
  --data_dir data/processed \
  --push_to_hub \
  $DRIVE_OPTS \
  --debug \
  2>&1 | tee logs/train_flan_ul2_a6000_$(date +%Y%m%d_%H%M%S).log

# Calculate actual training time
TRAINING_END_TIME=$(date +%s)
ACTUAL_TRAINING_TIME=$((TRAINING_END_TIME - TRAINING_START_TIME))
# Use Python instead of bc for floating point division
ACTUAL_HOURS=$(python -c "print('{:.2f}'.format($ACTUAL_TRAINING_TIME / 3600))")

echo "==== Training Summary ===="
echo "Actual training time: $ACTUAL_HOURS hours (estimated: $ESTIMATED_HOURS hours)"

# Create completion marker with timing information
COMPLETION_FILE="logs/flan_ul2_training_complete_$(date +%Y%m%d_%H%M%S).txt"
echo "Training completed at $(date)" > $COMPLETION_FILE
echo "Actual training time: $ACTUAL_HOURS hours" >> $COMPLETION_FILE
echo "Estimated training time: $ESTIMATED_HOURS hours" >> $COMPLETION_FILE
echo "Model parameters:" >> $COMPLETION_FILE
echo "- Max steps: $MAX_STEPS" >> $COMPLETION_FILE
echo "- Batch size: $BATCH_SIZE" >> $COMPLETION_FILE
echo "- Gradient accumulation: $GRAD_ACCUMULATION" >> $COMPLETION_FILE
echo "- Learning rate: $LR" >> $COMPLETION_FILE

# Calculate training statistics from log file
LOG_FILE=$(find logs -name "train_flan_ul2_a6000_*.log" -type f -exec ls -t {} \; | head -1)
if [ -f "$LOG_FILE" ]; then
  echo "Extracting statistics from training log..."
  
  # Extract key metrics
  FINAL_LOSS=$(grep -o "loss: [0-9.]*" $LOG_FILE | tail -1 | cut -d' ' -f2)
  STEPS_COMPLETED=$(grep -o "Step [0-9]*/" $LOG_FILE | tail -1 | cut -d' ' -f2 | cut -d'/' -f1)
  
  # Add to summary
  echo "Steps completed: $STEPS_COMPLETED of $MAX_STEPS" >> $COMPLETION_FILE
  echo "Final loss: $FINAL_LOSS" >> $COMPLETION_FILE
  
  # Calculate steps per second
  if [ -n "$STEPS_COMPLETED" ] && [ "$ACTUAL_TRAINING_TIME" -gt 0 ]; then
    # Use Python instead of bc for floating point calculations
    STEPS_PER_SEC=$(python -c "print('{:.4f}'.format($STEPS_COMPLETED / $ACTUAL_TRAINING_TIME))")
    STEPS_PER_HOUR=$(python -c "print('{:.2f}'.format($STEPS_PER_SEC * 3600))")
    echo "Training speed: $STEPS_PER_HOUR steps/hour" >> $COMPLETION_FILE
    echo "Training speed: $STEPS_PER_HOUR steps/hour"
  fi
fi

# Step 5: Sync any remaining results to Drive if enabled
if [ "$USE_DRIVE" = "true" ]; then
  echo "==== Syncing all results to Google Drive ===="
  python scripts/sync_to_drive.py --sync-all \
    --base-dir "$DRIVE_BASE_DIR" \
    --is-text-model \
    $([ "$SKIP_LOCAL" = "true" ] && echo "--delete-local")
fi

echo "==== Training completed! ===="
echo "Check the text_models directory for model files or the Google Drive folder if Drive integration was enabled." 