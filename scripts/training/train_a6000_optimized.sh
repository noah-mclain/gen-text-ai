#!/bin/bash
# Fully optimized training script for A6000 GPU with 48GB VRAM using advanced optimizations
# This script is configured for maximum performance with multi-language support
# Now with feature extraction from preprocessed datasets

# Run the comprehensive training environment fix script
echo "===== FIXING TRAINING ENVIRONMENT ====="
if [ -f "scripts/utilities/fix_training_environment.py" ]; then
    python scripts/utilities/fix_training_environment.py
    if [ $? -ne 0 ]; then
        echo "⚠️ Warning: Training environment fix encountered issues. Some features may not work properly."
    else
        echo "✅ Training environment fixed successfully."
    fi
else
    echo "⚠️ Warning: Training environment fix script not found. Will proceed with individual fixes."
    
    # Ensure the feature extractor module is properly set up
    echo "===== CHECKING FEATURE EXTRACTOR SETUP ====="
    if [ -f "scripts/utilities/ensure_feature_extractor.py" ]; then
        python scripts/utilities/ensure_feature_extractor.py
        if [ $? -ne 0 ]; then
            echo "⚠️ Warning: Feature extractor setup check failed. Some features may not work properly."
        else
            echo "✅ Feature extractor setup verified."
        fi
    else
        echo "⚠️ Warning: Feature extractor setup script not found. Will proceed without verification."
    fi

    # Create necessary symlinks for training files
    if [ -f "scripts/utilities/create_training_symlinks.py" ]; then
        python scripts/utilities/create_training_symlinks.py
        if [ $? -ne 0 ]; then
            echo "⚠️ Warning: Failed to create training symlinks. Some scripts may not be found."
        else
            echo "✅ Training symlinks created successfully."
        fi
    fi

    # Fix feature extractor path
    if [ -f "scripts/utilities/fix_feature_extractor_path.py" ]; then
        python scripts/utilities/fix_feature_extractor_path.py
        if [ $? -ne 0 ]; then
            echo "⚠️ Warning: Failed to fix feature extractor path. Feature extraction may fail."
        else
            echo "✅ Feature extractor path fixed successfully."
        fi
    fi

    # Set up Google Drive integration
    if [ -f "scripts/google_drive/setup_google_drive.py" ]; then
        python scripts/google_drive/setup_google_drive.py
        if [ $? -ne 0 ]; then
            echo "Google Drive authentication setup failed. Will proceed without Drive integration."
            export SKIP_DRIVE="--skip_drive"
        else
            echo "✅ Google Drive authentication setup successful."
        fi
    else
        echo "Google Drive authentication setup failed. Will proceed without Drive integration."
        export SKIP_DRIVE="--skip_drive"
    fi
fi

# Ensure xformers and Unsloth are properly configured
echo "===== CONFIGURING XFORMERS AND UNSLOTH ====="
if [ -f "scripts/environment/fix_xformers_env.py" ]; then
    python scripts/environment/fix_xformers_env.py
    if [ $? -ne 0 ]; then
        echo "⚠️ Warning: xformers configuration encountered issues. Performance may be reduced."
    else
        echo "✅ xformers and Unsloth configured successfully."
    fi
else
    echo "⚠️ Warning: xformers configuration script not found. Performance may be reduced."
fi

# Set environment variables for better performance
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false  # Explicitly disable tokenizers parallelism to avoid fork warnings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Set a reasonable number of dataloader workers to avoid "Too many workers" errors
# For small datasets or streaming, 1-2 workers is usually sufficient
export NUM_WORKERS=1  # Limit dataloader workers
export DATALOADER_NUM_WORKERS=1  # Additional environment variable for safety

# Feature extraction settings - can be overridden via command line
EXTRACT_FEATURES=true
FEATURES_DIR="data/processed/features/deepseek_coder"
MAX_LENGTH=2048
TEXT_COLUMN="text"
IS_ENCODER_DECODER=false

# Parse command line arguments for feature extraction
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-features)
            EXTRACT_FEATURES=false
            shift
            ;;
        --features-only)
            FEATURES_ONLY=true
            shift
            ;;
        --features-dir)
            FEATURES_DIR="$2"
            shift 2
            ;;
        --max-length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --text-column)
            TEXT_COLUMN="$2"
            shift 2
            ;;
        --is-encoder-decoder)
            IS_ENCODER_DECODER=true
            shift
            ;;
        *)
            # Keep all other arguments for later processing
            REMAINING_ARGS="$REMAINING_ARGS $1"
            shift
            ;;
    esac
done
# Reset the arguments to the ones we didn't process
set -- $REMAINING_ARGS

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

# Add xformers installation and setup
echo "===== CHECKING XFORMERS INSTALLATION ====="
if python -c "import xformers" 2>/dev/null; then
    echo "✅ xformers is already installed"
else
    echo "Installing xformers for memory-efficient attention..."
    pip install xformers --quiet
    if [ $? -ne 0 ]; then
        echo "⚠️ Failed to install xformers. Will proceed without it."
    else
        echo "✅ xformers installed successfully"
    fi
fi

# Setup xformers environment variables
export XFORMERS_ENABLE_FLASH_ATTN=true
export XFORMERS_MEM_EFF_ATTN=true
export TRANSFORMERS_USE_XFORMERS=true

# Set environment variables for mixed precision training (instead of command-line args)
export ACCELERATE_MIXED_PRECISION="bf16"

# Ensure unsloth is installed
echo "===== CHECKING UNSLOTH INSTALLATION ====="
if python -c "import unsloth" 2>/dev/null; then
    echo "✅ Unsloth is already installed"
else
    echo "Installing Unsloth for optimized training..."
    pip install unsloth --quiet
    if [ $? -ne 0 ]; then
        echo "⚠️ Failed to install Unsloth. Performance will be reduced."
    else
        echo "✅ Unsloth installed successfully"
    fi
fi

# Remove any existing DeepSpeed config files to prevent conflicts
if [ -f "ds_config.json" ]; then
    echo "Removing existing DeepSpeed config file: ds_config.json"
    rm ds_config.json
fi
if [ -f "ds_config_a6000.json" ]; then
    echo "Removing existing DeepSpeed config file: ds_config_a6000.json"
    rm ds_config_a6000.json
fi

# Create output directories for features
mkdir -p $FEATURES_DIR

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

# If running on Paperspace, make sure the notebooks directory exists and create dataset directory
if [ -d "/notebooks" ]; then
    echo "===== DETECTED PAPERSPACE ENVIRONMENT ====="
    echo "Creating datasets directory in /notebooks/data/processed"
    mkdir -p /notebooks/data/processed
    
    # If Google Drive is authenticated, sync all datasets from Drive to notebooks directory
    if [ -z "$SKIP_DRIVE" ]; then
        echo "===== SYNCING DATASETS FROM GOOGLE DRIVE TO PAPERSPACE ====="
        # Run a Python script to sync all datasets from Drive to notebooks directory
        python -c "
import os
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path for imports
project_root = Path(__name__).parent
sys.path.append(str(project_root))

# Import Google Drive manager
try:
    from src.utils.google_drive_manager import (
        drive_manager,
        sync_from_drive,
        test_authentication
    )
    
    # Test authentication
    if not drive_manager.authenticated:
        if not test_authentication():
            logger.error('Google Drive authentication failed')
            sys.exit(1)
    
    # Get all available datasets
    logger.info('Checking for datasets on Google Drive')
    drive_folder = 'preprocessed'
    
    # Find the drive folder
    folder_id = drive_manager.folder_ids.get(drive_folder)
    if not folder_id:
        folder_id = drive_manager.find_file_id(drive_folder)
        if not folder_id:
            logger.error(f'Drive folder \"{drive_folder}\" not found')
            sys.exit(1)
    
    # List all folders in the preprocessed directory
    try:
        results = drive_manager.service.files().list(
            q=f\"'{folder_id}' in parents and mimeType='application/vnd.google-apps.folder'\",
            fields='files(id, name)'
        ).execute()
        
        dataset_folders = results.get('files', [])
        
        if not dataset_folders:
            logger.warning('No dataset folders found on Google Drive')
            sys.exit(0)
        
        logger.info(f'Found {len(dataset_folders)} dataset folders on Google Drive')
        
        # Download each dataset folder to the notebooks directory
        notebooks_dir = '/notebooks/data/processed'
        os.makedirs(notebooks_dir, exist_ok=True)
        
        # Check what's already in the local notebooks directory
        existing_datasets = set(os.listdir(notebooks_dir)) if os.path.exists(notebooks_dir) else set()
        
        # First download the dataset_info.json files to check sizes
        for folder in dataset_folders:
            folder_name = folder['name']
            folder_id = folder['id']
            
            # Skip if already exists (to avoid redownloading large datasets)
            local_folder_path = os.path.join(notebooks_dir, folder_name)
            if os.path.exists(local_folder_path):
                logger.info(f'Dataset {folder_name} already exists in {notebooks_dir}, skipping download')
                continue
            
            logger.info(f'Downloading dataset {folder_name} from Google Drive to {notebooks_dir}')
            
            # Download the folder
            if sync_from_drive(f'{drive_folder}/{folder_name}', local_folder_path):
                logger.info(f'Successfully downloaded dataset {folder_name}')
            else:
                logger.error(f'Failed to download dataset {folder_name}')
        
        logger.info('Completed syncing datasets from Google Drive to notebooks directory')
        
    except Exception as e:
        logger.error(f'Error listing dataset folders: {e}')
        sys.exit(1)
        
except ImportError as e:
    logger.error(f'Failed to import Drive manager: {e}')
    sys.exit(1)
        "
        
        # Check if the sync was successful
        if [ $? -ne 0 ]; then
            echo "⚠️ Failed to sync datasets from Google Drive. Will use local datasets if available."
        else
            echo "✅ Successfully synced datasets from Google Drive to /notebooks/data/processed"
            echo "Using datasets from /notebooks/data/processed for training"
        fi
    fi
fi

# Clean any cached files for better memory management
echo "Cleaning cache directories..."
rm -rf ~/.cache/huggingface/datasets/downloads/completed.lock

# Fix dataset loading for Arrow format
if [ -f "scripts/utilities/fix_dataset_loading.py" ]; then
    echo "===== FIXING DATASET LOADING FOR ARROW FORMAT ====="
    python scripts/utilities/fix_dataset_loading.py
    if [ $? -ne 0 ]; then
        echo "⚠️ Warning: Failed to fix dataset loading. Training may use incomplete features."
    else
        echo "✅ Dataset loading fixed successfully!"
    fi
fi

# ===== FEATURE EXTRACTION STEP =====
if [ "$EXTRACT_FEATURES" = true ]; then
    echo "===== STARTING FEATURE EXTRACTION ====="
    
    # Build feature extraction flags
    DRIVE_FE_FLAG=""
    if [ -z "$SKIP_DRIVE" ]; then
        DRIVE_FE_FLAG="--from_google_drive --drive_base_dir DeepseekCoder"
    fi
    
    ENCODER_DECODER_FLAG=""
    if [ "$IS_ENCODER_DECODER" = true ]; then
        ENCODER_DECODER_FLAG="--is_encoder_decoder"
    fi
    
    # Determine model name from config
    BASE_MODEL=$(python -c "
import json
try:
    with open('config/training_config.json', 'r') as f:
        config = json.load(f)
    model_name = config.get('model', {}).get('base_model') or config.get('training', {}).get('model_name_or_path')
    print(model_name)
except Exception as e:
    print('deepseek-ai/deepseek-coder-6.7b-base')
    ")
    
    echo "Starting feature extraction for model: $BASE_MODEL"
    echo "Features will be saved to: $FEATURES_DIR"
    
    # Run the feature extraction script
    # First try the project path
    if [ -f "scripts/datasets/prepare_datasets_for_training.py" ]; then
        python scripts/datasets/prepare_datasets_for_training.py \
            --model_name "$BASE_MODEL" \
            --config "config/training_config.json" \
            --dataset_config "config/dataset_config.json" \
            --output_dir "$FEATURES_DIR" \
            --text_column "$TEXT_COLUMN" \
            --max_length "$MAX_LENGTH" \
            --batch_size 1000 \
            --num_proc 4 \
            $ENCODER_DECODER_FLAG \
            $DRIVE_FE_FLAG
    # If not found, try the notebooks path
    elif [ -f "/notebooks/scripts/prepare_datasets_for_training.py" ]; then
        python /notebooks/scripts/prepare_datasets_for_training.py \
            --model_name "$BASE_MODEL" \
            --config "config/training_config.json" \
            --dataset_config "config/dataset_config.json" \
            --output_dir "$FEATURES_DIR" \
            --text_column "$TEXT_COLUMN" \
            --max_length "$MAX_LENGTH" \
            --batch_size 1000 \
            --num_proc 4 \
            $ENCODER_DECODER_FLAG \
            $DRIVE_FE_FLAG
    else
        echo "⚠️ Error: prepare_datasets_for_training.py not found. Feature extraction will be skipped."
        EXTRACT_FEATURES=false
    fi
    
    # Check if feature extraction was successful
    if [ $? -ne 0 ] && [ "$EXTRACT_FEATURES" = true ]; then
        echo "⚠️ Feature extraction encountered errors. Training may use incomplete features."
    elif [ "$EXTRACT_FEATURES" = true ]; then
        echo "✅ Feature extraction completed successfully!"
    fi
    
    # If features-only flag is set, exit after feature extraction
    if [ "${FEATURES_ONLY:-false}" = true ]; then
        echo "Features-only flag was set. Exiting after feature extraction."
        exit 0
    fi
fi

# Continue with the rest of the script - dataset checking, processing, and training

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
echo "===== ENABLING DATASETS IN CONFIG (EXCLUDING THE_STACK) ====="
python -c "
import json
import sys
import os

try:
    # Get the skip_drive value from environment
    skip_drive = os.environ.get('SKIP_DRIVE', 'False')
    
    # Load the dataset config
    with open('config/dataset_config.json', 'r') as f:
        config = json.load(f)
    
    # Enable all datasets EXCEPT for 'the_stack' datasets
    for dataset_name in config:
        if 'the_stack' in dataset_name.lower():
            config[dataset_name]['enabled'] = False
            print(f'Disabled dataset: {dataset_name}')
        else:
            config[dataset_name]['enabled'] = True
            print(f'Enabled dataset: {dataset_name}')
    
    with open('config/dataset_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print('Dataset configuration updated - all non-stack datasets enabled')
except Exception as e:
    print(f'Error configuring datasets: {e}', file=sys.stderr)
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
        # Directory prefixes -> config dataset names
        'code_alpaca_processed': ['code_alpaca', 'alpaca'],
        'codeparrot_processed': ['codeparrot'],
        'codesearchnet_all_go_processed': ['codesearchnet_go'],
        'codesearchnet_go_processed': ['codesearchnet_go'],
        'codesearchnet_all_java_processed': ['codesearchnet_java'],
        'codesearchnet_java_processed': ['codesearchnet_java'],
        'codesearchnet_all_javascript_processed': ['codesearchnet_javascript'],
        'codesearchnet_javascript_processed': ['codesearchnet_javascript'],
        'codesearchnet_all_php_processed': ['codesearchnet_php'],
        'codesearchnet_php_processed': ['codesearchnet_php'],
        'codesearchnet_all_python_processed': ['codesearchnet_python'],
        'codesearchnet_python_processed': ['codesearchnet_python'],
        'codesearchnet_all_ruby_processed': ['codesearchnet_ruby'],
        'codesearchnet_ruby_processed': ['codesearchnet_ruby'],
        'humaneval_processed': ['humaneval'],
        'instruct_code_processed': ['instruct_code'],
        'mbpp_processed': ['mbpp'],
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
    
    # Add features directory configuration if feature extraction is enabled
    features_dir = os.environ.get('FEATURES_DIR', 'data/processed/features/deepseek_coder')
    if os.path.exists(features_dir):
        if 'features_dir' not in config['training']:
            config['training']['features_dir'] = features_dir
            print(f'Set features_dir to {features_dir} in config')
    
    # Dataloader worker settings to prevent 'too many workers' errors
    config['training']['dataloader_num_workers'] = 1
    config['training']['dataloader_pin_memory'] = True
    print('Set dataloader worker settings to avoid errors')
    
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
    
    # Explicitly enable xformers and Unsloth
    config['training']['use_xformers'] = True
    print('Enabled xformers for memory-efficient attention')
    
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
    
    # Function to check both local and notebooks directories
    def find_dataset_paths(search_prefixes, local_only=False):
        dataset_paths = {}
        
        # Check both potential dataset directories
        data_dirs = ['data/processed']
        if not local_only and os.path.exists('/notebooks/data/processed'):
            data_dirs.append('/notebooks/data/processed')
        
        for data_dir in data_dirs:
            if not os.path.exists(data_dir):
                continue
                
            # Get all directories in this data directory
            all_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
            
            # Check for each prefix
            for prefix in search_prefixes:
                matches = [d for d in all_dirs if d.startswith(prefix)]
                if matches:
                    # Sort to prioritize final over interim when multiple exist
                    matches.sort(key=lambda x: 0 if 'final' in x else 1)
                    dataset_paths[data_dir] = os.path.join(data_dir, matches[0])
                    print(f'Found dataset in {data_dir} with prefix {prefix}')
                    # Return once found in any directory
                    return dataset_paths[data_dir]
        
        # If nothing found, return None
        return None
    
    # Define dataset mapping structure
    dataset_mapping = {
        'code_alpaca': ['code_alpaca_processed_interim_6000', 'code_alpaca_processed_interim_final', 'code_alpaca_processed'],
        'codeparrot': ['codeparrot_processed'],
        'codesearchnet_go': ['codesearchnet_all_go_processed', 'codesearchnet_go_processed_interim_6000', 'codesearchnet_go_processed_interim_final', 'codesearchnet_go_processed'],
        'codesearchnet_java': ['codesearchnet_all_java_processed', 'codesearchnet_java_processed_interim_6000', 'codesearchnet_java_processed_interim_final', 'codesearchnet_java_processed'],
        'codesearchnet_javascript': ['codesearchnet_all_javascript_processed', 'codesearchnet_javascript_processed_interim_6000', 'codesearchnet_javascript_processed_interim_final', 'codesearchnet_javascript_processed'],
        'codesearchnet_php': ['codesearchnet_all_php_processed', 'codesearchnet_php_processed_interim_6000', 'codesearchnet_php_processed_interim_final', 'codesearchnet_php_processed'],
        'codesearchnet_python': ['codesearchnet_all_python_processed', 'codesearchnet_python_processed_interim_9000', 'codesearchnet_python_processed_interim_final', 'codesearchnet_python_processed'],
        'codesearchnet_ruby': ['codesearchnet_all_ruby_processed', 'codesearchnet_ruby_processed_interim_6000', 'codesearchnet_ruby_processed_interim_final', 'codesearchnet_ruby_processed'],
        'codesearchnet_all': ['codesearchnet_all_processed'],
        'humaneval': ['humaneval_processed'],
        'instruct_code': ['instruct_code_processed_interim_6000', 'instruct_code_processed_interim_final', 'instruct_code_processed'],
        'mbpp': ['mbpp_processed'],
    }
    
    # Create a new dataset paths dictionary
    new_dataset_paths = {}
    
    # Look for each dataset
    for config_name, search_prefixes in dataset_mapping.items():
        path = find_dataset_paths(search_prefixes)
        if path:
            new_dataset_paths[config_name] = path
            print(f'Mapped dataset {config_name} to {path}')
    
    # Update the config with the new paths
    config['dataset_paths'] = new_dataset_paths
    
    # Add a comment about notebook environment detection for dataset loading
    config['use_notebook_paths'] = os.path.exists('/notebooks')
    
    with open('config/training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print('Config file updated successfully for optimized training with Unsloth and xformers')
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

# Set explicit environment variables right before launching the training
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_NUM_PROC=1
export DATALOADER_NUM_WORKERS=1

# Determine which data directory to use based on environment
if [ -d "/notebooks" ] && [ -d "/notebooks/data/processed" ]; then
    echo "Using Paperspace /notebooks/data/processed directory for datasets"
    DATA_DIR="/notebooks/data/processed"
else
    echo "Using local data/processed directory for datasets"
    DATA_DIR="data/processed"
fi

# Add feature extraction flag if features are available
FEATURES_FLAG=""
if [ "$EXTRACT_FEATURES" = true ] && [ -d "$FEATURES_DIR" ]; then
    echo "Using extracted features from $FEATURES_DIR"
    FEATURES_FLAG="--features_dir $FEATURES_DIR"
fi

# Run training with proper flags to ensure no DeepSpeed and use Unsloth and xformers
python -m src.training.train \
    --config config/training_config.json \
    --data_dir $DATA_DIR \
    $USE_DRIVE_FLAG \
    $FEATURES_FLAG \
    --push_to_hub \
    --no_deepspeed \
    --estimate_time \
    --dataloader_workers 1 \
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
    
    # Also sync the extracted features if they exist
    if [ -d "$FEATURES_DIR" ]; then
      echo "Syncing extracted features to Google Drive..."
      python -c "
from src.utils.google_drive_manager import sync_to_drive
import os

features_dir = os.environ.get('FEATURES_DIR', 'data/processed/features/deepseek_coder')
if os.path.exists(features_dir):
    print(f'Syncing {features_dir} to Drive...')
    sync_to_drive(features_dir, 'features')
      "
    fi
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

# Add fix for dataloader worker issues in the source code
echo "===== PATCHING TRAINING MODULES FOR DATALOADER FIX ====="
python -c "
import sys
import os
import re

# Files to patch
training_files = [
    'src/training/train.py',
    'src/training/trainer.py'
]

for filepath in training_files:
    if not os.path.exists(filepath):
        print(f'File {filepath} not found, skipping')
        continue
        
    print(f'Checking {filepath} for dataloader settings...')
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check if file mentions dataloader_num_workers
    if 'dataloader_num_workers' in content:
        print(f'File {filepath} already has dataloader_num_workers mentioned, ensuring it respects environment variables')
        
        # Use regex to find TrainingArguments creation with proper dataloader_num_workers setting
        modified = False
        
        # Pattern to match TrainingArguments constructor
        pattern = r'(TrainingArguments\s*\(\s*[^)]*)'
        
        def add_dataloader_settings(match):
            args_text = match.group(1)
            # Only add if not already present
            if 'dataloader_num_workers' not in args_text:
                # Add dataloader_num_workers and return updated text
                return args_text + ',\\n    dataloader_num_workers=1'
            return args_text
        
        # Apply the substitution
        new_content = re.sub(pattern, add_dataloader_settings, content)
        
        if new_content != content:
            print(f'Updated {filepath} with dataloader_num_workers=1')
            with open(filepath, 'w') as f:
                f.write(new_content)
            modified = True
        
        # Check for DataLoader instantiation
        dataloader_pattern = r'DataLoader\s*\(\s*[^)]*\)'
        if re.search(dataloader_pattern, content):
            print(f'Found DataLoader usage in {filepath}, ensuring it uses limited workers')
            
            # Wrap DataLoader calls with num_workers=1
            dataloader_fix = '''
# Monkey patch DataLoader to always use limited workers
original_dataloader = torch.utils.data.DataLoader
def safe_dataloader(*args, **kwargs):
    # Always use 1 worker to avoid 'Too many workers' error
    kwargs['num_workers'] = int(os.environ.get('DATALOADER_NUM_WORKERS', 1))
    return original_dataloader(*args, **kwargs)
torch.utils.data.DataLoader = safe_dataloader
'''
            
            # Add the monkey patch if not present
            if 'Monkey patch DataLoader to always use limited workers' not in content:
                # Insert after imports
                import_section_end = max(content.find('import torch'), 0) + 20
                new_content = content[:import_section_end] + dataloader_fix + content[import_section_end:]
                
                with open(filepath, 'w') as f:
                    f.write(new_content)
                print(f'Added DataLoader monkey patch to {filepath}')
                modified = True
        
        if not modified:
            print(f'No changes needed for {filepath}')
    else:
        print(f'File {filepath} does not mention dataloader_num_workers, adding dataloader settings')
        
        # Find a good insertion point after imports
        import_section_end = max(
            content.rfind('import') + 20,
            content.find('logging.basicConfig') + 30,
            200  # default if above not found
        )
        
        # Add our dataloader worker limiting code
        dataloader_fix = '''
# Limit dataloader workers to avoid 'Too many workers' error
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Explicitly disable tokenizers parallelism

# Ensure dataset loading respects worker limits
if 'DATALOADER_NUM_WORKERS' not in os.environ:
    os.environ['DATALOADER_NUM_WORKERS'] = '1'

# Monkey patch DataLoader to always use limited workers
if 'torch' in sys.modules:
    import torch
    original_dataloader = torch.utils.data.DataLoader
    def safe_dataloader(*args, **kwargs):
        # Always use limited workers to avoid 'Too many workers' error
        kwargs['num_workers'] = int(os.environ.get('DATALOADER_NUM_WORKERS', 1))
        return original_dataloader(*args, **kwargs)
    torch.utils.data.DataLoader = safe_dataloader
'''
        
        # Insert the fix
        new_content = content[:import_section_end] + dataloader_fix + content[import_section_end:]
        
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f'Added dataloader worker limiting code to {filepath}')

print('Completed patching source files for dataloader worker limits')
"
# Check if train.py needs to be modified to support dataloader workers argument
echo "===== CHECKING IF TRAIN.PY NEEDS DATALOADER ARGUMENT SUPPORT ====="
python -c "
import os
import sys
import re

# Main training file
train_file = 'src/training/train.py'

if os.path.exists(train_file):
    print(f'Checking {train_file} for argparse arguments...')
    
    with open(train_file, 'r') as f:
        content = f.read()
    
    # Check if the parser already has dataloader workers argument
    if 'dataloader_workers' not in content:
        print('Adding dataloader_workers argument to train.py')
        
        # Find the argument parser section
        parser_pattern = r'(parser\.add_argument.*?--disable_wandb.*?\))'
        
        # Prepare the new argument
        dataloader_arg = '''
    parser.add_argument('--dataloader_workers', type=int, default=1,
                        help='Number of workers for DataLoader')'''
        
        # Add the argument after the disable_wandb argument
        modified_content = re.sub(parser_pattern, r'\\1' + dataloader_arg, content, flags=re.DOTALL)
        
        # Now find where args are processed
        args_process_pattern = r'(args = parser\.parse_args\(\))'
        
        # Prepare code to process the argument
        process_arg = '''
    # Set dataloader workers environment variable from argument
    os.environ['DATALOADER_NUM_WORKERS'] = str(args.dataloader_workers)
    logger.info(f'Setting dataloader workers to {args.dataloader_workers}')'''
        
        # Add the code after the args are parsed
        modified_content = re.sub(args_process_pattern, r'\\1' + process_arg, modified_content)
        
        # Write the modified file
        with open(train_file, 'w') as f:
            f.write(modified_content)
        
        print(f'Updated {train_file} to support --dataloader_workers argument')
    else:
        print(f'{train_file} already supports dataloader_workers argument')
else:
    print(f'Warning: {train_file} not found, cannot add dataloader_workers argument')
"

# Thoroughly validate local datasets and fix dataset directories using symbolic links
echo "===== FIXING DATASET MAPPING ISSUES ====="
# Apply our comprehensive dataset fixes
if [ -f "fix_dataset_features.sh" ]; then
    echo "Running comprehensive dataset fix script..."
    bash fix_dataset_features.sh
else
    # If the specific script doesn't exist, use the inline version
    python -c "
import os
import sys
import glob
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(console_handler)

# Dataset name mappings
DATASET_MAPPINGS = {
    # Expected name -> list of possible actual names
    'codesearchnet_all_processed': ['codesearchnet_all_*_processed', 'codesearchnet_all_*_processed_interim_*'],
    'codesearchnet_python_processed': ['codesearchnet_python_processed_*', 'codesearchnet_all_python_processed'],
    'codesearchnet_java_processed': ['codesearchnet_java_processed_*', 'codesearchnet_all_java_processed'],
    'codesearchnet_javascript_processed': ['codesearchnet_javascript_processed_*', 'codesearchnet_all_javascript_processed'],
    'codesearchnet_php_processed': ['codesearchnet_php_processed_*', 'codesearchnet_all_php_processed'],
    'codesearchnet_ruby_processed': ['codesearchnet_ruby_processed_*', 'codesearchnet_all_ruby_processed'],
    'codesearchnet_go_processed': ['codesearchnet_go_processed_*', 'codesearchnet_all_go_processed'],
    'code_alpaca_processed': ['code_alpaca_processed_interim_*'],
    'instruct_code_processed': ['instruct_code_processed_interim_*'],
}

def find_matching_dirs(data_dir, pattern):
    if not os.path.exists(data_dir):
        return []
    glob_pattern = os.path.join(data_dir, pattern)
    matches = glob.glob(glob_pattern)
    return [os.path.basename(path) for path in matches if os.path.isdir(path)]

def prioritize_dirs(dirs):
    def priority_key(directory):
        # Final versions have highest priority
        if 'final' in directory:
            return (0, 0)
        # For numbered versions, extract the number and sort by it
        import re
        numbers = re.findall(r'\\d+', directory)
        if numbers:
            # Use the highest number found
            max_number = max(int(n) for n in numbers)
            # Negative because we want higher numbers first
            return (1, -max_number)
        # Default priority based on position in list
        return (2, directory)
    return sorted(dirs, key=priority_key)

# First list datasets in each directory
for data_dir in ['data/processed', '/notebooks/data/processed']:
    if not os.path.exists(data_dir):
        continue
    
    logger.info(f'\\nChecking datasets in {data_dir}:')
    all_dirs = os.listdir(data_dir)
    dataset_dirs = [d for d in all_dirs if os.path.isdir(os.path.join(data_dir, d)) and ('_processed' in d or '_interim_' in d)]
    
    if dataset_dirs:
        logger.info(f'Found {len(dataset_dirs)} potential dataset directories:')
        for i, d in enumerate(sorted(dataset_dirs)):
            logger.info(f'  {i+1}. {d}')
    else:
        logger.info('No dataset directories found')
    
    # Create symbolic links for expected names
    for expected_name, patterns in DATASET_MAPPINGS.items():
        expected_path = os.path.join(data_dir, expected_name)
        if os.path.exists(expected_path):
            # Skip if it already exists
            continue
        
        # Find matching directories for all patterns
        all_matches = []
        for pattern in patterns:
            all_matches.extend(find_matching_dirs(data_dir, pattern))
        
        if all_matches:
            # Sort by priority
            best_matches = prioritize_dirs(all_matches)
            best_match = best_matches[0]
            source_path = os.path.join(data_dir, best_match)
            
            # Create symbolic link
            try:
                os.symlink(source_path, expected_path)
                logger.info(f'✅ Created symlink: {expected_name} -> {best_match}')
            except Exception as e:
                logger.warning(f'⚠️ Failed to create symlink for {expected_name}: {e}')
        else:
            logger.info(f'❌ No matching directory found for {expected_name}')
"
fi