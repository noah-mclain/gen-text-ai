#!/bin/bash

# Process and train using The Stack dataset with language filters
# This script efficiently samples and processes The Stack for training within a time constraint

# Set color codes for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================================${NC}"
echo -e "${GREEN}The Stack Direct Processing for Quick Training${NC}"
echo -e "${BLUE}=================================================${NC}"

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Navigate to the project root directory
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"
echo -e "${GREEN}Working directory: $(pwd)${NC}"

# Set PYTHONPATH to include the project root
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
echo -e "${GREEN}PYTHONPATH: $PYTHONPATH${NC}"

# Default configuration
CONFIG="config/dataset_config_updated.json"
TRAINING_CONFIG="config/training_config.json"
USE_DRIVE=false
DRIVE_BASE_DIR="DeepseekCoder"
SKIP_PREPROCESSING=true
MAX_TRAIN_TIME=4  # Hours available for training

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --training)
      TRAINING_CONFIG="$2"
      shift 2
      ;;
    --drive)
      USE_DRIVE=true
      shift
      ;;
    --drive-dir)
      DRIVE_BASE_DIR="$2"
      shift 2
      ;;
    --process)
      SKIP_PREPROCESSING=false
      shift
      ;;
    --max-hours)
      MAX_TRAIN_TIME="$2"
      shift 2
      ;;
    --help)
      echo -e "${GREEN}Usage: $0 [options]${NC}"
      echo -e "${YELLOW}Options:${NC}"
      echo "  --config FILE       Path to dataset config (default: config/dataset_config_updated.json)"
      echo "  --training FILE     Path to training config (default: config/training_config.json)"
      echo "  --drive             Enable Google Drive integration"
      echo "  --drive-dir DIR     Google Drive base directory (default: DeepseekCoder)"
      echo "  --process           Run preprocessing step (default: skip preprocessing)"
      echo "  --max-hours N       Maximum training time in hours (default: 4)"
      echo "  --help              Show this help message"
      exit 0
      ;;
    *)
      echo -e "${YELLOW}Unknown option: $1 (ignoring)${NC}"
      shift
      ;;
  esac
done

# Ensure necessary directories exist
mkdir -p data/raw data/processed models results visualizations logs

# Handle Hugging Face authentication
if [ -z "$HF_TOKEN" ]; then
  echo -e "${YELLOW}HF_TOKEN not set in environment, trying to set from credentials...${NC}"
  
  # Try to set token using the Python script
  python scripts/set_hf_token.py
  
  # Check if token was set
  if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}Warning: HF_TOKEN is not set. Some datasets may be inaccessible.${NC}"
  else
    echo -e "${GREEN}Successfully set HF_TOKEN from credentials${NC}"
  fi
else
  echo -e "${GREEN}HF_TOKEN is already set in environment${NC}"
fi

# Update training configuration for quick training
echo -e "${BLUE}Optimizing training configuration for time constraints...${NC}"

# Create a temporary file for the updated config
TMP_CONFIG=$(mktemp)

# Modify the training config for faster training
python - <<END
import json
import os

# Load the training config
with open("${TRAINING_CONFIG}", 'r') as f:
    config = json.load(f)

# Update for faster training
if "training" in config:
    # Limit epochs
    config["training"]["num_train_epochs"] = 1
    
    # Increase batch size if possible
    if config["training"].get("per_device_train_batch_size", 8) < 12:
        config["training"]["per_device_train_batch_size"] = 12
    
    # Increase gradient accumulation
    config["training"]["gradient_accumulation_steps"] = 8
    
    # Set evaluation and saving steps
    config["training"]["eval_steps"] = 100
    config["training"]["save_steps"] = 500
    
    # Enable mixed precision
    config["training"]["fp16"] = True
    
    # Set max training time
    max_hours = float("${MAX_TRAIN_TIME}")
    config["training"]["max_train_time_hours"] = max_hours

# Update dataset config for smaller sequence length
if "dataset" in config:
    # Reduce sequence length for faster training
    if config["dataset"].get("max_length", 2048) > 1024:
        config["dataset"]["max_length"] = 1024
    
    # Adjust sampling to account for the limited time
    if "max_samples" in config["dataset"]:
        # Reduce samples for all datasets except the_stack_filtered
        for key in config["dataset"]["max_samples"]:
            config["dataset"]["max_samples"][key] = min(5000, config["dataset"]["max_samples"].get(key, 5000))
    
    # Add our special stack dataset
    if "dataset_weights" in config["dataset"]:
        config["dataset"]["dataset_weights"]["the_stack_filtered"] = 3.0
        
    # Ensure streaming is enabled
    config["dataset"]["streaming"] = True

# Save the updated config
with open("${TMP_CONFIG}", 'w') as f:
    json.dump(config, f, indent=2)
END

# Move the temporary config to the original location
cp "${TMP_CONFIG}" "${TRAINING_CONFIG}"
echo -e "${GREEN}Updated training configuration for faster training${NC}"

# Prepare drive options flag
if [ "$USE_DRIVE" = "true" ]; then
  DRIVE_OPTS="--use_drive --drive_base_dir $DRIVE_BASE_DIR"
  echo -e "${GREEN}Using Google Drive integration with base directory: $DRIVE_BASE_DIR${NC}"
else
  DRIVE_OPTS=""
  echo -e "${YELLOW}Google Drive integration is disabled${NC}"
fi

# Process datasets if needed
if [ "$SKIP_PREPROCESSING" = "false" ]; then
    echo -e "${BLUE}Processing datasets before training...${NC}"
    
    # Build command
    PROCESS_CMD="python main_api.py --mode process --dataset_config $CONFIG --streaming --no_cache --datasets the_stack_filtered $DRIVE_OPTS"
    
    # Print and execute the processing command
    echo -e "${BLUE}===================${NC}"
    echo -e "${GREEN}Processing:${NC}"
    echo -e "${YELLOW}$PROCESS_CMD${NC}"
    echo -e "${BLUE}===================${NC}"
    
    eval "$PROCESS_CMD"
    
    if [ $? -ne 0 ]; then
      echo -e "${RED}Dataset processing failed with exit code $?${NC}"
      echo -e "${YELLOW}Continuing with training anyway using direct loading...${NC}"
    fi
fi

# Direct training with optimized parameters
echo -e "${BLUE}Starting training with time-optimized configuration...${NC}"

# Build command with direct loading and time optimizations
TRAIN_CMD="python main_api.py --mode train --training_config $TRAINING_CONFIG --streaming $DRIVE_OPTS"

# Print and execute the training command
echo -e "${BLUE}===================${NC}"
echo -e "${GREEN}Training with time constraint of ${MAX_TRAIN_TIME} hours:${NC}"
echo -e "${YELLOW}$TRAIN_CMD${NC}"
echo -e "${BLUE}===================${NC}"

# Record start time
START_TIME=$(date +%s)
echo -e "${GREEN}Training started at $(date)${NC}"

# Execute the command
eval "$TRAIN_CMD"

# Check result
TRAIN_RESULT=$?
if [ $TRAIN_RESULT -eq 0 ]; then
  echo -e "${GREEN}Training completed successfully!${NC}"
  
  # Sync results to Drive if enabled
  if [ "$USE_DRIVE" = "true" ]; then
    echo -e "${BLUE}Syncing results to Google Drive...${NC}"
    python scripts/sync_to_drive.py --sync-all --base-dir "$DRIVE_BASE_DIR"
  fi
else
  echo -e "${RED}Training failed with exit code $TRAIN_RESULT${NC}"
fi

# Calculate and display runtime
END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))
HOURS=$((RUNTIME / 3600))
MINUTES=$(( (RUNTIME % 3600) / 60 ))
SECONDS=$((RUNTIME % 60))

echo -e "${BLUE}===================${NC}"
echo -e "${GREEN}Training Complete${NC}"
echo -e "${GREEN}Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s${NC}"
echo -e "${BLUE}===================${NC}" 