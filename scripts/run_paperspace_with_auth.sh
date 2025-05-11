#!/bin/bash
#
# Enhanced script for running the pipeline on Paperspace with proper authentication
# This script ensures HF_TOKEN is properly set from environment variables
# before running the pipeline, and handles Google Drive integration.

# Set up error handling
set -e  # Exit immediately if a command fails
trap 'echo "Error occurred. Exiting..."; exit 1' ERR

# ANSI color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================================${NC}"
echo -e "${GREEN}DeepSeek-Coder Fine-Tuning Pipeline - Paperspace Edition${NC}"
echo -e "${BLUE}=========================================================${NC}"

# Parse command line arguments
MODE="process"
CONFIG_PATH="config/dataset_config.json"
TRAINING_CONFIG_PATH="config/training_config.json"
DRIVE_BASE_DIR="DeepseekCoder"
CREDENTIALS_PATH="credentials.json"
STREAMING=true
NO_CACHE=true
DATASETS=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --training_config)
      TRAINING_CONFIG_PATH="$2"
      shift 2
      ;;
    --drive_base_dir)
      DRIVE_BASE_DIR="$2"
      shift 2
      ;;
    --credentials)
      CREDENTIALS_PATH="$2"
      shift 2
      ;;
    --no-streaming)
      STREAMING=false
      shift
      ;;
    --use-cache)
      NO_CACHE=false
      shift
      ;;
    --datasets)
      shift
      while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
        DATASETS="$DATASETS $1"
        shift
      done
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      shift
      ;;
  esac
done

# Check if PYTHONPATH is properly set
if [[ ! "$PYTHONPATH" == *"$(pwd)"* ]]; then
  echo -e "${YELLOW}Adding current directory to PYTHONPATH${NC}"
  export PYTHONPATH="$(pwd):$PYTHONPATH"
fi

# Verify HF_TOKEN environment variable
if [ -z "$HF_TOKEN" ]; then
  echo -e "${RED}HF_TOKEN environment variable not set${NC}"
  echo -e "${YELLOW}Checking if it's in credentials.json...${NC}"
  
  if [ -f "$CREDENTIALS_PATH" ]; then
    # Check if jq is installed
    if command -v jq &> /dev/null; then
      if HF_TOKEN=$(jq -r '.huggingface.token // .hf_token // .api_keys.huggingface // ""' "$CREDENTIALS_PATH") && [ -n "$HF_TOKEN" ]; then
        echo -e "${GREEN}Found HF_TOKEN in credentials.json${NC}"
        export HF_TOKEN
      else
        echo -e "${YELLOW}No HF_TOKEN found in credentials.json${NC}"
      fi
    else
      echo -e "${YELLOW}jq not installed, using Python to extract token${NC}"
      HF_TOKEN=$(python -c "import json; f = open('$CREDENTIALS_PATH'); data = json.load(f); print(data.get('huggingface', {}).get('token') or data.get('hf_token') or data.get('api_keys', {}).get('huggingface', ''))")
      if [ -n "$HF_TOKEN" ]; then
        echo -e "${GREEN}Found HF_TOKEN in credentials.json${NC}"
        export HF_TOKEN
      else
        echo -e "${RED}No HF_TOKEN found in credentials.json${NC}"
        echo -e "${RED}Warning: Some datasets may be inaccessible${NC}"
      fi
    fi
  else
    echo -e "${RED}credentials.json not found at $CREDENTIALS_PATH${NC}"
    echo -e "${RED}Warning: Some datasets may be inaccessible${NC}"
  fi
else
  echo -e "${GREEN}HF_TOKEN environment variable is set${NC}"
fi

# Check if Google Drive credentials exist
if [ ! -f "$CREDENTIALS_PATH" ]; then
  echo -e "${RED}Google Drive credentials file not found at $CREDENTIALS_PATH${NC}"
  echo -e "${RED}Drive integration will be disabled${NC}"
  USE_DRIVE_API="false"
else
  USE_DRIVE_API="true"
  echo -e "${GREEN}Google Drive credentials found at $CREDENTIALS_PATH${NC}"
fi

# Build the base command
CMD="python main_api.py --mode $MODE"

# Add options based on mode
if [ "$MODE" = "process" ]; then
  CMD="$CMD --dataset_config $CONFIG_PATH"
  
  if [ -n "$DATASETS" ]; then
    CMD="$CMD --datasets$DATASETS"
  fi
  
  if [ "$STREAMING" = "true" ]; then
    CMD="$CMD --streaming"
  fi
  
  if [ "$NO_CACHE" = "true" ]; then
    CMD="$CMD --no_cache"
  fi
elif [ "$MODE" = "train" ]; then
  CMD="$CMD --training_config $TRAINING_CONFIG_PATH"
elif [ "$MODE" = "all" ]; then
  CMD="$CMD --dataset_config $CONFIG_PATH --training_config $TRAINING_CONFIG_PATH"
  
  if [ "$STREAMING" = "true" ]; then
    CMD="$CMD --streaming"
  fi
  
  if [ "$NO_CACHE" = "true" ]; then
    CMD="$CMD --no_cache"
  fi
fi

# Add Drive API options if enabled
if [ "$USE_DRIVE_API" = "true" ]; then
  CMD="$CMD --use_drive_api --credentials_path $CREDENTIALS_PATH --drive_base_dir $DRIVE_BASE_DIR --headless"
fi

# Display and execute the command
echo -e "${BLUE}===================${NC}"
echo -e "${GREEN}Executing command:${NC}"
echo -e "${YELLOW}$CMD${NC}"
echo -e "${BLUE}===================${NC}"

# Execute the command
eval "$CMD"

# Check the result
if [ $? -eq 0 ]; then
  echo -e "${GREEN}Command completed successfully${NC}"
else
  echo -e "${RED}Command failed with exit code $?${NC}"
  exit 1
fi 