#!/bin/bash

# Process datasets with authentication
# This script extracts the Hugging Face token from credentials.json and runs the dataset processing

# Set the path to credentials file
CREDENTIALS_PATH="credentials.json"

# Function to extract token from credentials JSON
extract_token() {
  # Check if jq is installed (for proper JSON parsing)
  if command -v jq >/dev/null 2>&1; then
    if jq -e '.huggingface.token' $CREDENTIALS_PATH >/dev/null 2>&1; then
      HF_TOKEN=$(jq -r '.huggingface.token' $CREDENTIALS_PATH)
    elif jq -e '.hf_token' $CREDENTIALS_PATH >/dev/null 2>&1; then
      HF_TOKEN=$(jq -r '.hf_token' $CREDENTIALS_PATH)
    elif jq -e '.api_keys.huggingface' $CREDENTIALS_PATH >/dev/null 2>&1; then
      HF_TOKEN=$(jq -r '.api_keys.huggingface' $CREDENTIALS_PATH)
    else
      echo "No Hugging Face token found in credentials file"
      return 1
    fi
  else
    # Fallback to Python if jq is not available
    HF_TOKEN=$(python -c "import json; f=open('$CREDENTIALS_PATH'); data=json.load(f); print(data.get('huggingface', {}).get('token', '') or data.get('hf_token', '') or data.get('api_keys', {}).get('huggingface', ''))")
  fi
  
  # Check if token was extracted
  if [ -z "$HF_TOKEN" ]; then
    echo "Could not extract token from credentials file"
    return 1
  fi
  
  return 0
}

# Check if credentials file exists
if [ ! -f "$CREDENTIALS_PATH" ]; then
  echo "Credentials file not found at $CREDENTIALS_PATH"
  exit 1
fi

# Extract token from credentials
echo "Extracting Hugging Face token from credentials..."
if extract_token; then
  echo "Successfully extracted HF_TOKEN"
  export HF_TOKEN
else
  echo "Failed to extract HF_TOKEN, some datasets may not be accessible"
fi

# Parse command line arguments
CONFIG="config/dataset_config.json"
STREAMING=0
NO_CACHE=0
USE_DRIVE_API=0
DRIVE_BASE_DIR="DeepseekCoder"
HEADLESS=0
DATASETS=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --datasets)
      shift
      while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
        DATASETS="$DATASETS $1"
        shift
      done
      ;;
    --streaming)
      STREAMING=1
      shift
      ;;
    --no_cache)
      NO_CACHE=1
      shift
      ;;
    --use_drive_api)
      USE_DRIVE_API=1
      shift
      ;;
    --drive_base_dir)
      DRIVE_BASE_DIR="$2"
      shift 2
      ;;
    --headless)
      HEADLESS=1
      shift
      ;;
    *)
      echo "Unknown option: $1"
      shift
      ;;
  esac
done

# Build the command
CMD="python -m src.data.process_datasets --config $CONFIG"

if [ ! -z "$DATASETS" ]; then
  CMD="$CMD --datasets $DATASETS"
fi

if [ $STREAMING -eq 1 ]; then
  CMD="$CMD --streaming"
fi

if [ $NO_CACHE -eq 1 ]; then
  CMD="$CMD --no_cache"
fi

if [ $USE_DRIVE_API -eq 1 ]; then
  CMD="$CMD --use_drive_api --credentials_path $CREDENTIALS_PATH --drive_base_dir $DRIVE_BASE_DIR"
  
  if [ $HEADLESS -eq 1 ]; then
    CMD="$CMD --headless"
  fi
fi

# Print and execute the command
echo "Running: $CMD"
eval $CMD 