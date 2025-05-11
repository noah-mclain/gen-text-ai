#!/bin/bash

# Combined script for processing datasets with authentication support
# This script handles both standard processing and authenticated access to datasets

# Set color codes for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}==========================================${NC}"
echo -e "${GREEN}DeepSeek-Coder Dataset Processing Script${NC}"
echo -e "${BLUE}==========================================${NC}"

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
CONFIG="config/dataset_config.json"
CREDENTIALS_PATH="credentials.json"
STREAMING=true
NO_CACHE=true
USE_DRIVE_API=false
DRIVE_BASE_DIR="DeepseekCoder"
HEADLESS=true
INSTALL_DEPS=false
DATASETS="code_alpaca mbpp codeparrot humaneval"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --credentials)
      CREDENTIALS_PATH="$2"
      shift 2
      ;;
    --datasets)
      shift
      DATASETS=""
      while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
        DATASETS="$DATASETS $1"
        shift
      done
      ;;
    --no-streaming)
      STREAMING=false
      shift
      ;;
    --use-cache)
      NO_CACHE=false
      shift
      ;;
    --drive)
      USE_DRIVE_API=true
      shift
      ;;
    --drive-dir)
      DRIVE_BASE_DIR="$2"
      shift 2
      ;;
    --browser)
      HEADLESS=false
      shift
      ;;
    --install)
      INSTALL_DEPS=true
      shift
      ;;
    *)
      echo -e "${YELLOW}Unknown option: $1 (ignoring)${NC}"
      shift
      ;;
  esac
done

# Install dependencies if requested
if [ "$INSTALL_DEPS" = "true" ]; then
    echo -e "${BLUE}Installing dependencies...${NC}"
    pip install -r requirements.txt
    pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
fi

# Ensure necessary directories exist
mkdir -p data/raw data/processed models results visualizations logs

# Handle Hugging Face authentication
if [ -z "$HF_TOKEN" ]; then
  echo -e "${YELLOW}HF_TOKEN not set in environment, checking credentials...${NC}"
  
  # Try to set token using the Python script
  python scripts/set_hf_token.py --credentials "$CREDENTIALS_PATH"
  
  # Check if token was set
  if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}Warning: HF_TOKEN is not set. Some datasets may be inaccessible.${NC}"
  else
    echo -e "${GREEN}Successfully set HF_TOKEN from credentials${NC}"
  fi
else
  echo -e "${GREEN}HF_TOKEN is already set in environment${NC}"
fi

# Build command
CMD="python main_api.py --mode process --dataset_config $CONFIG"

# Add datasets if specified
if [ -n "$DATASETS" ]; then
  CMD="$CMD --datasets $DATASETS"
fi

# Add streaming and caching options
if [ "$STREAMING" = "true" ]; then
  CMD="$CMD --streaming"
fi

if [ "$NO_CACHE" = "true" ]; then
  CMD="$CMD --no_cache"
fi

# Add Google Drive integration if enabled
if [ "$USE_DRIVE_API" = "true" ]; then
  # Check if credentials file exists
  if [ ! -f "$CREDENTIALS_PATH" ]; then
    echo -e "${RED}Credentials file not found at $CREDENTIALS_PATH${NC}"
    echo -e "${RED}Cannot use Google Drive API without credentials${NC}"
    USE_DRIVE_API=false
  else
    CMD="$CMD --use_drive_api --credentials_path $CREDENTIALS_PATH --drive_base_dir $DRIVE_BASE_DIR"
    
    if [ "$HEADLESS" = "true" ]; then
      CMD="$CMD --headless"
    fi
  fi
fi

# Print the command
echo -e "${BLUE}===================${NC}"
echo -e "${GREEN}Executing:${NC}"
echo -e "${YELLOW}$CMD${NC}"
echo -e "${BLUE}===================${NC}"

# Execute the command
eval "$CMD"

# Check result
if [ $? -eq 0 ]; then
  echo -e "${GREEN}Dataset processing completed successfully!${NC}"
else
  echo -e "${RED}Dataset processing failed with exit code $?${NC}"
  exit 1
fi 