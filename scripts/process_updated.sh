#!/bin/bash

# Process datasets with the improved configuration and preprocessing logic
# This script focuses on the working datasets and includes all recent fixes

# Set color codes for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================================${NC}"
echo -e "${GREEN}DeepSeek-Coder Fixed Dataset Processing Script${NC}"
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

# Default configuration - only using datasets that work well
CONFIG="config/dataset_config.json"
CREDENTIALS_PATH="credentials.json"
DATASETS="code_alpaca mbpp codeparrot humaneval"
USE_DRIVE_API=false
DRIVE_BASE_DIR="DeepseekCoder"
HEADLESS=true

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
    --help)
      echo -e "${GREEN}Usage: $0 [options]${NC}"
      echo -e "${YELLOW}Options:${NC}"
      echo "  --config FILE       Path to dataset config (default: config/dataset_config.json)"
      echo "  --credentials FILE  Path to Google API credentials (default: credentials.json)"
      echo "  --datasets D1 D2..  Specific datasets to process (default: code_alpaca mbpp codeparrot humaneval)"
      echo "  --drive             Enable Google Drive integration"
      echo "  --drive-dir DIR     Google Drive base directory (default: DeepseekCoder)"
      echo "  --browser           Use browser auth instead of headless"
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

# Build command with all our fixes
CMD="python main_api.py --mode process --dataset_config $CONFIG --datasets $DATASETS --streaming --no_cache"

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
  echo -e "${GREEN}All bug fixes applied. The following improvements were made:${NC}"
  echo -e "${BLUE}- Fixed batch size calculation to prevent division by zero${NC}"
  echo -e "${BLUE}- Added handling for empty datasets${NC}"
  echo -e "${BLUE}- Corrected dataset paths on Hugging Face${NC}"
  echo -e "${BLUE}- Added 'enabled' flag support in dataset configuration${NC}"
  echo -e "${BLUE}- Enhanced Google Drive integration to prevent duplicates${NC}"
else
  echo -e "${RED}Dataset processing failed with exit code $?${NC}"
  exit 1
fi 