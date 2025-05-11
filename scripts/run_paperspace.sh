#!/bin/bash

# Combined script for running the deepseek-coder fine-tuning pipeline on Paperspace
# Includes authentication for HF_TOKEN and Google Drive API integration

# Set up error handling
set -e  # Exit immediately if a command fails
trap 'echo "Error occurred. Exiting..."; exit 1' ERR

# ANSI color codes for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================================${NC}"
echo -e "${GREEN}DeepSeek-Coder Fine-Tuning Pipeline - Paperspace Edition${NC}"
echo -e "${BLUE}=========================================================${NC}"

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
MODE="all"
DATASET_CONFIG="config/dataset_config.json"
TRAINING_CONFIG="config/training_config.json"
CREDENTIALS_PATH="credentials.json"
DRIVE_BASE_DIR="DeepseekCoder"
STREAMING=true
NO_CACHE=true
DATASETS="code_alpaca mbpp codeparrot humaneval"
BASE_MODEL="deepseek-ai/deepseek-coder-6.7b-base"
MODEL_PATH="models/deepseek-coder-finetune"
SKIP_INSTALL=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --dataset-config)
      DATASET_CONFIG="$2"
      shift 2
      ;;
    --training-config)
      TRAINING_CONFIG="$2"
      shift 2
      ;;
    --credentials)
      CREDENTIALS_PATH="$2"
      shift 2
      ;;
    --drive-dir)
      DRIVE_BASE_DIR="$2"
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
    --base-model)
      BASE_MODEL="$2"
      shift 2
      ;;
    --model-path)
      MODEL_PATH="$2"
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
    --skip-install)
      SKIP_INSTALL=true
      shift
      ;;
    *)
      echo -e "${YELLOW}Unknown option: $1 (ignoring)${NC}"
      shift
      ;;
  esac
done

# Install dependencies unless skipped
if [ "$SKIP_INSTALL" = "false" ]; then
    echo -e "${BLUE}Installing dependencies...${NC}"
    pip install -r requirements.txt
    pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
fi

# Ensure necessary directories exist
mkdir -p data/raw data/processed models results visualizations logs

# Set up authentication for HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
  echo -e "${YELLOW}HF_TOKEN not set in environment, trying to set from credentials...${NC}"
  
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

# Verify Google Drive credentials
if [ ! -f "$CREDENTIALS_PATH" ]; then
  echo -e "${RED}Error: credentials.json file not found at $CREDENTIALS_PATH${NC}"
  echo -e "${YELLOW}Google Drive API integration will be disabled${NC}"
  echo -e "${YELLOW}Please download your credentials.json file from the Google Cloud Console:${NC}"
  echo -e "${YELLOW}1. Go to https://console.cloud.google.com/${NC}"
  echo -e "${YELLOW}2. Create a new project if you don't have one${NC}"
  echo -e "${YELLOW}3. Enable the Google Drive API${NC}"
  echo -e "${YELLOW}4. Create OAuth 2.0 credentials${NC}"
  echo -e "${YELLOW}5. Download the credentials JSON file and rename it to credentials.json${NC}"
  USE_DRIVE_API=false
else
  echo -e "${GREEN}Google Drive credentials found at $CREDENTIALS_PATH${NC}"
  USE_DRIVE_API=true
  
  # Authenticate with Google Drive API
  echo -e "${BLUE}Authenticating with Google Drive API...${NC}"
  python scripts/authenticate_headless.py --credentials "$CREDENTIALS_PATH"
fi

# Run different stages based on mode
if [ "$MODE" = "all" ] || [ "$MODE" = "process" ]; then
  echo -e "${BLUE}============================${NC}"
  echo -e "${GREEN}Processing datasets...${NC}"
  echo -e "${BLUE}============================${NC}"
  
  process_cmd="python main_api.py --mode process --dataset_config $DATASET_CONFIG --datasets $DATASETS"
  
  if [ "$STREAMING" = "true" ]; then
    process_cmd="$process_cmd --streaming"
  fi
  
  if [ "$NO_CACHE" = "true" ]; then
    process_cmd="$process_cmd --no_cache"
  fi
  
  if [ "$USE_DRIVE_API" = "true" ]; then
    process_cmd="$process_cmd --use_drive_api --credentials_path $CREDENTIALS_PATH --drive_base_dir $DRIVE_BASE_DIR --headless"
  fi
  
  echo -e "${YELLOW}Executing: $process_cmd${NC}"
  eval "$process_cmd"
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "train" ]; then
  echo -e "${BLUE}============================${NC}"
  echo -e "${GREEN}Training model...${NC}"
  echo -e "${BLUE}============================${NC}"
  
  train_cmd="python main_api.py --mode train --training_config $TRAINING_CONFIG"
  
  if [ "$USE_DRIVE_API" = "true" ]; then
    train_cmd="$train_cmd --use_drive_api --credentials_path $CREDENTIALS_PATH --drive_base_dir $DRIVE_BASE_DIR --headless"
  fi
  
  echo -e "${YELLOW}Executing: $train_cmd${NC}"
  eval "$train_cmd"
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "evaluate" ]; then
  echo -e "${BLUE}============================${NC}"
  echo -e "${GREEN}Evaluating model...${NC}"
  echo -e "${BLUE}============================${NC}"
  
  eval_cmd="python main_api.py --mode evaluate --model_path $MODEL_PATH --base_model $BASE_MODEL"
  
  if [ "$USE_DRIVE_API" = "true" ]; then
    eval_cmd="$eval_cmd --use_drive_api --credentials_path $CREDENTIALS_PATH --drive_base_dir $DRIVE_BASE_DIR --headless"
  fi
  
  echo -e "${YELLOW}Executing: $eval_cmd${NC}"
  eval "$eval_cmd"
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "visualize" ]; then
  echo -e "${BLUE}============================${NC}"
  echo -e "${GREEN}Generating visualizations...${NC}"
  echo -e "${BLUE}============================${NC}"
  
  vis_cmd="python main_api.py --mode visualize"
  
  if [ "$USE_DRIVE_API" = "true" ]; then
    vis_cmd="$vis_cmd --use_drive_api --credentials_path $CREDENTIALS_PATH --drive_base_dir $DRIVE_BASE_DIR --headless"
  fi
  
  echo -e "${YELLOW}Executing: $vis_cmd${NC}"
  eval "$vis_cmd"
fi

echo -e "${BLUE}==========================================${NC}"
echo -e "${GREEN}Pipeline completed successfully!${NC}"

if [ "$USE_DRIVE_API" = "true" ]; then
  echo -e "${GREEN}All data has been saved to Google Drive under $DRIVE_BASE_DIR${NC}"
  echo -e "${GREEN}You can access your model, results, and visualizations there.${NC}"
else
  echo -e "${GREEN}All data has been saved locally in the appropriate directories.${NC}"
fi 