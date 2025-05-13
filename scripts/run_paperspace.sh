#!/bin/bash
# Comprehensive training script for Paperspace A6000 GPUs
# Handles authentication, preprocessing, training, evaluation, and syncing in one go

# Set color codes for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}=================================================${NC}"
echo -e "${GREEN}Optimized ML Training for Paperspace A6000 GPUs${NC}"
echo -e "${BLUE}=================================================${NC}"

# Set environment variables for better performance
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Set DeepSpeed environment variables for A6000 GPUs
export DS_ACCELERATOR=cuda
export DS_OFFLOAD_PARAM=cpu
export DS_OFFLOAD_OPTIMIZER=cpu
export ACCELERATE_USE_DEEPSPEED=true
export ACCELERATE_DEEPSPEED_CONFIG_FILE=ds_config_a6000.json

# Environment and path setup
echo -e "${GREEN}Setting up environment...${NC}"
cd "$(dirname "$0")/.."  # Navigate to project root
PROJECT_ROOT=$(pwd)
echo -e "${GREEN}Working directory: ${PROJECT_ROOT}${NC}"

# Make sure PYTHONPATH includes the project root
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
echo -e "${GREEN}PYTHONPATH: ${PYTHONPATH}${NC}"

# Fix Unsloth import order warning
echo -e "${GREEN}Optimizing imports for Unsloth...${NC}"
cat > import_wrapper.py << EOL
from unsloth import FastLanguageModel
import transformers
import peft
print("✅ Imports properly ordered for optimal performance")
EOL
python import_wrapper.py
rm import_wrapper.py

# Ensure directories exist
mkdir -p data/raw data/processed
mkdir -p models/deepseek-coder-finetune
mkdir -p logs
mkdir -p results

# Check for and set HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
  echo -e "${YELLOW}HF_TOKEN not set, checking for credentials...${NC}"
  # Try to set token using utility script
  if [ -f "scripts/set_hf_token.py" ]; then
    python scripts/set_hf_token.py
    # Re-check if token is now set
  if [ -z "$HF_TOKEN" ]; then
      echo -e "${YELLOW}Warning: HF_TOKEN is not set. Some datasets might be inaccessible.${NC}"
      echo -e "${YELLOW}Set it using: export HF_TOKEN=your_huggingface_token${NC}"
  else
    echo -e "${GREEN}Successfully set HF_TOKEN from credentials${NC}"
  fi
else
    echo -e "${YELLOW}Warning: HF_TOKEN is not set and set_hf_token.py not found.${NC}"
    echo -e "${YELLOW}Set it using: export HF_TOKEN=your_huggingface_token${NC}"
  fi
else
  echo -e "${GREEN}HF_TOKEN is set. Will use for Hugging Face operations.${NC}"
  # Login to Hugging Face
  python -c "import huggingface_hub; import os; huggingface_hub.login(token=os.environ.get('HF_TOKEN', ''))" 2>/dev/null
fi

# Clean cache to free up space
echo -e "${GREEN}Cleaning cache directories...${NC}"
rm -rf ~/.cache/huggingface/datasets/downloads/completed.lock
rm -rf ~/.cache/huggingface/transformers/models--deepseek-ai--deepseek-coder-6.7b-base/snapshots/*/*.safetensors.tmp*

# Calculate training time based on current time (finish before 11pm)
current_hour=$(date +%H | sed 's/^0*//')
# If hour is empty after removing zeros, it was midnight
if [ -z "$current_hour" ]; then
  current_hour=0
fi
# Calculate hours until 11pm
hours_until_11pm=$((23 - 10#$current_hour - 1))
# Ensure minimum 2 hours
if [ $hours_until_11pm -lt 2 ]; then
  hours_until_11pm=2
fi
echo -e "${GREEN}Auto-calculated training time: ${hours_until_11pm} hours${NC}"

# Check for existing token to skip reauth
TOKEN_FILE="$HOME/.drive_token.json"
if [ -f "$TOKEN_FILE" ]; then
  echo -e "${GREEN}Found existing Google Drive token. Will attempt to use it.${NC}"
  # Quick fix for json scope error - recreate token file if needed
  python -c "
import json, os
try:
  token_path = os.path.expanduser('~/.drive_token.json')
  with open(token_path, 'r') as f:
    data = json.load(f)
  # Save back to ensure proper format
  with open(token_path, 'w') as f:
    json.dump(data, f)
  print('Token file looks good')
except Exception as e:
  print(f'Error with token file: {e}')
"
else
  echo -e "${YELLOW}No Google Drive token found. Will need to authenticate.${NC}"
fi

# Set up Google Drive once at the beginning
echo -e "${BLUE}Setting up Google Drive authentication...${NC}"
python scripts/setup_google_drive.py
if [ $? -ne 0 ]; then
  echo -e "${YELLOW}Google Drive authentication failed. Will proceed without Drive integration.${NC}"
  USE_DRIVE=false
  DRIVE_OPTS=""
else
  echo -e "${GREEN}Google Drive authentication successful. Will use Drive for storage.${NC}"
  USE_DRIVE=true
  DRIVE_OPTS="--use_drive --drive_base_dir DeepseekCoder"
fi

# Pre-download datasets (faster startup)
echo -e "${BLUE}Pre-downloading key datasets...${NC}"
python -c "
try:
    from datasets import load_dataset
    print('Loading CodeSearchNet...')
    load_dataset('code-search-net/code_search_net', split='train', streaming=True, trust_remote_code=True)
    print('Loading CodeAlpaca...')
    load_dataset('sahil2801/CodeAlpaca-20k', split='train', streaming=True, trust_remote_code=True)
    print('Loading MBPP...')
    load_dataset('mbpp', split='train', streaming=True, trust_remote_code=True)
    print('Loading CodeParrot...')
    load_dataset('codeparrot/codeparrot-clean', split='train', streaming=True, trust_remote_code=True)
    print('Loading HumanEval...')
    load_dataset('openai/openai_humaneval', streaming=True, trust_remote_code=True)
    print('Loading Magicoder...')
    load_dataset('ise-uiuc/Magicoder-OSS-Instruct-75K', split='train', streaming=True, trust_remote_code=True)
    print('Dataset pre-loading complete!')
except Exception as e:
    print(f'Error pre-loading datasets: {e}')
"

# Update training config to fix issues
echo -e "${BLUE}Fixing training configuration...${NC}"
python - <<END
import json
import os

try:
    # Load training config
    config_path = "config/training_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 1. Ensure max_steps is set for streaming datasets
    if "max_steps" not in config.get("training", {}):
        if "training" not in config:
            config["training"] = {}
        # Calculate reasonable number of steps (approximately 3 epochs worth)
        config["training"]["max_steps"] = 100000
        print(f"✓ Set max_steps to {config['training']['max_steps']} for streaming datasets")
    
    # 2. Fix dropout for Unsloth optimization
    if "lora_config" in config and "dropout" in config["lora_config"]:
        if config["lora_config"]["dropout"] > 0:
            print(f"⚠️ Found dropout={config['lora_config']['dropout']} - setting to 0 for Unsloth optimization")
            config["lora_config"]["dropout"] = 0
    
    # 3. Set other optimizations
    # Set max training time
    config["training"]["max_train_time_hours"] = $hours_until_11pm
    
    # Enable FP16 for A6000
    config["training"]["fp16"] = True
    
    # Write back updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✓ Successfully updated training config with all optimizations")
except Exception as e:
    print(f"Error updating config: {e}")
END

# Process datasets with fixed imports
echo -e "${BLUE}Processing datasets...${NC}"
python -m src.data.process_datasets \
    --config config/dataset_config.json \
    --output_dir data/processed \
    --datasets the_stack_filtered codesearchnet_all code_alpaca humaneval mbpp codeparrot instruct_code \
    --streaming \
    --no_cache \
    $DRIVE_OPTS \
    2>&1 | tee logs/dataset_processing_$(date +%Y%m%d_%H%M%S).log

# Check if dataset processing was successful
DATASET_STATUS=$?
if [ $DATASET_STATUS -ne 0 ]; then
    echo -e "${YELLOW}Dataset processing had some issues (exit code: $DATASET_STATUS).${NC}"
    echo -e "${YELLOW}Will continue with training using available datasets.${NC}"
fi

# Start training
echo -e "${BLUE}===================${NC}"
echo -e "${GREEN}Starting optimized training...${NC}"
echo -e "${BLUE}===================${NC}"
python -m src.training.train \
    --config config/training_config.json \
    --data_dir data/processed \
    $DRIVE_OPTS \
    --push_to_hub \
    --debug \
    2>&1 | tee logs/train_a6000_$(date +%Y%m%d_%H%M%S).log

# Check training result
TRAIN_STATUS=$?
if [ $TRAIN_STATUS -eq 0 ]; then
  echo -e "${GREEN}Training completed successfully!${NC}"
  
  # Create completion marker
  echo "Training completed at $(date)" > logs/training_complete_$(date +%Y%m%d_%H%M%S).txt
  
  # Sync results to Drive if available
  if [ "$USE_DRIVE" = true ]; then
    echo -e "${BLUE}Syncing results to Google Drive...${NC}"
    python scripts/sync_to_drive.py --sync-all
  fi
  
  # Run evaluation
  echo -e "${BLUE}Running model evaluation...${NC}"
  python -m src.evaluation.evaluate \
      --model_path models/deepseek-coder-finetune/final \
      --benchmark humaneval,mbpp \
      2>&1 | tee logs/eval_$(date +%Y%m%d_%H%M%S).log
  
  EVAL_STATUS=$?
  if [ $EVAL_STATUS -eq 0 ]; then
    echo -e "${GREEN}Evaluation completed successfully!${NC}"
  else
    echo -e "${YELLOW}Evaluation encountered issues (exit code: $EVAL_STATUS).${NC}"
  fi
  
  # Run visualization if available
  if [ -f "src/visualization/visualize_results.py" ]; then
    echo -e "${BLUE}Generating visualizations...${NC}"
    python -m src.visualization.visualize_results \
        --results_dir results \
        2>&1 | tee logs/visualize_$(date +%Y%m%d_%H%M%S).log
  fi
  
  # Final sync to Drive
  if [ "$USE_DRIVE" = true ]; then
    echo -e "${BLUE}Final sync of all results to Google Drive...${NC}"
    python scripts/sync_to_drive.py --sync-all
  fi
else
  echo -e "${RED}Training failed with exit code $TRAIN_STATUS.${NC}"
  echo -e "${YELLOW}Check the logs for details.${NC}"
fi

echo -e "${BLUE}===================${NC}"
echo -e "${GREEN}Process complete at $(date)${NC}"
echo -e "${BLUE}===================${NC}" 