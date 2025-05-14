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
echo -e "${GREEN}Configured for maximum efficiency and interrupt tolerance${NC}"
echo -e "${BLUE}=================================================${NC}"

# Create checkpoint directory if it doesn't exist
mkdir -p /notebooks/models/checkpoints
mkdir -p /notebooks/logs

# Fix DeepSpeed configuration - ensure this runs before anything else
echo -e "${GREEN}Fixing DeepSpeed configuration...${NC}"
python scripts/fix_deepspeed.py
if [ $? -ne 0 ]; then
  echo -e "${YELLOW}Warning: DeepSpeed config fix script failed. Will try to continue...${NC}"
fi

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
export ACCELERATE_DEEPSPEED_CONFIG_FILE=/notebooks/ds_config_a6000.json

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

# Check for existing checkpoints to resume from
echo -e "${BLUE}Checking for existing checkpoints...${NC}"
OUTPUT_DIR="/notebooks/models/deepseek-coder-finetune"
CHECKPOINT_PATH=""

# Function to find the latest checkpoint folder
find_latest_checkpoint() {
  local dir=$1
  local latest_step=0
  local latest_folder=""
  
  # Look for checkpoint folders
  for folder in "$dir"/checkpoint-*; do
    if [ -d "$folder" ]; then
      # Extract step number
      step=$(echo "$folder" | grep -oE 'checkpoint-([0-9]+)' | cut -d'-' -f2)
      if [ "$step" -gt "$latest_step" ]; then
        latest_step=$step
        latest_folder=$folder
      fi
    fi
  done
  
  echo "$latest_folder"
}

# Try to download latest checkpoint from Drive if available
if [ "$USE_DRIVE" = true ]; then
  echo -e "${BLUE}Checking for checkpoints on Google Drive...${NC}"
  python scripts/sync_to_drive.py --sync-from-drive checkpoints/deepseek-coder-finetune "$OUTPUT_DIR" --no-cleanup
  
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}Successfully checked for Drive checkpoints${NC}"
  else
    echo -e "${YELLOW}Warning: Could not sync checkpoints from Drive${NC}"
  fi
fi

# Find the latest checkpoint
CHECKPOINT_PATH=$(find_latest_checkpoint "$OUTPUT_DIR")

# Resume training options
RESUME_OPTS=""
if [ -n "$CHECKPOINT_PATH" ]; then
  echo -e "${GREEN}Found checkpoint to resume from: $CHECKPOINT_PATH${NC}"
  RESUME_OPTS="--resume_from_checkpoint $CHECKPOINT_PATH"
  # Get the step number for logging
  RESUME_STEP=$(echo "$CHECKPOINT_PATH" | grep -oE 'checkpoint-([0-9]+)' | cut -d'-' -f2)
  echo -e "${GREEN}Will resume training from step $RESUME_STEP${NC}"
else
  echo -e "${YELLOW}No existing checkpoints found. Starting training from scratch.${NC}"
fi

# Pre-download datasets (faster startup)
echo -e "${BLUE}Pre-downloading key datasets...${NC}"
# Load datasets directly from the configuration file
echo -e "${GREEN}Loading datasets from configuration...${NC}"
python -c "
import json
import logging
from datasets import load_dataset
import time
import os

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('dataset_preloader')

# Load the dataset config
try:
    with open('config/dataset_config.json', 'r') as f:
        dataset_config = json.load(f)
    logger.info(f'Successfully loaded dataset config with {len(dataset_config)} datasets')
    
    # Track statistics
    success_count = 0
    
    # Load each dataset
    for name, info in dataset_config.items():
        if 'path' not in info or info.get('enabled') is False:
            continue
            
        path = info['path']
        split = info.get('split', 'train')
        
        # Skip very large datasets
        if 'the-stack' in path:
            logger.info(f'Skipping large dataset {name} ({path})')
            continue
            
        logger.info(f'Loading dataset {name} from {path}...')
        try:
            # Try with split
            if split:
                load_dataset(path, split=split, streaming=True, trust_remote_code=True)
            else:
                load_dataset(path, streaming=True, trust_remote_code=True)
                
            logger.info(f'✓ Successfully loaded {name}')
            success_count += 1
            time.sleep(1)  # Short delay between loads
        except Exception as e:
            logger.error(f'Error loading {name}: {e}')
    
    logger.info(f'Successfully preloaded {success_count} datasets')
    
except Exception as e:
    logger.error(f'Error in dataset preloading: {e}')
"

# Ensure processed data directory exists
mkdir -p data/processed

# Sync datasets from Drive and verify their integrity
echo -e "${BLUE}Syncing datasets from Drive...${NC}"
python scripts/save_datasets.py --download --verify_integrity --repair --process_missing --drive_base_dir DeepseekCoder
if [ $? -ne 0 ]; then
  echo -e "${YELLOW}Warning: Dataset sync had some issues. Will continue with available datasets.${NC}"
fi

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
    
    # Enable resume from checkpoint
    config["training"]["resume_from_checkpoint"] = True
    
    # Decrease checkpoint frequency for better performance
    config["training"]["save_steps"] = 100
    config["training"]["save_total_limit"] = 5
    
    # Add save_safetensors if not present
    if "save_safetensors" not in config["training"]:
        config["training"]["save_safetensors"] = True
    
    # Write back updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✓ Successfully updated training config with all optimizations")
except Exception as e:
    print(f"Error updating config: {e}")
END

# Start training
echo -e "${BLUE}===================${NC}"
echo -e "${GREEN}Starting optimized training...${NC}"
echo -e "${BLUE}===================${NC}"

# Create a trap to handle interruptions
trap "echo -e '${YELLOW}Training interrupted! Saving checkpoint...${NC}'; \
      if [ "$USE_DRIVE" = true ]; then \
        echo -e '${BLUE}Syncing checkpoints to Drive...${NC}'; \
        python scripts/sync_to_drive.py --sync-checkpoints '$OUTPUT_DIR' --model-name deepseek-coder-finetune; \
      fi" SIGINT SIGTERM

# Start the training
python -m src.training.train \
  --config config/training_config.json \
  --data_dir data/processed \
  $RESUME_OPTS \
  $DRIVE_OPTS \
  2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log

TRAIN_STATUS=$?

# Sync final checkpoint to Drive regardless of completion status
if [ "$USE_DRIVE" = true ]; then
  echo -e "${BLUE}Syncing final checkpoints to Drive...${NC}"
  python scripts/sync_to_drive.py --sync-checkpoints "$OUTPUT_DIR" --model-name deepseek-coder-finetune
fi

if [ $TRAIN_STATUS -eq 0 ]; then
  echo -e "${GREEN}Training completed successfully!${NC}"
  
  # Create completion marker
  echo "Training completed at $(date)" > logs/training_complete_$(date +%Y%m%d_%H%M%S).txt
  
  # Run evaluation
  echo -e "${BLUE}Running model evaluation...${NC}"
  PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH} python -c "
import os
import sys
import subprocess
from pathlib import Path

# Set proper paths
model_path = 'models/deepseek-coder-finetune/final'
output_dir = 'results'

# Add the project root to sys.path to ensure imports work
project_root = Path('${PROJECT_ROOT}')
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Try to import the evaluator directly
try:
    from src.evaluation.evaluator import ModelEvaluator
    print('Successfully imported ModelEvaluator')
except Exception as e:
    print(f'Warning: Could not import evaluator directly: {e}')

# Ensure the evaluation directory exists
os.makedirs(output_dir, exist_ok=True)

# Run the evaluation script as a subprocess with explicit PYTHONPATH
cmd = [
    sys.executable, 
    '-m', 
    'src.evaluation.evaluate',
    '--model_path', model_path,
    '--output_dir', output_dir,
    '--eval_humaneval',
    '--eval_mbpp'
]

print(f'Running evaluation command: {\" \".join(cmd)}')
result = subprocess.run(cmd, env=os.environ)
sys.exit(result.returncode)
" 2>&1 | tee logs/eval_$(date +%Y%m%d_%H%M%S).log

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
else
  echo -e "${YELLOW}Training ended with status code $TRAIN_STATUS. Checkpoints were saved.${NC}"
  echo -e "${YELLOW}You can resume training by running this script again.${NC}"
fi

echo -e "${BLUE}===================${NC}"
echo -e "${GREEN}Done!${NC}"
echo -e "${BLUE}===================${NC}" 