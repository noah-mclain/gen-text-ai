#!/bin/bash
# train_deepseek_coder.sh - Script to train DeepSeek Coder with Google Drive integration

# Color definitions for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}   DeepSeek Coder Training Script     ${NC}"
echo -e "${BLUE}=======================================${NC}"

# Check for credentials.json file
if [ ! -f "credentials.json" ]; then
    echo -e "${YELLOW}Google Drive credentials not found.${NC}"
    echo -e "${YELLOW}You can still run training without Drive integration.${NC}"
    USE_DRIVE=false
else
    echo -e "${GREEN}Google Drive credentials found.${NC}"
    
    # Test if we can authenticate
    python -c "from src.utils.google_drive_manager import test_authentication; print('Authentication successful' if test_authentication() else 'Authentication failed')" > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Google Drive authentication successful.${NC}"
        USE_DRIVE=true
    else
        echo -e "${YELLOW}Google Drive authentication failed. Will continue without Drive integration.${NC}"
        USE_DRIVE=false
    fi
fi

# Check for environment variables
echo -e "\n${BLUE}Checking environment...${NC}"

# Check if CUDA is available via PyTorch
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'Device count: {torch.cuda.device_count()}')"
python -c "import torch; print(f'Current device: {torch.cuda.current_device()}')"
python -c "import torch; print(f'Device name: {torch.cuda.get_device_name(0)}')"

# Check for HF_TOKEN
if [ -z "${HF_TOKEN}" ]; then
    echo -e "${YELLOW}HF_TOKEN not set. Some datasets might be inaccessible.${NC}"
    echo -e "${YELLOW}Set it with: export HF_TOKEN=your_huggingface_token${NC}"
    PUSH_TO_HUB=false
else
    echo -e "${GREEN}HF_TOKEN is set. Will be able to access gated datasets.${NC}"
    PUSH_TO_HUB=true
    
    # Check if HF_HUB_MODEL_ID is set for pushing to HF Hub
    if [ -z "${HF_HUB_MODEL_ID}" ]; then
        echo -e "${YELLOW}HF_HUB_MODEL_ID not set. Model will not be pushed to Hub.${NC}"
        echo -e "${YELLOW}Set it with: export HF_HUB_MODEL_ID=your-username/model-name${NC}"
        PUSH_TO_HUB=false
    else
        echo -e "${GREEN}Will push model to Hugging Face Hub as: ${HF_HUB_MODEL_ID}${NC}"
    fi
fi

# Check if we want to skip preprocessing
if [ "$1" == "--skip-preprocessing" ]; then
    SKIP_PREPROCESSING=true
    shift
else
    SKIP_PREPROCESSING=false
fi

# Process datasets first
if [ "$SKIP_PREPROCESSING" = false ]; then
    echo -e "\n${BLUE}Processing datasets...${NC}"
    
    # Handle Drive integration for preprocessing if available
    if [ "$USE_DRIVE" = true ]; then
        python main_api.py --mode process --streaming --use_drive --headless
    else
        python main_api.py --mode process --streaming
    fi
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error processing datasets. Check the logs for details.${NC}"
        exit 1
    fi
else
    echo -e "\n${YELLOW}Skipping preprocessing as requested.${NC}"
fi

# Determine which config to use based on available GPU memory
VRAM=$(python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024)")
VRAM_INT=$(python -c "import torch; print(int(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024))")

echo -e "\n${BLUE}Detected GPU with ${VRAM_INT}GB VRAM${NC}"

# Select configuration based on available VRAM
if [ $VRAM_INT -ge 40 ]; then
    CONFIG_PATH="config/training_config.json"
    echo -e "${GREEN}Using standard training configuration${NC}"
elif [ $VRAM_INT -ge 24 ]; then
    CONFIG_PATH="config/training_config_a6000.json"
    echo -e "${GREEN}Using A6000-optimized configuration${NC}"
else
    CONFIG_PATH="config/training_config_low_memory.json"
    echo -e "${YELLOW}Using low-memory training configuration${NC}"
    
    # If the config doesn't exist, create it
    if [ ! -f "$CONFIG_PATH" ]; then
        echo -e "${YELLOW}Creating low-memory config from standard config...${NC}"
        cp config/training_config.json "$CONFIG_PATH"
        
        # Modify the config to use less memory
        python -c "
import json
with open('$CONFIG_PATH', 'r') as f:
    config = json.load(f)
if 'training' in config:
    config['training']['per_device_train_batch_size'] = 1
    config['training']['gradient_accumulation_steps'] = 16
    config['training']['fp16'] = True
    config['training']['deepspeed'] = 'config/ds_config_zero3.json'
if 'dataset' in config:
    if 'max_samples' in config['dataset']:
        for key in config['dataset']['max_samples']:
            config['dataset']['max_samples'][key] = min(5000, config['dataset']['max_samples'].get(key, 5000))
with open('$CONFIG_PATH', 'w') as f:
    json.dump(config, f, indent=2)
print('Low-memory configuration created')
        "
    fi
fi

# Train the model
echo -e "\n${BLUE}Starting model training...${NC}"

# Build the training command
TRAIN_CMD="python main_api.py --mode train --training_config $CONFIG_PATH"

# Add Drive integration if available
if [ "$USE_DRIVE" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --use_drive --headless"
fi

# Add HF Hub integration if available
if [ "$PUSH_TO_HUB" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --push_to_hub --hub_model_id $HF_HUB_MODEL_ID"
fi

# Run the training command
echo -e "${GREEN}Running: $TRAIN_CMD${NC}"
eval $TRAIN_CMD

# Check training success
if [ $? -ne 0 ]; then
    echo -e "${RED}Error during training. Check the logs for details.${NC}"
    exit 1
fi

echo -e "\n${GREEN}Training completed successfully!${NC}"

# Upload models to Google Drive if enabled
if [ "$USE_DRIVE" = true ]; then
    echo -e "\n${BLUE}Syncing results to Google Drive...${NC}"
    python -c "from src.utils.google_drive_manager import sync_to_drive; sync_to_drive('models/deepseek-coder-finetune', 'models')"
    python -c "from src.utils.google_drive_manager import sync_to_drive; sync_to_drive('results', 'results')"
    python -c "from src.utils.google_drive_manager import sync_to_drive; sync_to_drive('logs', 'logs')"
    echo -e "${GREEN}Sync completed.${NC}"
fi

echo -e "\n${BLUE}=======================================${NC}"
echo -e "${BLUE}   Training Pipeline Completed         ${NC}"
echo -e "${BLUE}=======================================${NC}"

# If HF Hub integration was used, show the model URL
if [ "$PUSH_TO_HUB" = true ]; then
    echo -e "${GREEN}Model pushed to HuggingFace Hub: https://huggingface.co/$HF_HUB_MODEL_ID${NC}"
fi

exit 0 