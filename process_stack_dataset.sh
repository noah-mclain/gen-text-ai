#!/bin/bash

# Process The Stack dataset with proper language filtering
# This script fixes the language detection and handles The Stack dataset correctly

# Set color codes for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================================${NC}"
echo -e "${GREEN}Processing The Stack Dataset with Language Filtering${NC}"
echo -e "${BLUE}=================================================${NC}"

# Step 1: Install language detection dependencies
echo -e "${BLUE}Installing language detection dependencies...${NC}"
pip install langid

# Step 2: Check/Set HF_TOKEN
if [ -n "$HF_TOKEN" ]; then
  echo -e "${GREEN}Using HF_TOKEN from environment${NC}"
elif [ -n "$1" ]; then
  echo -e "${GREEN}Setting HF_TOKEN from command line argument${NC}"
  export HF_TOKEN="$1"
else
  echo -e "${YELLOW}No Hugging Face token provided. If access fails, either:${NC}"
  echo -e "${YELLOW}  1. Run: export HF_TOKEN=your_token${NC}"
  echo -e "${YELLOW}  2. Or run this script with your token: ./process_stack_dataset.sh YOUR_HF_TOKEN${NC}"
fi

# Step 3: Process the dataset
echo -e "${BLUE}Processing The Stack filtered dataset...${NC}"
echo -e "${GREEN}This may take some time depending on your network and compute resources${NC}"

# Set config option from command line arg if provided
CONFIG=${2:-"config/dataset_config.json"}

# Run the processing command - note: the processed data will be saved to data/processed by default
CMD="python main_api.py --mode process --datasets the_stack_filtered --streaming --no_cache --dataset_config $CONFIG"
echo -e "${BLUE}Running command:${NC} $CMD"
echo -e "${GREEN}Note: Processed data will be saved to data/processed by default${NC}"

# Execute the command
eval $CMD

# Check result
if [ $? -eq 0 ]; then
  echo -e "${GREEN}Successfully processed The Stack dataset!${NC}"
  echo -e "${GREEN}Processed data should be available in data/processed${NC}"
else
  echo -e "${RED}Error processing The Stack dataset.${NC}"
  echo -e "${YELLOW}Check logs above for error details.${NC}"
  exit 1
fi

echo -e "${BLUE}=================================================${NC}"
echo -e "${GREEN}Processing complete!${NC}"
echo -e "${BLUE}=================================================${NC}"

# Print some helpful usage tips
echo -e "${GREEN}To train using the processed dataset:${NC}"
echo -e "python main_api.py --mode train --datasets the_stack_filtered --training_config config/training_config.json"

exit 0 