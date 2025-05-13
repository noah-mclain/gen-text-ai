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
pip install langid tqdm

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

# Set up variables
DATA_DIR="data/processed"
OUTPUT_DIR="$DATA_DIR/the_stack_filtered_processed"

# Step 3: Use direct loader approach first (most reliable)
echo -e "${BLUE}Using direct dataset loader to process The Stack...${NC}"
echo -e "${GREEN}This will download and save a sample of Python, Java, and JavaScript code${NC}"

python fix_stack_loader.py

# Check if direct loader succeeded
if [ -d "$OUTPUT_DIR" ]; then
  echo -e "${GREEN}Direct dataset loader succeeded!${NC}"
  
  # Count processed examples
  if [ -f "$OUTPUT_DIR/dataset_info.json" ]; then
    EXAMPLE_COUNT=$(grep -o '"num_examples":[^,]*' "$OUTPUT_DIR/dataset_info.json" | cut -d':' -f2)
    echo -e "${GREEN}Total examples processed: $EXAMPLE_COUNT${NC}"
  fi
  
  echo -e "${BLUE}=================================================${NC}"
  echo -e "${GREEN}Processing complete!${NC}"
  echo -e "${BLUE}=================================================${NC}"
  
  echo -e "${GREEN}To train using the processed dataset:${NC}"
  echo -e "python main_api.py --mode train --datasets the_stack_filtered --training_config config/training_config.json"
  
  exit 0
fi

# Step 4: If direct loader failed, try debug script
echo -e "${YELLOW}Direct loader failed. Running debug script to diagnose issues...${NC}"
python debug_stack_processing.py

# Step 5: Try standard processing as a last resort
echo -e "${BLUE}Attempting standard processing pipeline...${NC}"
echo -e "${GREEN}This may take some time depending on your network and compute resources${NC}"

# Set config option from command line arg if provided
CONFIG=${2:-"config/dataset_config.json"}

# Run the processing command
CMD="python main_api.py --mode process --datasets the_stack_filtered --streaming --no_cache --dataset_config $CONFIG"
echo -e "${BLUE}Running command:${NC} $CMD"
echo -e "${GREEN}Note: Processed data will be saved to $DATA_DIR${NC}"

# Execute the command
eval $CMD
PROCESS_RESULT=$?

# Check if command succeeded
if [ $PROCESS_RESULT -ne 0 ]; then
  echo -e "${RED}Command failed with exit code $PROCESS_RESULT${NC}"
  # Try individual languages as last resort
  echo -e "${YELLOW}Trying individual languages as a last resort...${NC}"
else
  # Check if any data was actually processed
  if [ -d "$OUTPUT_DIR" ]; then
    echo -e "${GREEN}Successfully processed The Stack dataset!${NC}"
    echo -e "${GREEN}Processed data is available at $OUTPUT_DIR${NC}"
    
    # Count processed examples
    if [ -f "$OUTPUT_DIR/dataset_info.json" ]; then
      EXAMPLE_COUNT=$(grep -o '"num_examples":[^,]*' "$OUTPUT_DIR/dataset_info.json" | cut -d':' -f2)
      echo -e "${GREEN}Total examples processed: $EXAMPLE_COUNT${NC}"
    fi
    
    echo -e "${BLUE}=================================================${NC}"
    echo -e "${GREEN}Processing complete!${NC}"
    echo -e "${BLUE}=================================================${NC}"
    
    echo -e "${GREEN}To train using the processed dataset:${NC}"
    echo -e "python main_api.py --mode train --datasets the_stack_filtered --training_config config/training_config.json"
    
    exit 0
  fi
fi

# Step 6: Advanced debugging - try to process each language individually as a last resort
echo -e "${BLUE}Attempting to process individual languages as a fallback...${NC}"
for LANG in python java javascript
do
  echo -e "${GREEN}Processing language: $LANG${NC}"
  python main_api.py --mode process --datasets the_stack_$LANG --streaming --no_cache --dataset_config $CONFIG
  
  if [ -d "$DATA_DIR/the_stack_${LANG}_processed" ]; then
    echo -e "${GREEN}Successfully processed $LANG examples!${NC}"
  fi
done

# Final check - did we get any data?
if [ -d "$OUTPUT_DIR" ] || ls $DATA_DIR/the_stack_*_processed 1> /dev/null 2>&1; then
  echo -e "${GREEN}Successfully processed at least some Stack data!${NC}"
  echo -e "${GREEN}You can now train using:${NC}"
  echo -e "python main_api.py --mode train --datasets the_stack_filtered --training_config config/training_config.json"
else
  echo -e "${RED}All processing attempts failed.${NC}"
  echo -e "${YELLOW}Check that your HF_TOKEN has access to The Stack dataset.${NC}"
  echo -e "${YELLOW}You may need to accept the terms at: https://huggingface.co/datasets/bigcode/the-stack${NC}"
  exit 1
fi

echo -e "${BLUE}=================================================${NC}"
echo -e "${GREEN}Processing complete!${NC}"
echo -e "${BLUE}=================================================${NC}"

exit 0 