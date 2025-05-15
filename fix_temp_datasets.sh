#!/bin/bash
# Script to create and use a visible non-temporary directory for dataset processing

# Set bash to exit immediately if a command exits with a non-zero status
set -e

# ANSI color codes for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create a dated temp directory name
DATE_SUFFIX=$(date +"%Y-%m-%d")
TEMP_DIR="$SCRIPT_DIR/temp_datasets_$DATE_SUFFIX"

echo -e "${BOLD}${BLUE}=========================================${NC}"
echo -e "${BOLD}${BLUE} Dataset Processing with Persistent Storage ${NC}"
echo -e "${BOLD}${BLUE}=========================================${NC}"
echo ""

# Create the persistent directory
mkdir -p "$TEMP_DIR"
mkdir -p "$TEMP_DIR/processed"

echo -e "${GREEN}Created persistent dataset directory:${NC} $TEMP_DIR"

# Set environment variables
export TMPDIR="$TEMP_DIR"
export TEMP="$TEMP_DIR"
export TMP="$TEMP_DIR"

# Create a .env file for later use
echo "# Created by fix_temp_datasets.sh" > "$SCRIPT_DIR/.env"
echo "# $(date)" >> "$SCRIPT_DIR/.env"
echo "TMPDIR=\"$TEMP_DIR\"" >> "$SCRIPT_DIR/.env"
echo "TEMP=\"$TEMP_DIR\"" >> "$SCRIPT_DIR/.env"
echo "TMP=\"$TEMP_DIR\"" >> "$SCRIPT_DIR/.env"

echo -e "${GREEN}Set environment variables and created .env file${NC}"
echo ""

# Default datasets to process
DATASETS="code_alpaca mbpp humaneval codeparrot"

# See if user wants to specify datasets
echo -e "${YELLOW}Enter datasets to process (space-separated) or press Enter to use defaults:${NC}"
echo -e "${BLUE}Default: $DATASETS${NC}"
read -r USER_DATASETS

# Use user input if provided
if [ -n "$USER_DATASETS" ]; then
  DATASETS="$USER_DATASETS"
fi

# Process datasets using our fixed code with the persistent directory
echo -e "${BOLD}${GREEN}Processing datasets using persistent directory${NC}"
echo -e "${BLUE}Datasets: $DATASETS${NC}"
echo -e "${BLUE}Temp directory: $TEMP_DIR${NC}"
echo ""

# Run processing with the temp_dir parameter
python src/main_api.py --mode process --datasets $DATASETS --temp_dir "$TEMP_DIR" --streaming

echo ""
echo -e "${BOLD}${GREEN}Processing completed!${NC}"
echo -e "Your datasets are stored in: ${BLUE}$TEMP_DIR/processed${NC}"
echo ""

# Ask if user wants to sync to Google Drive
echo -e "${YELLOW}Would you like to sync these datasets to Google Drive? [y/N]${NC}"
read -r SYNC_CHOICE

if [[ "$SYNC_CHOICE" == "y" || "$SYNC_CHOICE" == "Y" ]]; then
  echo -e "${BOLD}${GREEN}Syncing datasets to Google Drive${NC}"
  
  # Run sync_processed_datasets.py with the temp directory
  python scripts/google_drive/sync_processed_datasets.py --data-dir "$TEMP_DIR/processed"
  
  echo -e "${GREEN}Sync completed!${NC}"
fi

echo ""
echo -e "${BOLD}${GREEN}All tasks completed!${NC}"
echo -e "You can run this script again to process more datasets."
echo "" 