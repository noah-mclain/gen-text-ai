#!/bin/bash
# Process datasets and sync them to Google Drive using the fixed codebase
# This script creates a persistent directory for datasets and ensures they're properly synced

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

echo -e "${BOLD}${BLUE}=========================================${NC}"
echo -e "${BOLD}${BLUE} Dataset Processing and Google Drive Sync ${NC}"
echo -e "${BOLD}${BLUE}=========================================${NC}"
echo ""

# Function to show usage
show_usage() {
    echo -e "${YELLOW}Usage:${NC} $0 [options]"
    echo ""
    echo "Options:"
    echo "  --help, -h            Show this help message"
    echo "  --datasets LIST       Space-separated list of datasets to process (in quotes)"
    echo "  --config FILE         Path to the dataset config file"
    echo "  --drive-dir DIR       Google Drive folder to sync to"
    echo "  --streaming           Use streaming mode for datasets"
    echo "  --no-sync             Skip Google Drive sync step"
    echo ""
    echo "Examples:"
    echo "  $0 --datasets \"code_alpaca mbpp humaneval\""
    echo "  $0 --config config/dataset_config.json --streaming"
    echo ""
}

# Default values
DATASETS="code_alpaca mbpp humaneval codeparrot"
CONFIG_PATH="config/dataset_config.json"
DRIVE_DIR="processed_data"
USE_STREAMING=true
DO_SYNC=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_usage
            exit 0
            ;;
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --drive-dir)
            DRIVE_DIR="$2"
            shift 2
            ;;
        --streaming)
            USE_STREAMING=true
            shift
            ;;
        --no-sync)
            DO_SYNC=false
            shift
            ;;
        *)
            echo -e "${RED}Unknown option:${NC} $1"
            show_usage
            exit 1
            ;;
    esac
done

# Create a dated persistent directory for datasets
DATE_SUFFIX=$(date +"%Y-%m-%d")
TEMP_DIR="$SCRIPT_DIR/temp_datasets_$DATE_SUFFIX"
mkdir -p "$TEMP_DIR"
mkdir -p "$TEMP_DIR/processed"

echo -e "${GREEN}Created persistent dataset directory:${NC} $TEMP_DIR"

# Set environment variables
export TMPDIR="$TEMP_DIR"
export TEMP="$TEMP_DIR"
export TMP="$TEMP_DIR"

# Build the dataset processing command
CMD="python src/main_api.py --mode process --dataset_config $CONFIG_PATH --temp_dir $TEMP_DIR"

# Add datasets
if [ -n "$DATASETS" ]; then
    # Split the dataset string into individual arguments
    for ds in $DATASETS; do
        CMD="$CMD --datasets $ds"
    done
fi

# Add streaming flag if enabled
if [ "$USE_STREAMING" = true ]; then
    CMD="$CMD --streaming"
fi

# Add Google Drive sync parameters
if [ "$DO_SYNC" = true ]; then
    CMD="$CMD --use_drive --drive_base_dir DeepseekCoder"
fi

# Run the dataset processing
echo -e "${BOLD}${GREEN}Processing datasets:${NC} $DATASETS"
echo -e "${BLUE}Command:${NC} $CMD"
echo ""

eval $CMD

echo ""
echo -e "${BOLD}${GREEN}Dataset processing completed!${NC}"
echo -e "Your datasets are stored in: ${BLUE}$TEMP_DIR/processed${NC}"
echo ""

# Sync to Google Drive if enabled
if [ "$DO_SYNC" = true ]; then
    echo -e "${BOLD}${GREEN}Verifying Google Drive sync${NC}"
    
    # Run a dedicated sync command to ensure all datasets are synced
    SYNC_CMD="python scripts/google_drive/sync_processed_datasets.py --data-dir $TEMP_DIR/processed --drive-folder $DRIVE_DIR --force"
    
    echo -e "${BLUE}Command:${NC} $SYNC_CMD"
    echo ""
    
    eval $SYNC_CMD
    
    echo ""
    echo -e "${BOLD}${GREEN}Sync verification completed!${NC}"
else
    echo -e "${YELLOW}Google Drive sync skipped as requested${NC}"
fi

echo ""
echo -e "${BOLD}${GREEN}All tasks completed successfully!${NC}"
echo "" 