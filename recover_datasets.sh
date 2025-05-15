#!/bin/bash

# recover_datasets.sh
# Script to recover Arrow-format HuggingFace datasets and sync them to Google Drive
# Can search in temporary directories and the project directory

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
PROJECT_ROOT="$SCRIPT_DIR"

echo -e "${BOLD}${BLUE}=========================================${NC}"
echo -e "${BOLD}${BLUE} Dataset Recovery and Sync Utility ${NC}"
echo -e "${BOLD}${BLUE}=========================================${NC}"
echo ""

# Function to show usage
show_usage() {
    echo -e "${YELLOW}Usage:${NC} $0 [options]"
    echo ""
    echo "Options:"
    echo "  --help, -h            Show this help message"
    echo "  --temp-dir DIR        Specify a temporary directory to search"
    echo "  --recovery-dir DIR    Directory to store recovered datasets (default: recovered_datasets)"
    echo "  --drive-folder DIR    Google Drive folder to sync to (default: processed_data)"
    echo "  --scan-all-temp       Scan all temporary directories, not just the specified ones"
    echo "  --skip-recovery       Skip recovery step, only scan and sync existing datasets"
    echo "  --force               Don't ask for confirmation before syncing"
    echo ""
    echo "Examples:"
    echo "  $0                    Run with default settings"
    echo "  $0 --temp-dir /tmp/dataset_processing_123456"
    echo "  $0 --scan-all-temp --recovery-dir my_recovered_datasets"
    echo ""
}

# Default values
TEMP_DIR=""
RECOVERY_DIR="recovered_datasets"
DRIVE_FOLDER="processed_data"
SCAN_ALL_TEMP=false
SKIP_RECOVERY=false
FORCE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_usage
            exit 0
            ;;
        --temp-dir)
            TEMP_DIR="$2"
            shift 2
            ;;
        --recovery-dir)
            RECOVERY_DIR="$2"
            shift 2
            ;;
        --drive-folder)
            DRIVE_FOLDER="$2"
            shift 2
            ;;
        --scan-all-temp)
            SCAN_ALL_TEMP=true
            shift
            ;;
        --skip-recovery)
            SKIP_RECOVERY=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option:${NC} $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check if the specified temp directory exists
if [[ -n "$TEMP_DIR" && ! -d "$TEMP_DIR" ]]; then
    echo -e "${RED}Error:${NC} Specified temporary directory does not exist: $TEMP_DIR"
    exit 1
fi

# Create recovery directory if it doesn't exist
if [[ "$SKIP_RECOVERY" == "false" ]]; then
    mkdir -p "$RECOVERY_DIR"
    echo -e "${GREEN}Created recovery directory:${NC} $RECOVERY_DIR"
fi

echo -e "${BOLD}Project root:${NC} $PROJECT_ROOT"
echo -e "${BOLD}Recovery directory:${NC} $RECOVERY_DIR"
echo -e "${BOLD}Google Drive folder:${NC} $DRIVE_FOLDER"
echo ""

# Step 1: Recover datasets
if [[ "$SKIP_RECOVERY" == "false" ]]; then
    echo -e "${BOLD}${GREEN}Step 1: Recovering datasets from temporary directories${NC}"
    
    # Build the command
    RECOVER_CMD="python scripts/google_drive/recover_processed_datasets.py --recovery_dir $RECOVERY_DIR --drive_folder $DRIVE_FOLDER"
    
    # If a specific temp directory is specified, add it to the search paths
    if [[ -n "$TEMP_DIR" ]]; then
        echo -e "${BLUE}Searching in specified temporary directory:${NC} $TEMP_DIR"
        export ADDITIONAL_TEMP_DIR="$TEMP_DIR"
        # We'll add the temp dir to PYTHONPATH to make it available to the script
        export PYTHONPATH="$TEMP_DIR:$PYTHONPATH"
    fi
    
    echo -e "${BLUE}Running:${NC} $RECOVER_CMD"
    eval $RECOVER_CMD
    
    echo -e "${GREEN}Recovery complete!${NC}"
    echo ""
else
    echo -e "${YELLOW}Skipping recovery step as requested${NC}"
    echo ""
fi

# Step 2: Fix and sync datasets
echo -e "${BOLD}${GREEN}Step 2: Finding and syncing datasets to Google Drive${NC}"

# Build the command
FIX_CMD="python scripts/google_drive/fix_dataset_sync.py --drive_folder $DRIVE_FOLDER"

# Add recovery directory to scan paths
FIX_CMD="$FIX_CMD --scan_paths $RECOVERY_DIR data/processed"

# Add temp directory if specified
if [[ -n "$TEMP_DIR" ]]; then
    FIX_CMD="$FIX_CMD $TEMP_DIR"
fi

# Scan all temp directories if requested
if [[ "$SCAN_ALL_TEMP" == "true" ]]; then
    FIX_CMD="$FIX_CMD --scan_temp"
fi

# Force if requested
if [[ "$FORCE" == "true" ]]; then
    FIX_CMD="$FIX_CMD --force"
fi

echo -e "${BLUE}Running:${NC} $FIX_CMD"
eval $FIX_CMD

# Step 3: Final sync of processed datasets
echo -e "${BOLD}${GREEN}Step 3: Final sync of all processed datasets${NC}"

# Build the command
SYNC_CMD="python scripts/google_drive/sync_processed_datasets.py --drive-folder $DRIVE_FOLDER"

# Force if requested
if [[ "$FORCE" == "true" ]]; then
    SYNC_CMD="$SYNC_CMD --force"
fi

echo -e "${BLUE}Running:${NC} $SYNC_CMD"
eval $SYNC_CMD

echo ""
echo -e "${BOLD}${GREEN}All steps completed!${NC}"
echo -e "Datasets have been recovered and synced to Google Drive folder: ${BLUE}$DRIVE_FOLDER${NC}"
echo ""

# Make the script executable
chmod +x "$0" 