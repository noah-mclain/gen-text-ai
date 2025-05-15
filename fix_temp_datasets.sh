#!/bin/bash
# Script to create and use a visible non-temporary directory for dataset processing
# This prevents datasets from being lost when temporary directories are cleaned up

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
echo -e "${BOLD}${BLUE} Dataset Processing Directory Fix ${NC}"
echo -e "${BOLD}${BLUE}=========================================${NC}"
echo ""

# Function to show usage
show_usage() {
    echo -e "${YELLOW}Usage:${NC} $0 [options]"
    echo ""
    echo "Options:"
    echo "  --help, -h            Show this help message"
    echo "  --temp-dir DIR        Specify the temp directory to use (default: temp_datasets_YYYY-MM-DD)"
    echo "  --clean               Clean up existing temp directories after setup"
    echo ""
    echo "Examples:"
    echo "  $0                    Setup with auto-generated temp directory name"
    echo "  $0 --temp-dir my_dataset_processing --clean"
    echo ""
}

# Default values
# Create a unique dated name for the temp directory
DATE_SUFFIX=$(date +"%Y-%m-%d")
DEFAULT_TEMP_DIR="temp_datasets_$DATE_SUFFIX"
TEMP_DIR=""
DO_CLEAN=false

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
        --clean)
            DO_CLEAN=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option:${NC} $1"
            show_usage
            exit 1
            ;;
    esac
done

# Use default temp dir if not specified
if [[ -z "$TEMP_DIR" ]]; then
    TEMP_DIR="$DEFAULT_TEMP_DIR"
fi

# Make path absolute if relative
if [[ ! "$TEMP_DIR" = /* ]]; then
    TEMP_DIR="$PROJECT_ROOT/$TEMP_DIR"
fi

echo -e "${BOLD}Project root:${NC} $PROJECT_ROOT"
echo -e "${BOLD}Temp directory:${NC} $TEMP_DIR"

# Create the temporary directory
mkdir -p "$TEMP_DIR"
echo -e "${GREEN}Created temporary directory:${NC} $TEMP_DIR"

# Set up subdirectories
mkdir -p "$TEMP_DIR/processed"
echo -e "${GREEN}Created processed directory:${NC} $TEMP_DIR/processed"

# Set environment variables so Python scripts use this directory
export TMPDIR="$TEMP_DIR"
export TEMP="$TEMP_DIR"
export TMP="$TEMP_DIR"

# Create a .env file with the temp directory configuration
ENV_FILE="$PROJECT_ROOT/.env"
echo "# Created by fix_temp_datasets.sh" > "$ENV_FILE"
echo "# $(date)" >> "$ENV_FILE"
echo "TMPDIR=\"$TEMP_DIR\"" >> "$ENV_FILE"
echo "TEMP=\"$TEMP_DIR\"" >> "$ENV_FILE"
echo "TMP=\"$TEMP_DIR\"" >> "$ENV_FILE"
echo -e "${GREEN}Created .env file with temp directory settings:${NC} $ENV_FILE"

# Clean up old temporary directories if requested
if [[ "$DO_CLEAN" == "true" ]]; then
    echo -e "${YELLOW}Cleaning up old dataset processing directories...${NC}"
    
    # Find dataset_processing directories in standard tmp locations
    OLD_DIRS=$(find /tmp -maxdepth 1 -type d -name "dataset_processing*" 2>/dev/null || true)
    if [[ -n "$OLD_DIRS" ]]; then
        echo -e "${YELLOW}Found the following directories that can be cleaned:${NC}"
        echo "$OLD_DIRS"
        
        echo -e "${YELLOW}Would you like to delete these directories? [y/N]${NC}"
        read -r CONFIRM
        
        if [[ "$CONFIRM" == "y" || "$CONFIRM" == "Y" ]]; then
            echo -e "${YELLOW}Deleting old temporary directories...${NC}"
            # Use xargs to safely handle directory names
            echo "$OLD_DIRS" | xargs -I{} rm -rf {}
            echo -e "${GREEN}Cleaned up old temporary directories${NC}"
        else
            echo -e "${BLUE}Skipping cleanup${NC}"
        fi
    else
        echo -e "${GREEN}No old dataset_processing directories found to clean${NC}"
    fi
fi

echo ""
echo -e "${BOLD}${GREEN}Setup complete!${NC}"
echo -e "Now configured to use the following directory for dataset processing:"
echo -e "${BLUE}$TEMP_DIR${NC}"
echo ""
echo -e "${YELLOW}Important:${NC} For Python scripts to use this directory, run them with:"
echo -e "  TMPDIR=\"$TEMP_DIR\" python your_script.py"
echo ""
echo -e "Or source the .env file first:"
echo -e "  source .env && python your_script.py"
echo ""
echo -e "You can also add --temp_dir \"$TEMP_DIR\" to your Python script arguments"
echo -e "if the script supports this parameter."
echo ""

# Make the script executable
chmod +x "$0" 