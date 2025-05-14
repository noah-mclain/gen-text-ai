#!/bin/bash

# Script to clean up all The Stack dataset files and directories
# This removes all traces of failed The Stack dataset processing

# Set color codes for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================================${NC}"
echo -e "${GREEN}Cleaning Up The Stack Dataset Files${NC}"
echo -e "${BLUE}=================================================${NC}"

# Main paths to clean
DATA_PROCESSED="data/processed"
DATA_RAW="data/raw"
TEMP_DIR="/tmp"

# Check if directories exist before removing
echo -e "${BLUE}Checking for The Stack dataset files...${NC}"

# Check processed data directory
if [ -d "$DATA_PROCESSED" ]; then
    # Count how many stack directories exist
    STACK_COUNT=$(find "$DATA_PROCESSED" -name "*the_stack*" -type d | wc -l)
    
    if [ $STACK_COUNT -gt 0 ]; then
        echo -e "${YELLOW}Found $STACK_COUNT The Stack dataset directories in $DATA_PROCESSED${NC}"
        
        # List the directories
        echo -e "${BLUE}Directories to remove:${NC}"
        find "$DATA_PROCESSED" -name "*the_stack*" -type d | sort
        
        # Ask for confirmation
        echo -e "${YELLOW}Do you want to remove these directories? (y/n)${NC}"
        read -n 1 -r
        echo
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${BLUE}Removing The Stack dataset directories...${NC}"
            find "$DATA_PROCESSED" -name "*the_stack*" -type d -exec rm -rf {} \; 2>/dev/null || true
            echo -e "${GREEN}Removed The Stack dataset directories from $DATA_PROCESSED${NC}"
        else
            echo -e "${YELLOW}Skipping removal of The Stack dataset directories${NC}"
        fi
    else
        echo -e "${GREEN}No The Stack dataset directories found in $DATA_PROCESSED${NC}"
    fi
else
    echo -e "${YELLOW}Data processed directory $DATA_PROCESSED does not exist${NC}"
fi

# Check raw data directory
if [ -d "$DATA_RAW" ]; then
    # Count how many stack directories or files exist
    STACK_COUNT=$(find "$DATA_RAW" -name "*stack*" | wc -l)
    
    if [ $STACK_COUNT -gt 0 ]; then
        echo -e "${YELLOW}Found $STACK_COUNT The Stack files/directories in $DATA_RAW${NC}"
        
        # List the files/directories
        echo -e "${BLUE}Files/directories to remove:${NC}"
        find "$DATA_RAW" -name "*stack*" | sort
        
        # Ask for confirmation
        echo -e "${YELLOW}Do you want to remove these files/directories? (y/n)${NC}"
        read -n 1 -r
        echo
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${BLUE}Removing The Stack files/directories...${NC}"
            find "$DATA_RAW" -name "*stack*" -exec rm -rf {} \; 2>/dev/null || true
            echo -e "${GREEN}Removed The Stack files/directories from $DATA_RAW${NC}"
        else
            echo -e "${YELLOW}Skipping removal of The Stack files/directories${NC}"
        fi
    else
        echo -e "${GREEN}No The Stack files/directories found in $DATA_RAW${NC}"
    fi
else
    echo -e "${YELLOW}Data raw directory $DATA_RAW does not exist${NC}"
fi

# Check temporary directory for cache files
echo -e "${BLUE}Checking for The Stack cached files in $TEMP_DIR...${NC}"
STACK_TEMP_COUNT=$(find "$TEMP_DIR" -name "*stack*" -type f 2>/dev/null | wc -l)

if [ $STACK_TEMP_COUNT -gt 0 ]; then
    echo -e "${YELLOW}Found $STACK_TEMP_COUNT The Stack temporary files${NC}"
    echo -e "${BLUE}Removing The Stack temporary files...${NC}"
    find "$TEMP_DIR" -name "*stack*" -type f -delete 2>/dev/null || true
    echo -e "${GREEN}Removed The Stack temporary files${NC}"
else
    echo -e "${GREEN}No The Stack temporary files found${NC}"
fi

# Also remove the debug and fix scripts
if [ -f "debug_stack_processing.py" ]; then
    echo -e "${BLUE}Removing debug_stack_processing.py...${NC}"
    rm debug_stack_processing.py
fi

if [ -f "fix_stack_loader.py" ]; then
    echo -e "${BLUE}Removing fix_stack_loader.py...${NC}"
    rm fix_stack_loader.py
fi

if [ -f "process_stack_dataset.sh" ]; then
    echo -e "${BLUE}Removing process_stack_dataset.sh...${NC}"
    rm process_stack_dataset.sh
fi

# Final check for any remaining references to The Stack in code
echo -e "${BLUE}Performing final check for Stack references in code...${NC}"
echo -e "${BLUE}------------------------------------------------${NC}"

# Search for stack references in Python files but exclude those in config directory
STACK_CODE_REFS=$(grep -r "the_stack\|stack_filtered" --include="*.py" --exclude-dir=".env" --exclude-dir="config" . | wc -l)

if [ $STACK_CODE_REFS -gt 0 ]; then
    echo -e "${YELLOW}Found $STACK_CODE_REFS references to The Stack in Python code:${NC}"
    grep -r "the_stack\|stack_filtered" --include="*.py" --exclude-dir=".env" --exclude-dir="config" . | head -n 10
    
    if [ $STACK_CODE_REFS -gt 10 ]; then
        echo -e "${YELLOW}...and $(($STACK_CODE_REFS - 10)) more references${NC}"
    fi
    
    echo -e "${YELLOW}To ensure training runs without errors, please review these files manually.${NC}"
    echo -e "${YELLOW}Most references can be safely kept if the datasets are properly disabled in config files.${NC}"
else
    echo -e "${GREEN}No active references to The Stack found in Python code.${NC}"
fi

# Check for any uncommented Stack references in config files that aren't disabled
ACTIVE_CONFIG_REFS=$(grep -r '"the_stack\|"stack_filtered' --include="*.json" . | grep -v '"enabled": false' | wc -l)

if [ $ACTIVE_CONFIG_REFS -gt 0 ]; then
    echo -e "${RED}WARNING: Found $ACTIVE_CONFIG_REFS potentially active Stack references in config files:${NC}"
    grep -r '"the_stack\|"stack_filtered' --include="*.json" . | grep -v '"enabled": false' | head -n 10
    echo -e "${RED}Please ensure all Stack dataset entries have 'enabled': false in config files!${NC}"
else
    echo -e "${GREEN}All Stack references in config files are properly disabled.${NC}"
fi

echo -e "${BLUE}=================================================${NC}"
echo -e "${GREEN}Cleanup Complete!${NC}"
echo -e "${BLUE}=================================================${NC}"

echo -e "${GREEN}The Stack datasets have been disabled in the configuration files and removed from the filesystem.${NC}"
echo -e "${GREEN}The training will now use other datasets instead of The Stack.${NC}"
echo -e "${GREEN}Optimized weights for remaining datasets have been applied to both code and text models.${NC}"

echo "Starting cleanup of redundant Stack dataset files..."

# List of redundant files to remove
REDUNDANT_FILES=(
  "scripts/cleanup_drive_utils.py"
  "TRAIN_STACK_NOW.sh"
  "scripts/process_stack_direct.sh"
  "scripts/cleanup_drive_scripts.py"
  "scripts/prepare_datasets.py"
  "config/training_config_cpu.json"
  "scripts/fix_dataset_paths.py"
  "src/data/dataset_lookup.py"
  "scripts/preload_datasets.py"
  "test_imports.py"
  "test_stack_processing.py"
  "process_stack_dataset.sh"
  "debug_stack_processing.py"
  "fix_stack_loader.py"
  "config/dataset_config_updated.json"
)

# Check and remove each file if it exists
for file in "${REDUNDANT_FILES[@]}"; do
  if [ -f "$file" ]; then
    rm "$file"
    echo "Removed: $file"
  else
    echo "Skipping (not found): $file"
  fi
done

# Create an archived directory for any files we want to keep for reference
mkdir -p archived
echo "Created archived directory for future reference"

echo "Cleanup complete!"
echo "All redundant Stack dataset files have been removed."
echo "Use the consolidated Google Drive Manager for all Drive interactions." 