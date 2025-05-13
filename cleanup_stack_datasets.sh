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

echo -e "${BLUE}=================================================${NC}"
echo -e "${GREEN}Cleanup Complete!${NC}"
echo -e "${BLUE}=================================================${NC}"

echo -e "${GREEN}The Stack datasets have been disabled in the configuration files and removed from the filesystem.${NC}"
echo -e "${GREEN}The training will now use other datasets instead of The Stack.${NC}" 