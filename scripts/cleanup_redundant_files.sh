#!/bin/bash

# Script to safely clean up redundant Google Drive auth files
# that have been consolidated into google_drive_manager.py

set -e  # Exit immediately if a command fails

# ANSI color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo -e "${BLUE}=========================================================${NC}"
echo -e "${GREEN}Cleaning up redundant Google Drive authentication files${NC}"
echo -e "${BLUE}=========================================================${NC}"

# List of files to remove
FILES_TO_REMOVE=(
    "scripts/setup_drive_auth.py"
    "scripts/test_drive.py"
    "scripts/authenticate_headless.py"
    "scripts/direct_auth.py"
)

# Count how many files we've actually deleted
DELETED_COUNT=0

for FILE in "${FILES_TO_REMOVE[@]}"; do
    FULL_PATH="$PROJECT_ROOT/$FILE"
    
    if [ -f "$FULL_PATH" ]; then
        echo -e "${YELLOW}Removing redundant file: $FILE${NC}"
        echo -e "${BLUE}(Functionality now available in scripts/google_drive_manager.py)${NC}"
        
        # Check if git is available and the file is tracked by git
        if command -v git &> /dev/null && git -C "$PROJECT_ROOT" ls-files --error-unmatch "$FILE" &> /dev/null; then
            echo -e "${BLUE}File is tracked by git, using git rm${NC}"
            git -C "$PROJECT_ROOT" rm "$FULL_PATH"
        else
            rm "$FULL_PATH"
        fi
        
        DELETED_COUNT=$((DELETED_COUNT+1))
        echo -e "${GREEN}File removed successfully${NC}"
    fi
done

if [ "$DELETED_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}No redundant files found to clean up${NC}"
else
    echo -e "${GREEN}Cleaned up $DELETED_COUNT redundant files${NC}"
    echo -e "${BLUE}All functionality has been consolidated into scripts/google_drive_manager.py${NC}"
fi

echo -e "${GREEN}Done!${NC}" 