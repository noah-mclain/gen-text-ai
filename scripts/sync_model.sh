#!/bin/bash
# Script to automatically sync models to Google Drive during or after training
# Usage: ./sync_model.sh [model_path] [folder_id]

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

# Default values
MODEL_PATH=${1:-"${PROJECT_ROOT}/models"}
FOLDER_ID=${2:-""}
MODEL_NAME=$(basename "${MODEL_PATH}")

# If no folder ID is provided, try to find the models folder
if [ -z "${FOLDER_ID}" ]; then
    echo "No folder ID provided, attempting to find models folder in Drive..."
    FOLDER_INFO=$(${PYTHON} -c "
import sys, os
sys.path.insert(0, '$PROJECT_ROOT')
try:
    from src.utils.drive_utils import get_drive_service, find_files_by_name
    service = get_drive_service()
    if service:
        # Search for models folder
        files = find_files_by_name('models', None)
        if files:
            for file in files:
                if file.get('mimeType') == 'application/vnd.google-apps.folder':
                    print(file['id'])
                    break
except Exception as e:
    print(f'ERROR: {e}')
    sys.exit(1)
")

    # Check if folder ID was found
    if [[ ${FOLDER_INFO} == ERROR* ]]; then
        echo "Error finding models folder: ${FOLDER_INFO}"
        exit 1
    elif [ -n "${FOLDER_INFO}" ]; then
        FOLDER_ID=${FOLDER_INFO}
        echo "Found models folder with ID: ${FOLDER_ID}"
    else
        echo "Could not find models folder in Drive. Please specify a folder ID."
        exit 1
    fi
fi

# Validate arguments
if [ ! -d "${MODEL_PATH}" ]; then
    echo "Error: Model path '${MODEL_PATH}' does not exist or is not a directory"
    exit 1
fi

if [ -z "${FOLDER_ID}" ]; then
    echo "Error: No folder ID provided and could not find models folder"
    exit 1
fi

echo "==== Syncing model to Google Drive ===="
echo "  Local path: ${MODEL_PATH}"
echo "  Folder ID:  ${FOLDER_ID}"
echo "  Model name: ${MODEL_NAME}"

# Upload to Google Drive using our utility script
${PYTHON} "${PROJECT_ROOT}/scripts/setup_drive.py" --upload "${MODEL_PATH}" --to-folder "${FOLDER_ID}"

# Check if the upload was successful
if [ $? -eq 0 ]; then
    echo "==== Sync completed successfully ===="
    echo "Model synchronized at $(date)"
else
    echo "==== Sync failed ===="
    echo "Error occurred during synchronization"
    exit 1
fi 