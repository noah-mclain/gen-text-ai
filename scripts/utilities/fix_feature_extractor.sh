#!/bin/bash
# Fix Feature Extractor
# This script fixes the feature extractor issue in Paperspace by trying multiple methods

set -e
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../..")"

# Function to check if we're in Paperspace
is_paperspace() {
  [[ -d "/notebooks" ]]
}

# Only proceed if we're in Paperspace
if ! is_paperspace; then
  echo "Not running in Paperspace. No need to fix feature extractor."
  exit 0
fi

echo "Fixing feature extractor in Paperspace..."

# Create target directories
mkdir -p /notebooks/src/data/processors

# Method 1: Try to copy the file directly using bash
echo "Method 1: Copying file using bash..."
SRC_FILE="$PROJECT_ROOT/src/data/processors/feature_extractor.py"
DST_FILE="/notebooks/src/data/processors/feature_extractor.py"

if [[ -f "$SRC_FILE" ]]; then
  cp "$SRC_FILE" "$DST_FILE"
  echo "✅ Successfully copied feature extractor using bash"
  exit 0
fi

# Method 2: Try using our Python script without imports
echo "Method 2: Using minimal Python script..."
python "$SCRIPT_DIR/create_feature_extractor.py"
if [[ -f "$DST_FILE" ]]; then
  echo "✅ Successfully created feature extractor using Python script"
  exit 0
fi

# Method 3: Try to manually write the file using cat
echo "Method 3: Writing file using cat..."
CONTENT_FILE="$SCRIPT_DIR/feature_extractor_content.py"

if [[ -f "$CONTENT_FILE" ]]; then
  cat "$CONTENT_FILE" > "$DST_FILE"
  if [[ -f "$DST_FILE" ]]; then
    echo "✅ Successfully created feature extractor using cat"
    exit 0
  fi
fi

echo "❌ Failed to create feature extractor with any method."
exit 1 