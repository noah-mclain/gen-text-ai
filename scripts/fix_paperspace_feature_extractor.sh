#!/bin/bash
# Fix Paperspace Feature Extractor
# Main script to fix the feature extractor issue in Paperspace

set -e
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/..")"

echo "Fixing feature extractor issues in Paperspace..."

# Check if we're in Paperspace
if [[ ! -d "/notebooks" ]]; then
  echo "Not running in Paperspace. No need to fix feature extractor."
  exit 0
fi

# Create target directories
mkdir -p /notebooks/src/data/processors

# Method 1: Try to copy the file directly
echo "Method 1: Copying feature extractor directly..."
SRC_FILE="$PROJECT_ROOT/src/data/processors/feature_extractor.py"
DST_FILE="/notebooks/src/data/processors/feature_extractor.py"

if [[ -f "$SRC_FILE" ]]; then
  cp "$SRC_FILE" "$DST_FILE"
  echo "✅ Successfully copied feature extractor to $DST_FILE"
  
  # Create necessary symlinks to ensure the feature extractor is accessible
  mkdir -p "/notebooks/jar_env/lib/python3.11/site-packages/src/data/processors"
  ln -sf "$DST_FILE" "/notebooks/jar_env/lib/python3.11/site-packages/src/data/processors/feature_extractor.py"
  echo "✅ Created symlink to feature_extractor.py in site-packages"
  
  exit 0
fi

# Method 2: Try using the content file if available
echo "Method 2: Using content file..."
CONTENT_FILE="$SCRIPT_DIR/utilities/feature_extractor_content.py"

if [[ -f "$CONTENT_FILE" ]]; then
  cat "$CONTENT_FILE" > "$DST_FILE"
  
  if [[ -f "$DST_FILE" ]]; then
    echo "✅ Successfully created feature extractor from content file"
    
    # Create necessary symlinks
    mkdir -p "/notebooks/jar_env/lib/python3.11/site-packages/src/data/processors"
    ln -sf "$DST_FILE" "/notebooks/jar_env/lib/python3.11/site-packages/src/data/processors/feature_extractor.py"
    echo "✅ Created symlink to feature_extractor.py in site-packages"
    
    exit 0
  fi
fi

# Method 3: Try using utilities script if it exists
echo "Method 3: Using utilities scripts..."
if [[ -f "$SCRIPT_DIR/utilities/fix_feature_extractor.sh" ]]; then
  bash "$SCRIPT_DIR/utilities/fix_feature_extractor.sh"
  
  if [[ -f "$DST_FILE" ]]; then
    echo "✅ Feature extractor fixed successfully using utilities script"
    exit 0
  fi
fi

# If we got here, all methods failed
echo "❌ Failed to fix feature extractor with any method."
exit 1 