#!/bin/bash
# Fix Paperspace import issues

echo "Starting paperspace_install.sh script..."

# Add current directory to PYTHONPATH
export PYTHONPATH=/notebooks:$PYTHONPATH
echo "Added /notebooks to PYTHONPATH"

# Create necessary directories
mkdir -p src/utils
mkdir -p src/data
mkdir -p scripts/google_drive
mkdir -p scripts/src/utils
mkdir -p scripts/utilities
mkdir -p config/datasets
mkdir -p config/training
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models
mkdir -p logs
mkdir -p results
echo "Created necessary directories"

# Find Google Drive implementation file
DRIVE_IMPL=$(find . -name "google_drive_manager.py" -o -name "google_drive_manager_impl.py" | head -1)

if [ -n "$DRIVE_IMPL" ]; then
  echo "Found implementation at: $DRIVE_IMPL"
  
  # Copy to necessary locations, but only if the source and destination are different
  DEST_FILES=(
    "src/utils/google_drive_manager.py"
    "scripts/src/utils/google_drive_manager.py"
    "scripts/google_drive/google_drive_manager_impl.py"
  )
  
  for dest_file in "${DEST_FILES[@]}"; do
    # Get absolute paths
    src_abs=$(realpath "$DRIVE_IMPL")
    dest_abs=$(realpath -m "$dest_file")
    
    # Only copy if source and destination are different
    if [ "$src_abs" != "$dest_abs" ]; then
      # Create the directory if it doesn't exist
      mkdir -p "$(dirname "$dest_file")"
      cp "$DRIVE_IMPL" "$dest_file"
      echo "Copied implementation to: $dest_file"
    else
      echo "Skipped copying to $dest_file (same as source)"
    fi
  done
else
  echo "Could not find Google Drive implementation file"
fi

# Find and copy configuration files
CONFIG_FILES=$(find . -name "dataset_config.json" -o -name "training_config.json")
for config_file in $CONFIG_FILES; do
  filename=$(basename "$config_file")
  dest_file="config/$filename"
  
  # Only copy if source and destination are different
  src_abs=$(realpath "$config_file")
  dest_abs=$(realpath -m "$dest_file")
  
  if [ "$src_abs" != "$dest_abs" ]; then
    cp "$config_file" "$dest_file"
    echo "Copied $config_file to $dest_file"
  else
    echo "Skipped copying $config_file (same as destination)"
  fi
done

# If no config files found, create default configs
if [ ! -f "config/dataset_config.json" ]; then
  echo "Creating default dataset_config.json"
  echo '{
    "codesearchnet_all": {
      "path": "CodeSearchNet",
      "name": "all",
      "preprocessing": "get_codesearchnet",
      "max_samples": 100000
    },
    "the_stack_filtered": {
      "path": "bigcode/the-stack-dedup",
      "name": "python",
      "preprocessing": "get_the_stack",
      "max_samples": 100000
    }
  }' > config/dataset_config.json
fi

# Create simple import helper
cat > src/utils/import_helper.py << 'EOF'
import os
import sys

# Add project root to Python path
def add_project_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Added project root to Python path: {project_root}")
    return project_root
EOF

# Run the fix_paperspace_imports.py script if it exists
if [ -f "scripts/fix_paperspace_imports.py" ]; then
  echo "Running fix_paperspace_imports.py"
  python scripts/fix_paperspace_imports.py
fi

echo "Setup complete! Now run your scripts with the fixed imports." 