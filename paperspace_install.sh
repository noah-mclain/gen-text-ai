#!/bin/bash
# Fix Paperspace import issues

# Add current directory to PYTHONPATH
export PYTHONPATH=/notebooks:$PYTHONPATH

# Create necessary directories
mkdir -p src/utils
mkdir -p scripts/google_drive
mkdir -p scripts/src/utils
mkdir -p scripts/utilities

# Find Google Drive implementation file
DRIVE_IMPL=$(find . -name "google_drive_manager.py" -o -name "google_drive_manager_impl.py" | head -1)

if [ -n "$DRIVE_IMPL" ]; then
  echo "Found implementation at: $DRIVE_IMPL"
  
  # Copy to necessary locations
  cp "$DRIVE_IMPL" src/utils/google_drive_manager.py
  cp "$DRIVE_IMPL" scripts/src/utils/google_drive_manager.py
  cp "$DRIVE_IMPL" scripts/google_drive/google_drive_manager_impl.py
  
  echo "Copied implementation to all required locations"
else
  echo "Could not find Google Drive implementation file"
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

echo "Setup complete! Now run your scripts with the fixed imports." 