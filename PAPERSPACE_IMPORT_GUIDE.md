# Fixing Import Issues in Paperspace

After reorganizing the codebase, you may encounter import issues when running in the Paperspace environment. This guide explains how to fix these issues.

## The Problem

In Paperspace, Python can't find modules because:

1. The `src` module isn't in the Python path
2. The Google Drive manager implementation isn't where the code expects it
3. The imports use absolute paths (`from src.utils...`) which don't work when the project structure is different

## Quick Solution

Add this code at the beginning of any script you need to run in Paperspace:

```python
import os
import sys

# Add project root to Python path
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added project root to Python path: {project_root}")
```

## Step-by-Step Setup for Paperspace

1. First, make sure you're in the notebooks directory:

```bash
cd /notebooks
```

2. Make sure your repository is properly cloned and up to date:

```bash
git pull
```

3. Create the necessary directory structure:

```bash
mkdir -p src/utils
mkdir -p scripts/google_drive
mkdir -p scripts/src/utils
```

4. Copy the Google Drive manager implementation to all necessary locations:

```bash
# Identify where the implementation file is
find . -name "google_drive_manager.py" -o -name "google_drive_manager_impl.py"

# Copy the implementation to all required locations
cp [found_path] src/utils/google_drive_manager.py
cp [found_path] scripts/src/utils/google_drive_manager.py
cp [found_path] scripts/google_drive/google_drive_manager_impl.py
```

5. Set the PYTHONPATH environment variable:

```bash
export PYTHONPATH=/notebooks:$PYTHONPATH
```

6. Now you can run your scripts with the fixed imports:

```bash
python src/main_api.py --mode process --streaming --no_cache --dataset_config config/dataset_config.json --use_drive --drive_base_dir DeepseekCoder
```

## Permanent Solution

For a more permanent solution, we've updated these files:

1. `src/main_api.py` - Now tries multiple import paths
2. `src/data/process_datasets.py` - Handles imports more robustly
3. `src/utils/drive_api_utils.py` - Tries different import paths
4. Added `scripts/fix_paperspace_imports.py` - Run this script at the start of your session

The fixes use a combination of:

1. Adding the project root to the Python path
2. Trying multiple import paths (direct, relative, absolute)
3. Copying essential files to expected locations
4. Creating symbolic links for compatibility

Run the fix script at the beginning of your session:

```bash
python scripts/fix_paperspace_imports.py
```

## Testing the Fix

After applying the fixes, test with:

```bash
# Test importing the Drive manager directly
python -c "from scripts.google_drive.google_drive_manager import test_authentication; print('Import successful')"

# Run the process datasets command
python src/main_api.py --mode process --streaming --no_cache --dataset_config config/dataset_config.json --use_drive --drive_base_dir DeepseekCoder
```
