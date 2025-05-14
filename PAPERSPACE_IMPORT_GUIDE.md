# Fixing Import Issues in Paperspace

After reorganizing the codebase, you may encounter import issues when running in the Paperspace environment. This guide explains how to fix these issues.

## The Problem

In Paperspace, Python can't find modules because:

1. The `src` module isn't in the Python path
2. The Google Drive manager implementation isn't where the code expects it
3. The imports use absolute paths (`from src.utils...`) which don't work when the project structure is different
4. Configuration files may not be found in the expected locations

## Quick Solution

The simplest solution is to run the paperspace_install.sh script when you first start:

```bash
# Run the installation script
bash paperspace_install.sh

# Verify everything is set up correctly
python scripts/utilities/verify_paths.py
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

3. Run the paperspace_install.sh script:

```bash
bash paperspace_install.sh
```

4. Verify the setup:

```bash
python scripts/utilities/verify_paths.py --fix
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

For a more permanent solution, we've added several improvements:

1. `src/main_api.py` - Now tries multiple import paths and checks for config files in multiple locations
2. `src/data/process_datasets.py` - Handles imports more robustly and searches for config files
3. `src/utils/drive_api_utils.py` - Tries different import paths for Drive integration
4. `paperspace_install.sh` - Comprehensive setup script for Paperspace
5. `scripts/fix_paperspace_imports.py` - More advanced setup for complex cases
6. `scripts/utilities/verify_paths.py` - Verifies all paths are set up correctly

The fixes use a combination of:

1. Adding the project root to the Python path
2. Trying multiple import paths (direct, relative, absolute)
3. Copying essential files to expected locations
4. Creating symbolic links for compatibility
5. Checking for configuration files in multiple locations
6. Providing appropriate fallbacks when files aren't found

## Using the Verification Tool

The verification tool will check that everything is set up correctly:

```bash
python scripts/utilities/verify_paths.py
```

If issues are found, you can try to fix them automatically:

```bash
python scripts/utilities/verify_paths.py --fix
```

## Testing the Fix

After applying the fixes, test with:

```bash
# Test importing the Drive manager directly
python -c "from scripts.google_drive.google_drive_manager import test_authentication; print('Import successful')"

# Test checking paths
python scripts/utilities/verify_paths.py

# Run the process datasets command
python src/main_api.py --mode process --streaming --no_cache --dataset_config config/dataset_config.json --use_drive --drive_base_dir DeepseekCoder
```

## Troubleshooting

If you still encounter issues:

1. Check if the PYTHONPATH includes the project root: `echo $PYTHONPATH`
2. Verify the directory structure: `find . -type d -not -path "*/\.*" | sort`
3. Check if the config files exist: `find . -name "dataset_config.json"`
4. Try running with verbose logging: `python src/main_api.py --mode process --verbose ...`
