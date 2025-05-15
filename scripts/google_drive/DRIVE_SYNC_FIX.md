# Google Drive Sync Fix Guide

If you're encountering issues with syncing datasets to Google Drive in Paperspace, this guide will help you fix the problem.

## Common Symptoms

- You see the error: `Using minimal drive manager, actual functionality not available`
- Datasets are processed successfully but nothing gets uploaded to Google Drive
- You see this log message: `Found 0 processed datasets to sync`

## Quick Fix

Run the fix script directly in your Paperspace notebook:

```bash
# Navigate to your project directory if needed
cd /notebooks/gen-text-ai  # Adjust to your actual project path

# Run the fix script
python scripts/google_drive/fix_drive_sync.py
```

This will:

1. Fix Python import paths
2. Ensure the correct Google Drive Manager implementation is used
3. Clean Python cache files to prevent import issues
4. Find and sync all processed datasets to Google Drive

## If You're Still Having Issues

### 1. Make sure dependencies are installed

```bash
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

### 2. Verify your credentials

Make sure you have a valid `credentials.json` file in the root of your project:

```bash
cat credentials.json | grep client_id
```

If missing or invalid, set up credentials using:

```bash
python scripts/google_drive/setup_google_drive.py
```

### 3. Manually run steps

If the automatic fix doesn't work, try running the individual steps:

```bash
# 1. Fix imports and copy implementation
python scripts/google_drive/ensure_drive_manager.py

# 2. Test authentication
python scripts/google_drive/setup_google_drive.py

# 3. Try manual sync
python scripts/google_drive/sync_datasets.py
```

## Detailed Explanation

The issue occurs because Paperspace's file structure and Python import system can sometimes lead to the wrong implementation of the Google Drive manager being used. Specifically:

1. When imports fail, the system falls back to a "minimal drive manager" that doesn't have actual functionality
2. Python's caching system may keep using the minimal implementation even after fixes
3. Import paths may be incorrect due to the different environment in Paperspace

The fix script addresses all these issues by:

1. Ensuring the proper implementation file is available in all necessary locations
2. Fixing Python's import paths so it finds the correct implementation
3. Clearing Python's cache to force fresh imports
4. Directly calling the sync functionality with the real implementation

## Preventing Future Issues

To avoid this issue in the future:

1. Always run `ensure_drive_manager.py` at the start of your Paperspace sessions
2. Make sure your imports explicitly use full paths like `from src.utils.google_drive_manager import ...`
3. Use the `--use_drive_api` flag when running data processing scripts

Example:

```bash
python src/data/process_datasets.py --use_drive_api --drive_base_dir "DeepseekCoder"
```
