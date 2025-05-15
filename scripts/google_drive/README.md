# Google Drive Integration for Paperspace

This directory contains scripts for setting up and using Google Drive with the Gen-Text-AI project, especially in headless environments like Paperspace.

## Setup Instructions

To set up Google Drive authentication in Paperspace:

1. **First, run the setup script**:

   ```bash
   python scripts/google_drive/setup_google_drive.py
   ```

   This script will:

   - Verify Google Drive manager implementation is available
   - Check for valid credentials
   - Guide you through authentication
   - Test the connection

2. **If you encounter issues**, try running the helper script first:

   ```bash
   python scripts/google_drive/ensure_drive_manager.py
   ```

   This will:

   - Fix import paths to prioritize the full implementation
   - Install required dependencies if missing
   - Copy the implementation file to the correct location if needed

## Troubleshooting

If you encounter the "minimal drive manager" warning:

```
Using minimal drive manager, actual functionality not available
```

This means the system is failing to find or properly import the full Google Drive manager implementation. The most common causes are:

1. **Missing dependencies**: The Google API libraries aren't installed. Run:

   ```bash
   pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
   ```

2. **Import path issues**: The system can't find the real implementation. Run the helper script:

   ```bash
   python scripts/google_drive/ensure_drive_manager.py
   ```

3. **Invalid credentials**: Your credentials.json file might be missing or invalid. The setup script will guide you through creating valid credentials.

## Alternative: Using rclone

If you continue having issues with the OAuth authentication, you can use rclone as an alternative:

```bash
python scripts/google_drive/setup_google_drive.py --show-rclone
```

This will display detailed instructions for setting up rclone for Google Drive access.

## Usage

Once authentication is set up, other scripts in the project that need Google Drive access will work automatically. The authentication token is cached, so you generally only need to run the setup once.

If you need to manually upload or download files, you can use the `google_drive_manager` directly in your scripts:

```python
from src.utils.google_drive_manager import drive_manager

# Authenticate (this will use cached credentials if available)
drive_manager.authenticate()

# Upload a file
drive_manager.upload_file("path/to/local/file.txt", "results")

# Download a file
file_id = "google_drive_file_id"
drive_manager.download_file(file_id, "path/to/save/file.txt")
```

## Advanced Configuration

You can set a custom base directory on Google Drive:

```bash
python scripts/google_drive/setup_google_drive.py --base_dir "MyCustomFolder"
```

This will create and use "MyCustomFolder" as the root directory for all project files on Google Drive.
