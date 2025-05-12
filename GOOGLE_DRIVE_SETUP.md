# Google Drive Integration Setup Guide

This guide explains how to set up Google Drive integration for the gen-text-ai project, which allows saving and loading models, datasets, and other files directly to/from your Google Drive.

## Prerequisites

1. A Google account
2. Python packages:
   - google-api-python-client
   - google-auth-httplib2
   - google-auth-oauthlib

## Installation

Install the required packages:

```bash
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

## Setup Process

### Step 1: Create a Google Cloud Project

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Navigate to "APIs & Services" > "Library"
4. Search for "Google Drive API" and enable it

### Step 2: Create OAuth Credentials

1. In the Google Cloud Console, navigate to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth client ID"
3. Set the application type to "Desktop application"
4. Give it a name (e.g., "Gen-Text-AI Drive Client")
5. Click "Create"
6. Download the credentials JSON file

### Step 3: Configure Credentials

1. Save the downloaded JSON file as `credentials.json` in the project root directory
2. **IMPORTANT**: Ensure the file contains both of these redirect URIs:
   ```json
   "redirect_uris": ["http://localhost:8080", "urn:ietf:wg:oauth:2.0:oob"]
   ```
   The first URI is needed for browser-based authentication, and the second is required for headless environments like Paperspace notebooks.

### Step 4: Run the Setup Script

We've provided a comprehensive Google Drive manager script:

```bash
# For environments with a browser:
python scripts/google_drive_manager.py --action setup

# For headless environments (like Paperspace notebooks):
python scripts/google_drive_manager.py --action setup --headless
```

The setup process will:

1. Check if your credentials file is properly formatted
2. Launch the authentication flow appropriate for your environment
3. Save your authentication token for future use

### Step 5: Verify the Integration

Test that the integration is working correctly:

```bash
python scripts/google_drive_manager.py --action test
```

This will perform basic operations like authenticating, creating a directory, and uploading a test file to ensure the integration is working properly.

## Using the Drive Manager

The Google Drive manager script provides several useful functions:

```bash
# Create a directory structure in Google Drive
python scripts/google_drive_manager.py --action create_folders --base_dir your_project_name

# Upload files or directories
python scripts/google_drive_manager.py --action upload --local_path /path/to/file --remote_folder folder_id

# Download files
python scripts/google_drive_manager.py --action download --remote_file file_id --local_path /path/to/save

# List files in a Drive folder (with optional search)
python scripts/google_drive_manager.py --action list --remote_folder folder_id --search "filename"

# Quick check if Drive is accessible
python scripts/google_drive_manager.py --action check
```

## Using the Drive Integration in Your Code

To use Google Drive integration in your code:

```python
from src.utils.drive_utils import is_drive_mounted, save_model_to_drive, get_drive_path

# Check if Drive is accessible (for headless environment)
if is_drive_mounted(headless=True):
    # Save a model to Drive
    save_model_to_drive(
        model_path="/local/path/to/model",
        drive_dir="your_folder_id_on_drive",
        model_name="my_model",
        headless=True  # For headless environments
    )

    # Convert a local path to a Google Drive path
    drive_path = get_drive_path(
        local_path="/local/path/to/file",
        drive_base_path="/drive/path"
    )
```

## For Training Scripts

When using the `DeepseekFineTuner` class, you can enable Google Drive integration:

```python
from src.training.trainer import DeepseekFineTuner

tuner = DeepseekFineTuner(
    config_path="path/to/config.json",
    use_drive=True,
    drive_base_dir="my_project_folder"
)
```

This will automatically save models and checkpoints to Google Drive.

## Paperspace Integration

For using Google Drive with Paperspace notebooks (which are headless):

1. In your local environment:

   - Set up the Google Cloud project and credentials as described above
   - Ensure `"urn:ietf:wg:oauth:2.0:oob"` is included in the redirect URIs
   - Upload the credentials.json file to your Paperspace notebook

2. In Paperspace:

   ```bash
   # Install dependencies
   pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib

   # Set up authentication in headless mode
   python scripts/google_drive_manager.py --action setup --headless
   ```

3. When the authorization URL appears, open it in a browser on your local machine
4. Log in with your Google account and authorize the application
5. Copy the authorization code and paste it back in the Paperspace terminal
6. Once authenticated, the token will be saved for future use

## Troubleshooting

### "Missing required parameter: redirect_uri" Error

If you see this error:

1. Ensure your credentials.json file includes both required redirect URIs:
   ```json
   "redirect_uris": ["http://localhost:8080", "urn:ietf:wg:oauth:2.0:oob"]
   ```
2. Delete the token file (located at `~/.drive_token.json` by default)
3. Run the setup script again with the appropriate mode:
   ```bash
   python scripts/google_drive_manager.py --action setup --headless
   ```

### Authentication Issues

If you're having authentication problems:

1. Make sure the Google Drive API is enabled for your project
2. Check that your credentials.json file is properly formatted with both redirect URIs
3. Verify that you're logged into the correct Google account during authentication
4. Try deleting the token file and authenticating again

### File Not Found Errors

If files aren't being found on Google Drive:

1. Ensure your authentication is working by running the test action:
   ```bash
   python scripts/google_drive_manager.py --action test
   ```
2. Check that the paths and folder IDs you're using are correct
3. Verify that your Google account has the necessary permissions

## Advanced Usage

### Using a Different Token Path

You can specify a different location for your authentication token:

```bash
python scripts/google_drive_manager.py --action setup --token_path /path/to/token.json
```

Or in code:

```python
from src.utils.drive_utils import get_credentials

creds = get_credentials(token_path="/path/to/token.json")
```

### Working with Folder IDs

Google Drive uses folder IDs rather than paths. You can:

1. Find a folder's ID in your browser URL when viewing the folder
2. List folders with their IDs:
   ```bash
   python scripts/google_drive_manager.py --action list
   ```
3. Use the `find_or_create_folder` function to get or create folders:

   ```python
   from src.utils.drive_utils import find_or_create_folder

   folder_id = find_or_create_folder("My Folder Name", headless=True)
   ```

## Need More Help?

If you encounter any issues not covered in this guide, please:

1. Check the [Google Drive API documentation](https://developers.google.com/drive/api/v3/about-sdk)
2. Open an issue in the project repository with details about your problem
