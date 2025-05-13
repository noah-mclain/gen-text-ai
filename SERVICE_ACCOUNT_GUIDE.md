# Google Drive Service Account Authentication Guide

This guide explains how to set up Google Drive authentication using a service account, which is the recommended approach for headless environments like Paperspace.

## Why Use a Service Account?

- **No browser needed**: Perfect for headless environments like Paperspace where browser access is limited
- **More secure**: No need to handle personal Google account tokens
- **Persistent access**: No need to re-authenticate frequently
- **Automation-friendly**: Better for scheduled tasks and CI/CD pipelines

## Setup Steps

### 1. Create a Google Cloud Project

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or select an existing one)
3. Enable the Google Drive API:
   - Navigate to "APIs & Services > Library"
   - Search for "Google Drive API" and enable it

### 2. Create a Service Account

1. In your project, go to "APIs & Services > Credentials"
2. Click "Create Credentials" and select "Service Account"
3. Enter a name for your service account (e.g., "Gen-Text-AI Drive Service")
4. Click "Create and Continue"
5. For role, select "Project > Editor" (or a custom role with Drive access)
6. Click "Continue" and then "Done"

### 3. Create and Download the JSON Key

1. Find your new service account in the list and click on it
2. Go to the "Keys" tab
3. Click "Add Key" > "Create new key"
4. Choose JSON format and click "Create"
5. The key file will download automatically
6. Upload this file to your Paperspace environment

### 4. Share Your Google Drive Folders

**Important**: The service account needs access to your Google Drive folders.

1. Open the JSON key file and look for the `client_email` field
2. Copy this email address (it looks like: `service-account-name@project-id.iam.gserviceaccount.com`)
3. In Google Drive, right-click on your project folders
4. Click "Share" and add the service account email with "Editor" access

### 5. Set Up the Environment Variable

Run our setup script:

```bash
python scripts/setup_service_account.py
```

This script will:

1. Ask for the path to your service account JSON file
2. Copy it to a standard location
3. Set up the environment variable
4. Optionally add it to your shell profile
5. Test that everything works

Alternatively, you can manually:

1. Place your service account JSON file in a secure location
2. Set the environment variable:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your-service-account.json
   ```
3. Add this to your .bashrc or .zshrc for persistence

## Verifying the Setup

Test that your service account can access Google Drive:

```bash
python scripts/google_drive_manager.py --action check
```

If successful, you should see:

```
✅ Google Drive is properly authenticated and accessible.
```

## Using with Training Scripts

Our training scripts now automatically detect and use service account authentication when available. The `train_a6000_optimized.sh` script will check for a file named `service-account.json` in the project root directory.

When running custom commands, make sure the `GOOGLE_APPLICATION_CREDENTIALS` environment variable is set.

## Troubleshooting

### Common Issues

1. **"Access denied" errors**:

   - Make sure you shared your Drive folders with the service account email
   - Check that the email address matches exactly
   - Ensure you gave "Editor" permissions

2. **"Invalid credentials" errors**:

   - Verify the JSON file is valid and not corrupted
   - Ensure the environment variable points to the correct file
   - Check that the service account has the Drive API enabled

3. **File not found errors**:
   - Verify the Drive folder structure matches what the script expects
   - Check folder IDs in the logs

### Debugging Tips

Add these flags for more verbose output:

```bash
python -m src.training.train --config config/training_config.json --data_dir data/processed --use_drive --drive_base_dir DeepseekCoder --verbose
```

Check the Drive API quota in Google Cloud Console if you receive quota exceeded errors.

## Advanced Configuration

For advanced use cases like cross-project authentication or domain-wide delegation, see the [Google Drive API documentation](https://developers.google.com/drive/api/v3/about-auth).
