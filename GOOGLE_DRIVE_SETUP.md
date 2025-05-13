# Google Drive Integration Setup Guide

This document provides detailed instructions for setting up Google Drive integration with this project. Since we're working in headless environments like Paperspace, we use the OAuth "Out-of-Band" (OOB) flow for authentication instead of service accounts.

## Prerequisites

1. A Google account with Google Drive access
2. Python 3.6+ installed
3. Required packages: `google-api-python-client google-auth-httplib2 google-auth-oauthlib`

## Step 1: Create Google Cloud Project & OAuth Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Google Drive API:
   - Go to "APIs & Services" > "Library"
   - Search for "Google Drive API"
   - Click on it and press "Enable"
4. Create OAuth credentials:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "OAuth client ID"
   - Select "Desktop application" as application type
   - Give it a name (e.g., "GenTextAI Drive Integration")
   - Click "Create"
5. Download the credentials by clicking the download icon (JSON)
6. Save the downloaded JSON file as `credentials.json` in your project root directory

## Step 2: Run the Setup Script

We've created a setup script that guides you through the OAuth authentication process:

```bash
# Run the setup script
python setup_google_drive.py
```

For headless environments (like Paperspace):

```bash
python setup_google_drive.py --headless
```

The script will:

1. Validate your credentials file
2. Generate an authorization URL
3. For headless mode: Display the URL and prompt you to visit it, authorize access, and paste the authorization code
4. Test authentication and verify Drive access
5. Create necessary directories in your Drive

## Step 3: Test the Integration

After setup, you can verify that everything is working correctly:

```bash
# Test the Google Drive integration
python -m tests.test_google_drive --headless
```

This will run a series of tests that:

1. Authenticate with Google Drive
2. Create test directories
3. Upload a test file
4. Download the test file
5. Clean up test data

## Step 4: Using Drive Integration in Training

The Drive integration is built into all major scripts. Use these flags to enable it:

```bash
# Process datasets with Drive integration
python main_api.py --mode process --use_drive --headless

# Train with Drive integration
python main_api.py --mode train --use_drive --headless

# Full pipeline with Drive integration
python main_api.py --mode all --use_drive --headless

# Training with the text model
python train_text_flan.py --use_drive --headless
```

## Advanced: Using Alternative Methods

### Option 1: rclone (Alternative to API)

If you prefer using rclone instead of the API:

1. Install rclone:

```bash
curl https://rclone.org/install.sh | sudo bash
```

2. Configure rclone for Google Drive:

```bash
rclone config
```

3. Follow the interactive setup, selecting Google Drive
4. Use the rclone integration in scripts:

```bash
python main_api.py --use_drive --drive_tool rclone
```

### Option 2: Google Drive API Direct Usage

For direct API usage without our helper module:

```python
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# Setup credentials
SCOPES = ['https://www.googleapis.com/auth/drive']
flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
creds = flow.run_local_server(port=0)

# Build Drive service
drive_service = build('drive', 'v3', credentials=creds)

# Use Drive API
results = drive_service.files().list(pageSize=10, fields="nextPageToken, files(id, name)").execute()
items = results.get('files', [])
```

## Troubleshooting

### Authentication Issues

- **Error: redirect_uri_mismatch**  
  Solution: Use the `--headless` flag to enable the OOB flow which doesn't require redirect

- **Error: Invalid client**  
  Solution: Re-download credentials and ensure you selected "Desktop application" type

- **Error: invalid_grant**  
  Solution: Authentication codes expire quickly. Request a new code and use it immediately

### File Operation Issues

- **Error: File not found**  
  Solution: Check folder IDs and ensure paths are correct

- **Error: Insufficient permissions**  
  Solution: Request 'drive' scope (not just 'drive.file') during authentication

### Connectivity Issues

- **Error: Connection reset by peer**  
  Solution: Check internet connectivity and retry with exponential backoff

- **Error: Quota exceeded**  
  Solution: Google Drive API has usage limits. Implement rate limiting in your code

## Best Practices

1. Store only necessary data in Drive to avoid reaching storage limits
2. Always clean up test files after testing
3. Use structured folder hierarchies in Drive
4. Implement proper error handling for Drive operations
5. Use token caching to avoid repeated authentication
6. For large file transfers, use resumable uploads
