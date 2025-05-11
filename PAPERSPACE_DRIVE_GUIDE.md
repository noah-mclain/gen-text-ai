# Google Drive API Integration for Paperspace

This guide explains how to set up Google Drive API integration for Paperspace notebooks without FUSE mounting.

## Prerequisites

1. A Google account
2. A Paperspace account with a notebook instance
3. Google Cloud Platform project with Drive API enabled

## Step 1: Set Up Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Navigate to "APIs & Services" > "Library"
4. Search for "Google Drive API" and enable it
5. Go to "APIs & Services" > "Credentials"
6. Click "Create Credentials" > "OAuth client ID"
7. For application type, select "Desktop application"
8. Give it a name (e.g., "Paperspace Integration")
9. Click "Create"
10. Download the JSON file (this is your `credentials.json`)

## Step 2: Install Dependencies in Paperspace

1. Open your Paperspace notebook
2. Install the required packages:

```bash
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

## Step 3: Copy the Credentials to Paperspace

You have three options:

### Option A: Upload the credentials.json file directly

1. In Paperspace, go to "Files" tab
2. Click "Upload" and select your `credentials.json` file
3. Place it in your project root directory

### Option B: Copy the credentials JSON content

1. Open the downloaded `credentials.json` file on your local machine
2. Copy the entire JSON content
3. In Paperspace, use the `--credentials` flag with the JSON string (see below)

### Option C: Environment variable

1. Set an environment variable with the JSON content:

```bash
export GOOGLE_CREDENTIALS='{"installed":{"client_id":"..."}}'
```

## Step 4: Run the Setup Script

Run the setup script with one of the following methods:

### Using the uploaded credentials.json file:

```bash
python scripts/setup_drive.py --setup --install-deps --base-dir DeepseekCoder
```

### Using JSON string directly:

```bash
python scripts/setup_drive.py --setup --install-deps --base-dir DeepseekCoder --credentials '{"installed":{"client_id":"..."}}'
```

### Using environment variable:

```bash
python scripts/setup_drive.py --setup --install-deps --base-dir DeepseekCoder --credentials "$GOOGLE_CREDENTIALS"
```

## Step 5: Complete Authentication Flow

1. The script will display a URL
2. Copy this URL and open it in a browser on your local machine (not in Paperspace)
3. Sign in with your Google account and grant the requested permissions
4. You'll receive an authorization code
5. Copy the code and paste it back into the Paperspace terminal prompt
6. If successful, you'll see "Google Drive API setup successful. Drive is accessible."

## Step 6: Using the Integration

Now that you've set up the integration, you can use it with your training scripts:

```bash
# Start training with Drive integration
python src/training/train.py --config config/training_config.json --data_dir data/processed --use_drive --drive_base_dir DeepseekCoder
```

## Syncing Models to Drive

You can manually sync models at any time using the sync script:

```bash
# Sync the latest model directory
./scripts/sync_model.sh

# Or sync a specific model directory
./scripts/sync_model.sh /path/to/model/directory
```

## Troubleshooting

### "Authentication required" errors

- Your token may have expired. Run the setup script again.

### "credentials.json not found" error

- Make sure your credentials file is in one of the searched locations.
- Alternatively, pass the credentials JSON with the `--credentials` flag.

### "Error during authentication" error

- Make sure you've copied the entire authorization code correctly.
- Try running the setup with `--install-deps` to ensure all packages are installed.

## Advanced Operations

The setup script supports several operations:

### List files in Drive:

```bash
python scripts/setup_drive.py --list "model*"
```

### Upload a file or directory:

```bash
python scripts/setup_drive.py --upload /path/to/file --to-folder folder_id
```

### Download a file:

```bash
python scripts/setup_drive.py --download file_id --to-path /path/to/save
```
