# Using DeepSeek-Coder Fine-Tuning on Paperspace

This guide explains how to run the DeepSeek-Coder fine-tuning pipeline on Paperspace Gradient using direct Google Drive API integration for data storage, which works around the limitations of FUSE filesystem mounting on Paperspace.

## Why We Need This Approach

Paperspace Gradient environments don't support FUSE filesystem mounting or the `google.colab.drive.mount()` approach that works in Google Colab. This is a common limitation in cloud environments for security reasons. Instead, we use the Google Drive API directly to:

1. Save and retrieve datasets, models, logs, and results
2. Maintain the same workflow without requiring manual file uploads/downloads
3. Ensure data persistence between notebook sessions

## Prerequisites

1. A Paperspace Gradient account with an A4000, A5000, or A6000 GPU (recommended)
2. Google account with Google Drive access
3. Google Cloud Platform project with Drive API enabled

## Setup Instructions

### 1. Set Up Google Cloud and Drive API

1. Visit [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or use an existing one
3. Enable the Google Drive API:
   - Go to "APIs & Services" > "Library"
   - Search for "Google Drive API" and enable it
4. Create OAuth 2.0 credentials:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "OAuth client ID"
   - Choose "Desktop application" as application type
   - **Important:** Make sure a valid redirect URI is configured (the default http://localhost is fine)
   - Download the credentials JSON file

### 2. Prepare Your Paperspace Environment

1. Create a new Gradient Notebook with PyTorch runtime
2. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/gen-text-ai.git
   cd gen-text-ai
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
   ```
4. Upload your credentials.json file to the project root directory

### 3. Authentication in Headless Environments

You have two options for authentication in headless environments:

#### Option 1: Using the standalone script (recommended)

This script doesn't require project imports and works in any directory:

```bash
python scripts/direct_auth.py --credentials credentials.json
```

#### Option 2: Using the project-aware script

This script requires proper Python path setup:

```bash
# First make sure the project root is in the Python path
export PYTHONPATH="/path/to/gen-text-ai:$PYTHONPATH"
python scripts/authenticate_headless.py --credentials credentials.json
```

With either option, you'll see the authentication process:

1. The script will generate and display an authorization URL
2. Copy this URL and open it in a browser on your local machine (not in Paperspace)
3. Log in with your Google account and grant the requested permissions
4. You'll be redirected to a page (usually showing an error, which is expected)
5. **Important:** Copy the ENTIRE URL from your browser's address bar after redirection
6. Paste this URL back into the terminal when prompted
7. The script will extract the authorization code from the URL and complete the authentication
8. The authentication token will be saved as `token.pickle` and reused for future sessions

If the script cannot extract the code from the URL, it will ask you to manually find and enter the "code" parameter from the URL, which will look something like:

```
http://localhost/?code=4/XXXX_LONG_CODE_HERE_XXXX&scope=https://www.googleapis.com/auth/drive
```

## Improved Google Drive Integration

Our latest updates to the Google Drive API integration feature several key improvements:

1. **Preventing Directory Duplication**: The system now intelligently checks for existing directories and files, preventing the creation of duplicates.

2. **Automatic Overwriting**: When saving processed datasets or models, the system will overwrite existing files rather than creating new copies, keeping your Drive organized.

3. **Robust Error Handling**: Better error detection and recovery for network issues, authentication problems, and file operations.

4. **Memory Optimization**: Files are processed in chunks to minimize memory usage, making it suitable for large datasets.

## Running the Pipeline

Use the `main_api.py` script instead of the standard `main.py` and include the `--headless` flag:

### Process Datasets with Memory Efficiency

```bash
python main_api.py \
    --mode process \
    --dataset_config config/dataset_config.json \
    --datasets code_alpaca mbpp humaneval \
    --streaming \
    --no_cache \
    --use_drive_api \
    --credentials_path credentials.json \
    --drive_base_dir DeepseekCoder \
    --headless
```

### Train the Model

```bash
python main_api.py \
    --mode train \
    --training_config config/training_config.json \
    --use_drive_api \
    --credentials_path credentials.json \
    --drive_base_dir DeepseekCoder \
    --headless
```

### Evaluate the Model

```bash
python main_api.py \
    --mode evaluate \
    --model_path models/deepseek-coder-finetune \
    --base_model deepseek-ai/deepseek-coder-6.7b-base \
    --use_drive_api \
    --credentials_path credentials.json \
    --drive_base_dir DeepseekCoder \
    --headless
```

For data science evaluation with specific libraries in DS-1000:

```bash
python main_api.py \
    --mode evaluate \
    --model_path models/deepseek-coder-finetune \
    --base_model deepseek-ai/deepseek-coder-6.7b-base \
    --ds1000_libraries numpy pandas matplotlib \
    --use_drive_api \
    --credentials_path credentials.json \
    --drive_base_dir DeepseekCoder \
    --headless
```

### Run the Complete Pipeline

For convenience, we've provided a script that runs the entire pipeline with headless mode enabled:

```bash
./scripts/run_paperspace.sh
```

The script automatically:

1. Sets up the proper Python path
2. Authenticates with Google Drive
3. Runs the complete pipeline with all necessary flags

You can customize the script's behavior with command-line arguments:

```bash
./scripts/run_paperspace.sh --mode all --datasets code_alpaca mbpp humaneval
```

To use specific libraries for the DS-1000 benchmark:

```bash
./scripts/run_paperspace.sh --mode evaluate --ds1000-libs numpy pandas matplotlib
```

## Accessing Your Files on Google Drive

All files will be stored in a directory structure under the specified base directory:

```
DeepseekCoder/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── logs/
├── results/
└── visualizations/
```

## Memory Efficiency Options

For large datasets or systems with limited memory:

- `--streaming`: Processes datasets in streaming mode, minimizing memory usage
- `--no_cache`: Disables the dataset cache to save disk space

## Troubleshooting

1. **Authentication Error - Missing redirect_uri**: Make sure your credentials.json file has at least one valid redirect_uri configured (typically http://localhost).

2. **Authentication Issues**: If you encounter authentication problems, delete the `token.pickle` file and run the authentication process again with the standalone script:

   ```bash
   rm token.pickle
   python scripts/direct_auth.py --credentials credentials.json
   ```

3. **ModuleNotFoundError for 'src'**: This indicates Python can't find the project modules:

   ```bash
   # Use the standalone script instead:
   python scripts/direct_auth.py --credentials credentials.json

   # Or set the PYTHONPATH:
   export PYTHONPATH="/path/to/gen-text-ai:$PYTHONPATH"
   ```

4. **Cannot Extract Authorization Code**: If you get an error extracting the code from the redirect URL, make sure you're copying the entire URL from your browser's address bar. The URL should contain a "code" parameter.

5. **Rate Limits**: Google Drive API has rate limits. If you hit them, the pipeline will automatically handle retries, but you may need to wait or split your work into smaller batches.

6. **File Not Found**: Double-check that your file paths are correct and that you have the appropriate permissions on your Google Drive account.

7. **Memory Issues**: If you still encounter memory problems, try processing smaller subsets of the datasets by specifying them explicitly with the `--datasets` parameter.

## References

- [Google Drive API Documentation](https://developers.google.com/drive/api/v3/about-sdk)
- [Google OAuth 2.0 for Installed Applications](https://developers.google.com/identity/protocols/oauth2/native-app)
- [DeepSeek-Coder Model Hub](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base)
- [Paperspace Gradient Documentation](https://docs.paperspace.com/gradient/)
