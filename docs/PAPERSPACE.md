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

We now provide a comprehensive Google Drive manager script that handles authentication in both regular and headless environments:

```bash
# Authenticate in headless environments like Paperspace
python scripts/google_drive_manager.py --action setup --headless

# Create necessary directory structure in your Drive
python scripts/google_drive_manager.py --action create_folders --base_dir DeepseekCoder --headless
```

When running the headless authentication:

1. The script will generate and display an authorization URL
2. Copy this URL and open it in a browser on your local machine (not in Paperspace)
3. Log in with your Google account and grant the requested permissions
4. You'll be redirected or shown an authorization code
5. Copy the authorization code and paste it back into the terminal when prompted
6. The authentication token will be saved and reused for future sessions

To verify that authentication worked correctly:

```bash
python scripts/google_drive_manager.py --action check --headless
```

> **Important**: When setting up your Google Cloud credentials, make sure to include both of these redirect URIs:
>
> ```
> http://localhost:8080
> urn:ietf:wg:oauth:2.0:oob
> ```
>
> The second URI is required for headless authentication to work properly.

## Improved Google Drive Integration

Our latest updates to the Google Drive API integration feature several key improvements:

1. **Preventing Directory Duplication**: The system now intelligently checks for existing directories and files, preventing the creation of duplicates.

2. **Automatic Overwriting**: When saving processed datasets or models, the system will overwrite existing files rather than creating new copies, keeping your Drive organized.

3. **Advanced Preprocessing Pipeline**:

   - **Deduplication of Examples**: Automatically removes redundant prompt+completion pairs during preprocessing
   - **Consistent Formatting**: Applies uniform formatting with "User:" and "Assistant:" prefixes
   - **Clean Text Processing**: Strips whitespace, normalizes text, and removes too-short examples
   - **Multi-Language Support**: Process code datasets in multiple programming languages:
     - CodeSearchNet: Python, Java, JavaScript, PHP, Ruby, Go
     - The Stack: Python, Java, JavaScript, PHP, Ruby, Go, C, C++, C#, and many more
     - Process specific languages or all available languages based on configuration

4. **Improved Training Process**:

   - **Dataset Shuffling**: Explicitly enables shuffling during training for better model convergence
   - **Proper Train/Val/Test Splits**: Ensures data is correctly distributed for training and evaluation

5. **Robust Error Handling**:

   - Better error detection and recovery for network issues, authentication problems, and file operations
   - Properly handles 404 errors when attempting to delete files that don't exist
   - Improved file ID caching with automatic cache clearing to ensure latest versions are always used

6. **Memory Optimization**: Files are processed in chunks to minimize memory usage, making it suitable for large datasets.

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

### Process Multiple Programming Languages

```bash
# Process specific languages
python main_api.py \
    --mode process \
    --dataset_config config/dataset_config.json \
    --datasets the_stack_python the_stack_java codesearchnet_javascript \
    --streaming \
    --no_cache \
    --use_drive_api \
    --credentials_path credentials.json \
    --drive_base_dir DeepseekCoder \
    --headless

# Process all available languages at once
python main_api.py \
    --mode process \
    --dataset_config config/dataset_config.json \
    --datasets the_stack_all codesearchnet_all \
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
2. Authenticates with Google Drive using the consolidated manager script
3. Runs the complete pipeline with all necessary flags

You can customize the script's behavior with command-line arguments:

```bash
./scripts/run_paperspace.sh --mode all --datasets code_alpaca mbpp humaneval
```

To use specific libraries for the DS-1000 benchmark:

```bash
./scripts/run_paperspace.sh --mode evaluate --ds1000-libs numpy pandas matplotlib
```

## Using the Drive Manager Directly

The Google Drive manager provides several useful functions for managing files:

```bash
# List files in your Drive
python scripts/google_drive_manager.py --action list --headless

# Upload a directory of results
python scripts/google_drive_manager.py --action upload --local_path ./results --remote_folder your_folder_id --headless

# Download a file from Drive
python scripts/google_drive_manager.py --action download --remote_file file_id --local_path ./local_file.txt --headless
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
   python scripts/google_drive_manager.py --action setup --headless
   ```

3. **ModuleNotFoundError for 'src'**: This indicates Python can't find the project modules:

   ```bash
   # Use the standalone script instead:
   python scripts/google_drive_manager.py --action setup --headless

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
