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

### 3. First-Time Authentication

The first time you run the pipeline, you'll need to complete the OAuth flow:

1. Run the authentication script:
   ```bash
   python -c "from src.utils.drive_api_utils import initialize_drive_api; initialize_drive_api()"
   ```
2. A URL will be printed to the console. Open this URL in your browser.
3. Choose your Google account and grant the requested permissions
4. Copy the authorization code displayed and paste it back into the console
5. The authentication token is now saved as `token.pickle` and will be reused for future sessions

## Running the Pipeline

Use the `main_api.py` script instead of the standard `main.py`:

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
    --drive_base_dir DeepseekCoder
```

### Train the Model

```bash
python main_api.py \
    --mode train \
    --training_config config/training_config.json \
    --use_drive_api \
    --credentials_path credentials.json \
    --drive_base_dir DeepseekCoder
```

### Evaluate the Model

```bash
python main_api.py \
    --mode evaluate \
    --model_path models/deepseek-coder-finetune \
    --base_model deepseek-ai/deepseek-coder-6.7b-base \
    --use_drive_api \
    --credentials_path credentials.json \
    --drive_base_dir DeepseekCoder
```

### Run the Complete Pipeline

For convenience, we've provided a script that runs the entire pipeline:

```bash
./scripts/run_paperspace.sh
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

1. **Authentication Issues**: If you encounter authentication problems, delete the `token.pickle` file and run the authentication process again.

2. **Rate Limits**: Google Drive API has rate limits. If you hit them, the pipeline will automatically handle retries, but you may need to wait or split your work into smaller batches.

3. **File Not Found**: Double-check that your file paths are correct and that you have the appropriate permissions on your Google Drive account.

4. **Memory Issues**: If you still encounter memory problems, try processing smaller subsets of the datasets by specifying them explicitly with the `--datasets` parameter.

## References

- [Google Drive API Documentation](https://developers.google.com/drive/api/v3/about-sdk)
- [DeepSeek-Coder Model Hub](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base)
- [Paperspace Gradient Documentation](https://docs.paperspace.com/gradient/)
