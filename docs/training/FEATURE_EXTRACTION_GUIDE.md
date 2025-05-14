# Feature Extraction Guide

This guide explains how to extract features from preprocessed datasets stored on Google Drive and prepare them for model training.

## Overview

The feature extraction process involves:

1. Downloading preprocessed datasets from Google Drive (if needed)
2. Tokenizing the text using the appropriate model tokenizer
3. Converting the tokenized data into the format required for training
4. Saving the processed features to disk for later use in training

## Prerequisites

Before starting the feature extraction process, make sure:

1. You have set up Google Drive authentication following the instructions in [GOOGLE_DRIVE_SETUP.md](../GOOGLE_DRIVE_SETUP.md)
2. Your preprocessed datasets are available on Google Drive
3. You have installed all required dependencies with `pip install -r requirements.txt`
4. You have a Hugging Face authentication token set in your environment with `export HF_TOKEN=your_token` (if needed for accessing private models)

## Using the Feature Extraction Script

The project includes a script that automates the feature extraction process:

```bash
./scripts/prepare_drive_datasets.sh
```

### Command Line Options

The script supports various command line options:

- `--model_name NAME`: Model name for tokenization (default: deepseek-ai/deepseek-coder-6.7b-base)
- `--config PATH`: Path to training config (default: config/training_config.json)
- `--dataset_config PATH`: Path to dataset config (default: config/dataset_config.json)
- `--output_dir DIR`: Output directory for features (default: data/processed/features)
- `--text_column COL`: Name of text column in datasets (default: text)
- `--max_length LEN`: Maximum sequence length (default: 1024)
- `--batch_size SIZE`: Batch size for processing (default: 1000)
- `--num_proc NUM`: Number of processes for parallel processing (default: 4)
- `--drive_base_dir DIR`: Base directory on Google Drive (optional)
- `--is_encoder_decoder`: Whether the model is an encoder-decoder model
- `--text`: Use text generation configuration
- `--code`: Use code generation configuration
- `--help`: Show help message

### Examples

Prepare datasets for DeepSeek-Coder (for code generation):

```bash
./scripts/prepare_drive_datasets.sh --code
```

Prepare datasets for Flan-UL2 (for text generation):

```bash
./scripts/prepare_drive_datasets.sh --text
```

Custom model and parameters:

```bash
./scripts/prepare_drive_datasets.sh \
    --model_name "google/flan-t5-large" \
    --max_length 512 \
    --is_encoder_decoder \
    --drive_base_dir "MyProject"
```

## Feature Extraction Process

### 1. Dataset Loading

The feature extractor first loads the preprocessed datasets from either local storage or Google Drive. If loading from Google Drive, it uses the `sync_from_drive` function from the Google Drive manager to download the datasets.

### 2. Tokenization

The tokenizer appropriate for the target model is used to convert text data into token IDs. This includes:

- Setting the maximum sequence length
- Handling padding and truncation
- Converting tokens to PyTorch tensors

For encoder-decoder models (like Flan-UL2), the labels are set to be the same as the input IDs. For causal language models (like DeepSeek-Coder), the labels are shifted to predict the next token.

### 3. Post-Processing

After tokenization, the feature extractor:

- Removes original columns to save memory
- Sets the format to PyTorch tensors
- Combines multiple datasets with appropriate weights

### 4. Output

The processed features are saved to disk in a format that can be directly loaded for training. The features include:

- `input_ids`: Token IDs for the input sequence
- `attention_mask`: Mask indicating which tokens should be attended to
- `labels`: Target token IDs for computing the loss function

## Programmatic Usage

You can also use the feature extractor programmatically in your Python code:

```python
from src.data.processors.feature_extractor import FeatureExtractor
from src.utils.google_drive_manager import DriveManager

# Initialize the feature extractor
feature_extractor = FeatureExtractor(
    model_name="deepseek-ai/deepseek-coder-6.7b-base",
    max_length=1024
)

# Initialize the drive manager (if loading from Google Drive)
drive_manager = DriveManager(base_dir="DeepseekCoder")
drive_manager.authenticate()

# Define dataset paths
dataset_paths = {
    "dataset1": "data/processed/dataset1_processed",
    "dataset2": "data/processed/dataset2_processed"
}

# Extract features
processed_dataset = feature_extractor.extract_features_from_drive_datasets(
    dataset_paths=dataset_paths,
    output_dir="data/processed/features",
    from_google_drive=True,
    drive_manager=drive_manager
)
```

## Troubleshooting

If you encounter issues during feature extraction:

1. **Google Drive authentication fails**: Make sure you have valid credentials.json and token files. Run `python scripts/setup_google_drive.py` to set up authentication.

2. **Memory errors**: Reduce the batch size (`--batch_size`) or maximum sequence length (`--max_length`).

3. **Missing text column**: The feature extractor will try to find an alternative text column or combine available columns if the specified text column is not found.

4. **Slow processing**: Increase the number of processes (`--num_proc`) if you have more CPU cores available.

## Next Steps

After extracting features, you can proceed to model training using the extracted features. See the appropriate training guides:

- [TEXT_GENERATION_GUIDE.md](TEXT_GENERATION_GUIDE.md) for text generation models
- [DIRECT_STACK_GUIDE.md](DIRECT_STACK_GUIDE.md) for code generation models
