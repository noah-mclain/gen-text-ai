# Text Dataset Feature Extraction Guide

This guide explains how to extract features from text datasets configured in `dataset_config_text.json` and prepare them for model training.

## Overview

The text feature extraction process involves:

1. Loading preprocessed text datasets (optionally from Google Drive)
2. Tokenizing the text using the appropriate model tokenizer
3. Converting the tokenized data into the format required for training
4. Saving the processed features to disk for later use in training

## Prerequisites

Before starting the feature extraction process, make sure:

1. Your text datasets are properly configured in `config/datasets/dataset_config_text.json`
2. The required processors for your datasets are implemented in `src/data/processors/text_processors.py`
3. You have installed all required dependencies with `pip install -r requirements.txt`
4. If using Google Drive, you have set up authentication following the instructions in the Google Drive setup guide

## Configuration File Location

The system will automatically search for the dataset configuration file in multiple locations:

- The path provided via command line (`--dataset_config` argument)
- In the `config/datasets/` directory
- In the `/notebooks/config/datasets/` directory (for Paperspace environments)
- It will also search recursively in the current directory and `/notebooks` directory

This ensures the script works both locally and in cloud environments like Paperspace.

## Supported Text Datasets

The current implementation supports the following text datasets:

- GPTeacher General Instruct (`gpteacher_general`)
- Pile dataset (`pile`)
- Synthetic Persona Chat (`synthetic_persona`)
- Writing Prompts (`writingprompts`)

Additional datasets can be added by:

1. Adding their configuration to `config/datasets/dataset_config_text.json`
2. Implementing their processor in `src/data/processors/text_processors.py`

## Using the Feature Extraction Script

To extract features from text datasets, run:

```bash
./scripts/datasets/prepare_text_features.sh
```

### Command Line Options

The script supports various command line options:

- `--model_name NAME`: Model name for tokenization (default: google/flan-ul2)
- `--config PATH`: Path to training config (default: config/training_config_text.json)
- `--dataset_config PATH`: Path to dataset config (default: config/datasets/dataset_config_text.json)
- `--output_dir DIR`: Output directory for features (default: data/processed/text_features)
- `--text_column COL`: Name of text column in datasets (default: text)
- `--max_length LEN`: Maximum sequence length (default: 512)
- `--batch_size SIZE`: Batch size for processing (default: 1000)
- `--num_proc NUM`: Number of processes for parallel processing (default: 4)
- `--drive_base_dir DIR`: Base directory on Google Drive (optional)
- `--from_google_drive`: Load datasets from Google Drive
- `--is_encoder_decoder`: Set model as encoder-decoder (default for text models)
- `--no_is_encoder_decoder`: Unset model as encoder-decoder (for causal language models)

### Examples

Basic usage with default settings (for Flan-UL2):

```bash
./scripts/datasets/prepare_text_features.sh
```

Using a specific model:

```bash
./scripts/datasets/prepare_text_features.sh --model_name "facebook/opt-1.3b"
```

For a causal language model (not encoder-decoder):

```bash
./scripts/datasets/prepare_text_features.sh --model_name "facebook/opt-1.3b" --no_is_encoder_decoder
```

Loading from Google Drive:

```bash
./scripts/datasets/prepare_text_features.sh --from_google_drive --drive_base_dir "TextModels"
```

## Understanding the Text Feature Extraction Process

### 1. Dataset Loading

The feature extractor first loads the preprocessed datasets from either local storage or Google Drive. The datasets should have been processed by the corresponding processors defined in `text_processors.py`.

### 2. Handling Text Column Detection

The text feature extractor is designed to be robust to different dataset formats:

1. It first checks if the specified `text_column` exists in the dataset
2. If not, it looks for common text-related columns like "instruction", "response", "content", etc.
3. If still not found, it combines available text columns based on common patterns:
   - Instruction/Response pairs
   - Prompt/Completion pairs
   - Question/Answer pairs
   - Or simply concatenating all text columns

### 3. Tokenization

Text is tokenized using the model's tokenizer with:

- Maximum sequence length
- Appropriate padding and truncation
- Labels generation based on whether the model is encoder-decoder or causal

### 4. Output

The processed features are saved to the specified output directory with:

- `input_ids`: Token IDs for the input text
- `attention_mask`: Mask indicating which tokens should be attended to
- `labels`: Target token IDs appropriate for the model type

## Differences from Code Dataset Feature Extraction

This feature extraction system is specifically tailored for text datasets:

1. Default model is Flan-UL2 instead of DeepSeek-Coder
2. Default maximum length is 512 instead of 1024
3. Encoder-decoder model type is enabled by default
4. Text column detection logic is focused on common text dataset formats
5. No code-specific processing is performed

## Troubleshooting

If you encounter issues during feature extraction:

1. **Configuration file not found**: The script will search in multiple locations including `config/datasets/` and `/notebooks/config/datasets/`. Make sure the file exists in one of these locations.

2. **Missing text column**: The feature extractor will try to find an alternative text column or combine available columns.

3. **Tokenization errors**: Ensure your model name is correct and publicly available. Try reducing the max_length if you encounter memory issues.

4. **Empty datasets**: Check that your dataset configuration points to valid, preprocessed datasets.

5. **Memory errors**: Reduce batch_size or max_length.

6. **Google Drive errors**: If you encounter Google Drive related errors but your datasets are already processed locally, try running without the `--from_google_drive` flag.

## Next Steps

After extracting features, you can proceed to model training using the features for your text generation model.
