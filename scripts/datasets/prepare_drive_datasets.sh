#!/bin/bash
#
# Prepare Datasets from Google Drive
#
# This script prepares preprocessed datasets from Google Drive for model training.
# It handles downloading the datasets from Google Drive and extracting features
# using the tokenizer appropriate for the target model.
#

# Stop on errors
set -e

# Load environment variables if .env file exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    source .env
fi

# Default values
MODEL_NAME="deepseek-ai/deepseek-coder-6.7b-base"
CONFIG="config/training_config.json"
DATASET_CONFIG="config/dataset_config.json"
OUTPUT_DIR="data/processed/features"
TEXT_COLUMN="text"
MAX_LENGTH=1024
BATCH_SIZE=1000
NUM_PROC=4
IS_ENCODER_DECODER=false
DRIVE_BASE_DIR=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --dataset_config)
            DATASET_CONFIG="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --text_column)
            TEXT_COLUMN="$2"
            shift 2
            ;;
        --max_length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_proc)
            NUM_PROC="$2"
            shift 2
            ;;
        --drive_base_dir)
            DRIVE_BASE_DIR="$2"
            shift 2
            ;;
        --is_encoder_decoder)
            IS_ENCODER_DECODER=true
            shift
            ;;
        --text)
            # Use text generation configuration
            CONFIG="config/training_config_text.json"
            DATASET_CONFIG="config/dataset_config_text.json"
            MODEL_NAME="google/flan-ul2"
            IS_ENCODER_DECODER=true
            echo "Using text generation configuration"
            shift
            ;;
        --code)
            # Use code generation configuration
            CONFIG="config/training_config.json"
            DATASET_CONFIG="config/dataset_config.json"
            MODEL_NAME="deepseek-ai/deepseek-coder-6.7b-base"
            IS_ENCODER_DECODER=false
            echo "Using code generation configuration"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model_name NAME         Model name for tokenization (default: $MODEL_NAME)"
            echo "  --config PATH             Path to training config (default: $CONFIG)"
            echo "  --dataset_config PATH     Path to dataset config (default: $DATASET_CONFIG)"
            echo "  --output_dir DIR          Output directory for features (default: $OUTPUT_DIR)"
            echo "  --text_column COL         Name of text column in datasets (default: $TEXT_COLUMN)"
            echo "  --max_length LEN          Maximum sequence length (default: $MAX_LENGTH)"
            echo "  --batch_size SIZE         Batch size for processing (default: $BATCH_SIZE)"
            echo "  --num_proc NUM            Number of processes for parallel processing (default: $NUM_PROC)"
            echo "  --drive_base_dir DIR      Base directory on Google Drive (optional)"
            echo "  --is_encoder_decoder      Whether the model is an encoder-decoder model"
            echo "  --text                    Use text generation configuration"
            echo "  --code                    Use code generation configuration"
            echo "  --help                    Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build encoder-decoder flag
ENCODER_DECODER_FLAG=""
if [ "$IS_ENCODER_DECODER" = true ]; then
    ENCODER_DECODER_FLAG="--is_encoder_decoder"
fi

# Build drive base directory flag
DRIVE_BASE_DIR_FLAG=""
if [ -n "$DRIVE_BASE_DIR" ]; then
    DRIVE_BASE_DIR_FLAG="--drive_base_dir $DRIVE_BASE_DIR"
fi

echo "======================================================"
echo "        Preparing Datasets from Google Drive"
echo "======================================================"
echo ""
echo "Model:               $MODEL_NAME"
echo "Training Config:     $CONFIG"
echo "Dataset Config:      $DATASET_CONFIG"
echo "Output Directory:    $OUTPUT_DIR"
echo "Text Column:         $TEXT_COLUMN"
echo "Max Length:          $MAX_LENGTH"
echo "Batch Size:          $BATCH_SIZE"
echo "Num Processes:       $NUM_PROC"
echo "Encoder-Decoder:     $IS_ENCODER_DECODER"
echo "Drive Base Dir:      $DRIVE_BASE_DIR"
echo ""
echo "======================================================"

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN environment variable is not set"
    echo "You may encounter issues downloading models or datasets from Hugging Face"
    echo "Consider setting HF_TOKEN in your .env file or environment"
    echo ""
fi

# Run the Python script
echo "Starting dataset preparation..."
python scripts/prepare_datasets_for_training.py \
    --model_name "$MODEL_NAME" \
    --config "$CONFIG" \
    --dataset_config "$DATASET_CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --from_google_drive \
    --text_column "$TEXT_COLUMN" \
    --max_length "$MAX_LENGTH" \
    --batch_size "$BATCH_SIZE" \
    --num_proc "$NUM_PROC" \
    $ENCODER_DECODER_FLAG \
    $DRIVE_BASE_DIR_FLAG

echo ""
echo "Dataset preparation complete! Features saved to $OUTPUT_DIR"
echo ""

# Make script executable
chmod +x scripts/prepare_datasets_for_training.py 