#!/bin/bash
#
# Prepare Text Datasets for Training
#
# This script prepares preprocessed text datasets for model training
# by extracting features using the appropriate tokenizer.
#

# Stop on errors
set -e

# Load environment variables if .env file exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    source .env
fi

# Default values
MODEL_NAME="google/flan-ul2"
CONFIG="config/training_config_text.json"
DATASET_CONFIG="config/datasets/dataset_config_text.json"
OUTPUT_DIR="data/processed/text_features"
TEXT_COLUMN="text"
MAX_LENGTH=512
BATCH_SIZE=1000
NUM_PROC=4
IS_ENCODER_DECODER=true
DRIVE_BASE_DIR=""
FROM_GOOGLE_DRIVE=false

# Print banner
echo "================================================================"
echo "           Text Dataset Feature Extraction Tool                  "
echo "================================================================"
echo ""

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
        --from_google_drive)
            FROM_GOOGLE_DRIVE=true
            shift
            ;;
        --is_encoder_decoder)
            IS_ENCODER_DECODER=true
            shift
            ;;
        --no_is_encoder_decoder)
            IS_ENCODER_DECODER=false
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model_name NAME       Model name for tokenization (default: google/flan-ul2)"
            echo "  --config PATH           Path to training config (default: config/training_config_text.json)"
            echo "  --dataset_config PATH   Path to dataset config (default: config/datasets/dataset_config_text.json)"
            echo "  --output_dir DIR        Output directory for features (default: data/processed/text_features)"
            echo "  --text_column COL       Name of text column in datasets (default: text)"
            echo "  --max_length LEN        Maximum sequence length (default: 512)"
            echo "  --batch_size SIZE       Batch size for processing (default: 1000)"
            echo "  --num_proc NUM          Number of processes for parallel processing (default: 4)"
            echo "  --drive_base_dir DIR    Base directory on Google Drive (optional)"
            echo "  --from_google_drive     Load datasets from Google Drive"
            echo "  --is_encoder_decoder    Set model as encoder-decoder (default for text models)"
            echo "  --no_is_encoder_decoder Unset model as encoder-decoder (for causal models)"
            echo "  --help                  Show this help message"
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

# Prepare encoder-decoder flag
if [ "$IS_ENCODER_DECODER" = true ]; then
    ENCODER_DECODER_FLAG="--is_encoder_decoder"
else
    ENCODER_DECODER_FLAG=""
fi

# Prepare Google Drive flag
if [ "$FROM_GOOGLE_DRIVE" = true ]; then
    DRIVE_FLAG="--from_google_drive"
    
    # If drive base dir is specified, add it to command
    if [ -n "$DRIVE_BASE_DIR" ]; then
        DRIVE_FLAG="$DRIVE_FLAG --drive_base_dir $DRIVE_BASE_DIR"
    fi
else
    DRIVE_FLAG=""
fi

# Display settings
echo "Settings:"
echo "  Model: $MODEL_NAME"
echo "  Config: $CONFIG"
echo "  Dataset Config: $DATASET_CONFIG"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Text Column: $TEXT_COLUMN"
echo "  Max Length: $MAX_LENGTH"
echo "  Batch Size: $BATCH_SIZE"
echo "  Number of Processes: $NUM_PROC"
echo "  Encoder-Decoder Model: $IS_ENCODER_DECODER"
echo "  From Google Drive: $FROM_GOOGLE_DRIVE"
if [ -n "$DRIVE_BASE_DIR" ]; then
    echo "  Drive Base Directory: $DRIVE_BASE_DIR"
fi
echo ""

# Run the Python script with the specified settings
echo "Starting feature extraction..."
python scripts/datasets/prepare_text_datasets_for_training.py \
    --model_name "$MODEL_NAME" \
    --config "$CONFIG" \
    --dataset_config "$DATASET_CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --text_column "$TEXT_COLUMN" \
    --max_length "$MAX_LENGTH" \
    --batch_size "$BATCH_SIZE" \
    --num_proc "$NUM_PROC" \
    $ENCODER_DECODER_FLAG \
    $DRIVE_FLAG

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo "✅ Feature extraction completed successfully!"
    echo "Features saved to: $OUTPUT_DIR"
else
    echo "❌ Feature extraction failed!"
    exit 1
fi 