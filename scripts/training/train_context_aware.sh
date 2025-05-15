#!/bin/bash
#
# Train Context-Aware FLAN-UT Model
#
# This script trains a context-aware FLAN-UT model with intent analysis
# and conversation memory capabilities for enhanced text generation.
#

# Stop on errors
set -e

# Load environment variables if .env file exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    source .env
fi

# Default values
CONFIG_PATH="config/training/context_aware_config.json"
DATA_DIR="data/processed"
USE_DRIVE=false
DRIVE_BASE_DIR=""
PREPARE_DATASETS=false
DATASET_PATHS=""
PRETRAINED_MODEL_PATH=""

# Print banner
echo "================================================================"
echo "           Context-Aware FLAN-UT Model Training                  "
echo "================================================================"
echo ""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --use_drive)
            USE_DRIVE=true
            shift
            ;;
        --drive_base_dir)
            DRIVE_BASE_DIR="$2"
            shift 2
            ;;
        --prepare_datasets)
            PREPARE_DATASETS=true
            shift
            ;;
        --pretrained_model_path)
            PRETRAINED_MODEL_PATH="$2"
            shift 2
            ;;
        --dataset_paths)
            # Collect all dataset paths until the next flag
            DATASET_PATHS=""
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                DATASET_PATHS="$DATASET_PATHS $1"
                shift
            done
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --config PATH           Path to the training configuration file (default: config/training/context_aware_config.json)"
            echo "  --data_dir DIR          Directory containing the processed datasets (default: data/processed)"
            echo "  --use_drive             Use Google Drive for storage"
            echo "  --drive_base_dir DIR    Base directory on Google Drive"
            echo "  --prepare_datasets      Prepare conversation datasets with context and intent before training"
            echo "  --dataset_paths PATHS   Paths to conversation datasets to process (if prepare_datasets is True)"
            echo "  --pretrained_model_path PATH  Path to a previously fine-tuned model to use as base"
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

# Prepare command arguments
CMD_ARGS=""

# Add config path
CMD_ARGS="$CMD_ARGS --config $CONFIG_PATH"

# Add data directory
CMD_ARGS="$CMD_ARGS --data_dir $DATA_DIR"

# Add Google Drive flag if enabled
if [ "$USE_DRIVE" = true ]; then
    CMD_ARGS="$CMD_ARGS --use_drive"
    
    # Add drive base directory if specified
    if [ -n "$DRIVE_BASE_DIR" ]; then
        CMD_ARGS="$CMD_ARGS --drive_base_dir $DRIVE_BASE_DIR"
    fi
fi

# Add prepare datasets flag if enabled
if [ "$PREPARE_DATASETS" = true ]; then
    CMD_ARGS="$CMD_ARGS --prepare_datasets"
    
    # Add dataset paths if specified
    if [ -n "$DATASET_PATHS" ]; then
        CMD_ARGS="$CMD_ARGS --dataset_paths $DATASET_PATHS"
    fi
fi

# Add pretrained model path if specified
if [ -n "$PRETRAINED_MODEL_PATH" ]; then
    CMD_ARGS="$CMD_ARGS --pretrained_model_path $PRETRAINED_MODEL_PATH"
fi

# Display settings
echo "Settings:"
echo "  Config: $CONFIG_PATH"
echo "  Data Directory: $DATA_DIR"
echo "  Use Google Drive: $USE_DRIVE"
if [ -n "$DRIVE_BASE_DIR" ]; then
    echo "  Drive Base Directory: $DRIVE_BASE_DIR"
fi
echo "  Prepare Datasets: $PREPARE_DATASETS"
if [ -n "$DATASET_PATHS" ]; then
    echo "  Dataset Paths: $DATASET_PATHS"
fi
if [ -n "$PRETRAINED_MODEL_PATH" ]; then
    echo "  Pretrained Model Path: $PRETRAINED_MODEL_PATH"
fi
echo ""

# Verify if needed directories exist
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found at $CONFIG_PATH"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "Warning: Data directory $DATA_DIR does not exist. Creating it."
    mkdir -p "$DATA_DIR"
fi

if [ -n "$PRETRAINED_MODEL_PATH" ] && [ ! -d "$PRETRAINED_MODEL_PATH" ]; then
    echo "Error: Pretrained model directory not found at $PRETRAINED_MODEL_PATH"
    exit 1
fi

# Run the Python script
echo "Starting context-aware model training..."
python scripts/training/train_context_aware_model.py $CMD_ARGS

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed!"
    exit 1
fi 