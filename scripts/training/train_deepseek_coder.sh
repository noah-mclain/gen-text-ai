#!/bin/bash
#
# Train DeepSeek-Coder with Feature Extraction
#
# This script:
# 1. Downloads preprocessed datasets from Google Drive
# 2. Extracts features using the appropriate tokenizer
# 3. Trains a DeepSeek-Coder model with LoRA fine-tuning
#

# Stop on errors
set -e

# Set default values
BASE_MODEL="deepseek-ai/deepseek-coder-6.7b-base"
OUTPUT_DIR="models/deepseek-coder-finetune"
CONFIG="config/training_config.json"
DATASET_CONFIG="config/dataset_config.json"
FEATURES_DIR="data/processed/features/deepseek_coder"
DATA_DIR="data/processed"
USE_DRIVE=true
DRIVE_BASE_DIR="DeepseekCoder"
DEVICE="cuda"
USE_DEEPSPEED=false
USE_4BIT=true
USE_UNSLOTH=true
PUSH_TO_HUB=false
TRAIN_ONLY=false
FEATURES_ONLY=false
USE_WANDB=true
MAX_LENGTH=2048
BATCH_SIZE=4
GRAD_STEPS=16
EPOCHS=3
LEARNING_RATE=2e-4

# Load environment variables if .env file exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    source .env
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --base_model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
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
        --features_dir)
            FEATURES_DIR="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --no_drive)
            USE_DRIVE=false
            shift
            ;;
        --drive_base_dir)
            DRIVE_BASE_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --use_deepspeed)
            USE_DEEPSPEED=true
            shift
            ;;
        --no_4bit)
            USE_4BIT=false
            shift
            ;;
        --no_unsloth)
            USE_UNSLOTH=false
            shift
            ;;
        --push_to_hub)
            PUSH_TO_HUB=true
            shift
            ;;
        --train_only)
            TRAIN_ONLY=true
            shift
            ;;
        --features_only)
            FEATURES_ONLY=true
            shift
            ;;
        --no_wandb)
            USE_WANDB=false
            shift
            ;;
        --max_length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --grad_steps)
            GRAD_STEPS="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --base_model NAME        Model name to fine-tune (default: $BASE_MODEL)"
            echo "  --output_dir DIR         Directory to save the fine-tuned model (default: $OUTPUT_DIR)"
            echo "  --config PATH            Path to training config (default: $CONFIG)"
            echo "  --dataset_config PATH    Path to dataset config (default: $DATASET_CONFIG)"
            echo "  --features_dir DIR       Directory to save extracted features (default: $FEATURES_DIR)"
            echo "  --data_dir DIR           Directory containing processed datasets (default: $DATA_DIR)"
            echo "  --no_drive               Don't use Google Drive for datasets and models"
            echo "  --drive_base_dir DIR     Base directory on Google Drive (default: $DRIVE_BASE_DIR)"
            echo "  --device DEVICE          Device to use for training (default: $DEVICE)"
            echo "  --use_deepspeed          Use DeepSpeed for training"
            echo "  --no_4bit                Disable 4-bit quantization"
            echo "  --no_unsloth             Disable Unsloth optimization"
            echo "  --push_to_hub            Push trained model to Hugging Face Hub"
            echo "  --train_only             Skip feature extraction and train directly"
            echo "  --features_only          Only extract features, don't train"
            echo "  --no_wandb               Disable Weights & Biases logging"
            echo "  --max_length LEN         Maximum sequence length (default: $MAX_LENGTH)"
            echo "  --batch_size SIZE        Batch size for training (default: $BATCH_SIZE)"
            echo "  --grad_steps STEPS       Gradient accumulation steps (default: $GRAD_STEPS)"
            echo "  --epochs NUM             Number of training epochs (default: $EPOCHS)"
            echo "  --lr RATE                Learning rate (default: $LEARNING_RATE)"
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

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$FEATURES_DIR"

echo "======================================================"
echo "      Training DeepSeek-Coder with Feature Extraction"
echo "======================================================"
echo ""
echo "Base Model:         $BASE_MODEL"
echo "Output Directory:   $OUTPUT_DIR"
echo "Configuration:      $CONFIG"
echo "Dataset Config:     $DATASET_CONFIG"
echo "Features Directory: $FEATURES_DIR"
echo "Data Directory:     $DATA_DIR"
echo "Use Google Drive:   $USE_DRIVE"
echo "Drive Base Dir:     $DRIVE_BASE_DIR"
echo "Device:             $DEVICE"
echo "Use DeepSpeed:      $USE_DEEPSPEED"
echo "Use 4-bit:          $USE_4BIT"
echo "Use Unsloth:        $USE_UNSLOTH"
echo "Push to Hub:        $PUSH_TO_HUB"
echo "Train Only:         $TRAIN_ONLY"
echo "Features Only:      $FEATURES_ONLY"
echo "Use W&B:            $USE_WANDB"
echo "Max Length:         $MAX_LENGTH"
echo "Batch Size:         $BATCH_SIZE"
echo "Gradient Steps:     $GRAD_STEPS"
echo "Epochs:             $EPOCHS"
echo "Learning Rate:      $LEARNING_RATE"
echo ""
echo "======================================================"

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ] && [ "$PUSH_TO_HUB" = true ]; then
    echo "WARNING: HF_TOKEN environment variable is not set, but push_to_hub is enabled"
    echo "You may encounter issues pushing to Hugging Face Hub"
    echo "Consider setting HF_TOKEN in your .env file or environment"
    echo ""
fi

# Build drive flag
DRIVE_FLAG=""
if [ "$USE_DRIVE" = true ]; then
    DRIVE_FLAG="--from_google_drive --drive_base_dir $DRIVE_BASE_DIR"
fi

# Extract features if not skipping
if [ "$TRAIN_ONLY" = false ]; then
    echo "Starting feature extraction..."
    
    # Run the feature extraction script
    python scripts/prepare_datasets_for_training.py \
        --model_name "$BASE_MODEL" \
        --config "$CONFIG" \
        --dataset_config "$DATASET_CONFIG" \
        --output_dir "$FEATURES_DIR" \
        --text_column "text" \
        --max_length "$MAX_LENGTH" \
        --batch_size 1000 \
        --num_proc 4 \
        $DRIVE_FLAG
    
    echo "Feature extraction complete!"
fi

# Exit if only extracting features
if [ "$FEATURES_ONLY" = true ]; then
    echo "Feature extraction completed. Skipping training as requested."
    exit 0
fi

# Build additional flags for training
QUANTIZE_FLAG=""
if [ "$USE_4BIT" = true ]; then
    QUANTIZE_FLAG="--use_4bit"
fi

UNSLOTH_FLAG=""
if [ "$USE_UNSLOTH" = true ]; then
    UNSLOTH_FLAG="--use_unsloth"
fi

DEEPSPEED_FLAG=""
if [ "$USE_DEEPSPEED" = true ]; then
    DEEPSPEED_FLAG="--use_deepspeed"
fi

WANDB_FLAG=""
if [ "$USE_WANDB" = false ]; then
    WANDB_FLAG="--no_wandb"
fi

HUB_FLAG=""
if [ "$PUSH_TO_HUB" = true ]; then
    HUB_FLAG="--push_to_hub"
fi

# Start training
echo "Starting model training..."

python src/training/train.py \
    --base_model "$BASE_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --config "$CONFIG" \
    --device "$DEVICE" \
    --features_dir "$FEATURES_DIR" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum_steps "$GRAD_STEPS" \
    --num_epochs "$EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    $QUANTIZE_FLAG $UNSLOTH_FLAG $DEEPSPEED_FLAG $WANDB_FLAG $HUB_FLAG $DRIVE_FLAG

echo ""
echo "Training complete! Model saved to $OUTPUT_DIR"
echo "" 