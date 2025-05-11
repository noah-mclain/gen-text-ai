# Using The Stack Dataset Directly

This guide explains how to use the filtered Stack dataset with your existing command structure directly within main_api.py.

## Quick Start - One Command Solution

I've integrated all the TRAIN_STACK_NOW.sh functionality directly into main_api.py. Use the new `quick-stack` mode:

```bash
python main_api.py \
    --mode quick-stack \
    --auto-time \
    --dataset_config config/dataset_config.json \
    --training_config config/training_config.json \
    --use_drive_api \
    --credentials_path credentials.json \
    --drive_base_dir DeepseekCoder \
    --headless
```

This command:

- Uses the `quick-stack` mode to activate all time-optimized functionality
- Automatically calculates hours until midnight with `--auto-time`
- Uses the filtered Stack dataset with specific languages
- Optimizes training parameters for time constraints
- Handles both processing and training in one step

## Time Constraint Options

Control the training time directly:

```bash
# Specify exact number of hours
python main_api.py --mode quick-stack --max-hours 4 ...

# Let it calculate hours until midnight automatically
python main_api.py --mode quick-stack --auto-time ...
```

## Preprocessing Control

Skip preprocessing to use direct loading (saves time):

```bash
python main_api.py --mode quick-stack --skip-preprocessing ...
```

## Process First, Then Train

You can still use your original approach if preferred:

```bash
# Step 1: Process only the filtered Stack dataset
python main_api.py \
    --mode process \
    --dataset_config config/dataset_config.json \
    --datasets the_stack_filtered \
    --streaming \
    --no_cache \
    --use_drive_api \
    --credentials_path credentials.json \
    --drive_base_dir DeepseekCoder \
    --headless

# Step 2: Train with the processed data
python main_api.py \
    --mode train \
    --training_config config/training_config.json \
    --streaming \
    --use_drive_api \
    --credentials_path credentials.json \
    --drive_base_dir DeepseekCoder \
    --headless
```

## Advanced: Specify Multiple Datasets with Quick-Stack Mode

You can mix the filtered Stack dataset with other datasets in quick-stack mode:

```bash
python main_api.py \
    --mode quick-stack \
    --datasets the_stack_filtered mbpp code_alpaca \
    --max-hours 5 \
    --use_drive_api \
    --credentials_path credentials.json \
    --drive_base_dir DeepseekCoder \
    --headless
```

This makes it easy to combine datasets while still using the time-optimized training.
