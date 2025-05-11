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

## Multi-Language Dataset Support

The pipeline now offers significantly improved support for multi-language datasets:

### CodeSearchNet With All Languages

To use CodeSearchNet with all supported languages (python, java, javascript, php, ruby, go) in combination with The Stack:

```bash
python main_api.py \
    --mode quick-stack \
    --datasets the_stack_filtered codesearchnet_all \
    --auto-time \
    --use_drive_api \
    --credentials_path credentials.json \
    --headless
```

### Language-Specific Processing

For more targeted language training, specify individual language datasets:

```bash
python main_api.py \
    --mode quick-stack \
    --datasets codesearchnet_python codesearchnet_javascript the_stack_filtered \
    --auto-time \
    --use_drive_api \
    --credentials_path credentials.json \
    --headless
```

### Memory-Optimized Multi-Language Processing

For systems with memory constraints, use these optimized flags:

```bash
# Low memory systems (8GB RAM)
python main_api.py \
    --mode quick-stack \
    --datasets codesearchnet_all \
    --streaming \
    --no_cache \
    --max_samples 5000 \
    --auto-time \
    --headless

# Medium memory systems (16GB RAM)
python main_api.py \
    --mode quick-stack \
    --datasets codesearchnet_all the_stack_filtered \
    --streaming \
    --max_samples 10000 \
    --auto-time \
    --headless
```

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

## Advanced: Language Distribution Monitoring

The processing pipeline now tracks and reports language distribution, allowing you to see the balance of programming languages in your training data:

```bash
# Process with language distribution monitoring
python main_api.py \
    --mode process \
    --datasets codesearchnet_all \
    --verbose \
    --report_lang_stats
```

This will log language distribution information during processing to help you ensure a balanced dataset.

## Optimized HumanEval Processing

The HumanEval dataset is now processed using direct loading from Hugging Face, ensuring all 164 examples are correctly processed:

```bash
python main_api.py \
    --mode process \
    --datasets humaneval
```

## Combined Dataset Training Example

For the optimal training experience with multi-language support, use:

```bash
python main_api.py \
    --mode quick-stack \
    --datasets the_stack_filtered codesearchnet_all code_alpaca humaneval \
    --auto-time \
    --use_drive_api \
    --credentials_path credentials.json \
    --headless
```

This setup:

1. Processes CodeSearchNet with all languages (properly separated)
2. Adds The Stack with filtering for your selected languages
3. Includes CodeAlpaca for instruction tuning capabilities
4. Adds HumanEval directly from source for evaluation
5. Automatically optimizes for your time constraints
6. Preserves language information throughout processing and training
