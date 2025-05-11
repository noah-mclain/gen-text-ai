# Quick Training with The Stack

This guide explains how to use The Stack dataset directly for training without extensive preprocessing, focusing on specific programming languages and natural language filters.

## Overview

The Stack is a 6TB+ dataset containing code in 358 programming languages. For time-constrained training, we can:

1. Filter for specific programming languages
2. Filter for specific natural languages in comments/docstrings
3. Sample a subset of the data
4. Optimize training parameters

## Quick Start

To begin training immediately with time constraints (e.g., complete by midnight):

```bash
# Start training with a 5-hour time limit
./scripts/process_stack_direct.sh --max-hours 5
```

This will:

- Filter The Stack for popular languages: python, java, javascript, c, c++, c#, typescript, html, sql, tex, dockerfile
- Only include code with English or Arabic comments/docstrings
- Sample a subset of the data to fit in the time constraint
- Use mixed precision and gradient accumulation for speed

## Customization Options

### Adjusting Time Constraints

```bash
# For very tight deadlines (1-2 hours)
./scripts/process_stack_direct.sh --max-hours 1

# For overnight training (8+ hours)
./scripts/process_stack_direct.sh --max-hours 8
```

### Preprocessing (Optional)

By default, the script skips preprocessing to save time. To enable preprocessing:

```bash
./scripts/process_stack_direct.sh --process
```

### Configuration Files

Use custom configurations:

```bash
./scripts/process_stack_direct.sh --config config/my_dataset_config.json --training config/my_training_config.json
```

## Understanding the Process

1. **Direct Loading**: The script uses The Stack dataset directly from Hugging Face
2. **Streaming**: Data is streamed to avoid downloading the entire dataset
3. **Filtering**:
   - Programming languages: python, java, javascript, c, c++, c#, typescript, html, sql, tex, dockerfile
   - Natural languages: English and Arabic
4. **Sampling**: Only a subset of examples is used (controlled by sampling_ratio)
5. **Optimization**:
   - Reduced sequence length (1024 tokens)
   - Mixed precision training (FP16)
   - Gradient accumulation
   - Limited number of epochs (1)

## Monitoring Training

The script displays:

- Time elapsed and estimated completion time
- Training metrics (loss, perplexity)
- Resource usage (memory, GPU utilization)

## After Training

Your model will be saved in the `models/` directory. You can evaluate it using:

```bash
python main.py --mode evaluate --model_path models/deepseek-coder-finetune --base_model deepseek-ai/deepseek-coder-6.7b-base
```

## Troubleshooting

If you encounter memory issues:

- Try reducing the batch size in the training config
- Use a smaller model or higher quantization (8-bit instead of 4-bit)
- Further reduce the sampling ratio for The Stack dataset

For authentication issues with Hugging Face:

- Ensure your HF_TOKEN is properly set in the environment
- Try using `scripts/set_hf_token.py` to set your token
