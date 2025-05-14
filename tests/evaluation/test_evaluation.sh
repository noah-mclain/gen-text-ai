#!/bin/bash

# Test evaluation with new metrics for deepseek-coder model
# This script is for testing the evaluation features on Paperspace Gradient

# Create necessary directories
mkdir -p results

# Set HF token if needed
# export HF_TOKEN=your_token_here

# Run evaluation on pre-trained model with all metrics
echo "Starting evaluation with advanced metrics..."

python -m src.evaluation.evaluate \
  --model_path deepseek-ai/deepseek-coder-6.7b-instruct \
  --output_dir results \
  --eval_humaneval \
  --eval_code_quality \
  --eval_runtime \
  --eval_semantic \
  --custom_prompts examples/custom_prompts.json

# Generate visualization
echo "Generating visualization..."
python -m src.utils.visualize \
  --training_log results/evaluation_metadata.json \
  --results_dir results \
  --output_dir visualizations

echo "Evaluation test completed. Results saved in 'results' and visualizations in 'visualizations'" 