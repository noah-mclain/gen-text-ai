#!/bin/bash
# Script to fix wandb integration and run training

# Make sure wandb is properly installed
pip install wandb --upgrade

# Login to wandb
wandb login 636def25145822bffada9359d3c3b3ace65380a4

# Ensure wandb is enabled
export WANDB_DISABLED=false

# Check if WANDB_API_KEY is set
if [ -z "$WANDB_API_KEY" ]; then
  export WANDB_API_KEY="636def25145822bffada9359d3c3b3ace65380a4"
  echo "Set WANDB_API_KEY environment variable"
fi

# Choose which script to run
if [ "$1" = "text" ]; then
  echo "Running text model training with wandb enabled..."
  
  # Edit the config file to ensure wandb is in the report_to list
  python -c "
import json
with open('config/training_config_text.json', 'r') as f:
    config = json.load(f)
if 'training' not in config:
    config['training'] = {}
config['training']['report_to'] = ['wandb', 'tensorboard']
with open('config/training_config_text.json', 'w') as f:
    json.dump(config, f, indent=2)
print('Ensured wandb is enabled in text config')
"
  
  # Run the text training script with explicit wandb flag
  ./train_text_flan_a6000.sh
else
  echo "Running code model training with wandb enabled..."
  
  # Edit the config file to ensure wandb is in the report_to list
  python -c "
import json
with open('config/training_config.json', 'r') as f:
    config = json.load(f)
if 'training' not in config:
    config['training'] = {}
config['training']['report_to'] = ['wandb', 'tensorboard']
with open('config/training_config.json', 'w') as f:
    json.dump(config, f, indent=2)
print('Ensured wandb is enabled in code config')
"
  
  # Run the code training script
  python -m src.training.train \
    --config config/training_config.json \
    --data_dir data/processed \
    --no_deepspeed
fi 