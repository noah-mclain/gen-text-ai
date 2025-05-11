#!/usr/bin/env python3
import os
import argparse
import logging

# Try different import approaches to handle various module structures
try:
    from src.training.trainer import DeepseekFineTuner
except ImportError:
    try:
        from trainer import DeepseekFineTuner  # Original import
    except ImportError:
        from .trainer import DeepseekFineTuner  # Relative import

# Import drive utils
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.drive_utils import mount_google_drive, setup_drive_directories, is_drive_mounted

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune deepseek-coder models")
    parser.add_argument("--config", type=str, default="../../config/training_config.json",
                        help="Path to training configuration file")
    parser.add_argument("--data_dir", type=str, default="../../data/processed",
                        help="Directory containing processed datasets")
    parser.add_argument("--use_drive", action="store_true", 
                        help="Use Google Drive for storage")
    parser.add_argument("--drive_base_dir", type=str, default="DeepseekCoder",
                        help="Base directory on Google Drive (if using Drive)")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push the final model to Hugging Face Hub")
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="Model ID for pushing to Hugging Face Hub")
    
    args = parser.parse_args()
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(args.config), exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Setup Google Drive if requested
    drive_paths = None
    if args.use_drive:
        logger.info("Attempting to mount Google Drive...")
        if mount_google_drive():
            logger.info(f"Setting up directories in Google Drive under {args.drive_base_dir}")
            drive_paths = setup_drive_directories(os.path.join("/content/drive/MyDrive", args.drive_base_dir))
            
            # If pushing to HF Hub is enabled through command line, update config
            if args.push_to_hub and os.path.exists(args.config):
                import json
                with open(args.config, 'r') as f:
                    config = json.load(f)
                
                if "training" not in config:
                    config["training"] = {}
                
                config["training"]["push_to_hub"] = True
                if args.hub_model_id:
                    config["training"]["hub_model_id"] = args.hub_model_id
                
                with open(args.config, 'w') as f:
                    json.dump(config, f, indent=2)
                
                logger.info(f"Updated config to push to Hub with model ID: {args.hub_model_id or 'auto-generated'}")
        else:
            logger.warning("Failed to mount Google Drive. Using local storage instead.")
            args.use_drive = False
    
    # Create fine-tuner
    logger.info(f"Initializing fine-tuner with config {args.config}")
    tuner = DeepseekFineTuner(
        args.config, 
        use_drive=args.use_drive, 
        drive_base_dir=os.path.join("/content/drive/MyDrive", args.drive_base_dir) if args.use_drive else None
    )
    
    # Start training
    logger.info(f"Starting training with data from {args.data_dir}")
    metrics = tuner.train(args.data_dir)
    
    logger.info(f"Training completed with metrics: {metrics}")

if __name__ == "__main__":
    main() 