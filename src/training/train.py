#!/usr/bin/env python3
import os
import sys
import argparse
import logging

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
sys.path.append(os.path.dirname(project_root))  # In case notebook is one level up

# Fix import order to ensure Unsloth optimizations are applied correctly
try:
    from unsloth import FastLanguageModel
except ImportError:
    print("Unsloth not installed. Install with: pip install unsloth")

# Standard imports
import json
import torch
from datetime import datetime
from typing import Dict, Any, List, Optional

# Import other libraries after unsloth
import transformers
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
import peft
from peft import LoraConfig

# Import trainer directly with absolute path reference
try:
    # Try to import the trainer as absolute path when run from project root
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.training.trainer import DeepseekFineTuner
except (ImportError, ModuleNotFoundError):
    try:
        # Try direct import from current directory
        from trainer import DeepseekFineTuner
    except (ImportError, ModuleNotFoundError):
        try:
            # Try relative import
            from .trainer import DeepseekFineTuner
        except (ImportError, ModuleNotFoundError):
            # Last resort: create a simple wrapper class as fallback
            print("WARNING: Could not import DeepseekFineTuner - creating minimal fallback class")
            
            class DeepseekFineTuner:
                """Fallback class when trainer.py cannot be imported"""
                def __init__(self, config_path, use_drive=False, drive_base_dir=None):
                    self.config_path = config_path
                    self.use_drive = use_drive
                    self.drive_base_dir = drive_base_dir
                    
                    # Load configuration
                    try:
                        with open(config_path, 'r') as f:
                            self.config = json.load(f)
                    except Exception as e:
                        print(f"Error loading config: {str(e)}")
                        self.config = {}
                    
                    print(f"Initialized fallback DeepseekFineTuner with config: {config_path}")
                
                def train(self, data_dir):
                    """Fallback training method"""
                    print(f"WARNING: Using fallback training with data from {data_dir}")
                    print("This is a minimal implementation as the real trainer could not be imported")
                    
                    try:
                        # Minimal dataset loading
                        from datasets import load_from_disk
                        datasets = {}
                        
                        # Try to load at least one dataset
                        for entry in os.listdir(data_dir):
                            path = os.path.join(data_dir, entry)
                            if os.path.isdir(path) and entry.endswith("_processed"):
                                try:
                                    dataset = load_from_disk(path)
                                    print(f"Loaded dataset from {path} with {len(dataset)} examples")
                                    return {"success": False, "error": "Training aborted - proper trainer not available"}
                                except Exception:
                                    pass
                    except Exception as e:
                        print(f"Fallback training failed: {str(e)}")
                    
                    return {"success": False, "error": "Training aborted - proper trainer not available"}

# Import drive utils with fallbacks
try:
    from src.utils.drive_utils import mount_google_drive, setup_drive_directories
except (ImportError, ModuleNotFoundError):
    try:
        # Try relative import
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.drive_utils import mount_google_drive, setup_drive_directories
    except (ImportError, ModuleNotFoundError):
        print("WARNING: drive_utils not found. Creating fallback functions.")
        
        def mount_google_drive():
            """Fallback Google Drive mounting function"""
            print("Google Drive mounting not available")
            return False
        
        def setup_drive_directories(base_dir):
            """Fallback for setting up Google Drive directories"""
            print(f"Cannot set up directories in Google Drive under {base_dir}")
            return None

# Fallback for drive mounting check
try:
    from src.utils.drive_utils import is_drive_mounted
except (ImportError, ModuleNotFoundError):
    try:
        from utils.drive_utils import is_drive_mounted
    except (ImportError, ModuleNotFoundError):
        def is_drive_mounted():
            """Fallback for checking if drive is mounted"""
            return False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
    
    # Ensure absolute paths for config and data_dir
    if not os.path.isabs(args.config):
        if args.config.startswith("../") or args.config.startswith("./"):
            # Resolve relative paths
            args.config = os.path.abspath(os.path.join(current_dir, args.config))
        else:
            # Assume paths relative to project root if not explicitly relative
            args.config = os.path.join(project_root, args.config)
    
    if not os.path.isabs(args.data_dir):
        if args.data_dir.startswith("../") or args.data_dir.startswith("./"):
            # Resolve relative paths
            args.data_dir = os.path.abspath(os.path.join(current_dir, args.data_dir))
        else:
            # Assume paths relative to project root if not explicitly relative
            args.data_dir = os.path.join(project_root, args.data_dir)
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(args.config), exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    
    logger.info(f"Using config path: {args.config}")
    logger.info(f"Using data directory: {args.data_dir}")
    
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
    return 0  # Return success code

if __name__ == "__main__":
    sys.exit(main()) 