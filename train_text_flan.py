#!/usr/bin/env python3
import os
import sys
import argparse
import logging
from datetime import datetime

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Fix import order to ensure Unsloth optimizations are applied correctly
try:
    from unsloth import FastLanguageModel
except ImportError:
    print("Unsloth not installed. Install with: pip install unsloth")

# Import the trainer module
try:
    from src.training.text_trainer import FlanUL2TextTrainer
except (ImportError, ModuleNotFoundError):
    print("Error importing FlanUL2TextTrainer. Make sure src/training/text_trainer.py exists.")
    sys.exit(1)

# Import drive utils with fallbacks
try:
    from src.utils.drive_utils import mount_google_drive, setup_drive_directories, is_drive_mounted
except (ImportError, ModuleNotFoundError):
    print("WARNING: drive_utils not found. Google Drive functionality will be limited.")
    
    def mount_google_drive():
        """Fallback Google Drive mounting function"""
        print("Google Drive mounting not available")
        return False
    
    def setup_drive_directories(base_dir):
        """Fallback for setting up Google Drive directories"""
        print(f"Cannot set up directories in Google Drive under {base_dir}")
        return None
    
    def is_drive_mounted():
        """Fallback for checking if drive is mounted"""
        return False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune FLAN-UL2 model for text and story generation")
    parser.add_argument("--config", type=str, default="config/training_config_text.json",
                        help="Path to training configuration file")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                        help="Directory containing processed datasets")
    parser.add_argument("--use_drive", action="store_true", 
                        help="Use Google Drive for storage")
    parser.add_argument("--drive_base_dir", type=str, default="FlanUL2Text",
                        help="Base directory on Google Drive (if using Drive)")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push the final model to Hugging Face Hub")
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="Model ID for pushing to Hugging Face Hub")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--process_only", action="store_true",
                        help="Only process datasets without training")
    
    args = parser.parse_args()
    
    # Set up verbose logging if debug mode is enabled
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("transformers").setLevel(logging.DEBUG)
        logging.getLogger("datasets").setLevel(logging.DEBUG)
    
    # Ensure absolute paths for config and data_dir
    if not os.path.isabs(args.config):
        if args.config.startswith("../") or args.config.startswith("./"):
            # Resolve relative paths
            args.config = os.path.abspath(os.path.join(current_dir, args.config))
        else:
            # Assume paths relative to project root if not explicitly relative
            args.config = os.path.join(current_dir, args.config)
    
    if not os.path.isabs(args.data_dir):
        if args.data_dir.startswith("../") or args.data_dir.startswith("./"):
            # Resolve relative paths
            args.data_dir = os.path.abspath(os.path.join(current_dir, args.data_dir))
        else:
            # Assume paths relative to project root if not explicitly relative
            args.data_dir = os.path.join(current_dir, args.data_dir)
    
    # Check if config file exists
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        return 1
        
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        logger.warning(f"Data directory not found: {args.data_dir}, creating it")
        try:
            os.makedirs(args.data_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create data directory: {e}")
            return 1
    
    logger.info(f"Using config path: {args.config}")
    logger.info(f"Using data directory: {args.data_dir}")
    
    # Setup Google Drive if requested
    drive_paths = None
    if args.use_drive:
        logger.info("Attempting to mount Google Drive...")
        try:
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
        except Exception as e:
            logger.error(f"Error during Google Drive setup: {e}")
            logger.warning("Failed to setup Google Drive. Using local storage instead.")
            args.use_drive = False
    
    # Process datasets if needed
    if args.process_only or not os.listdir(args.data_dir):
        logger.info("Processing datasets...")
        process_datasets(args.config, args.data_dir, args.use_drive, args.drive_base_dir)
        
        if args.process_only:
            logger.info("Dataset processing completed. Exiting as --process_only flag was set.")
            return 0
    
    # Try/except around the entire training process to catch and log any errors
    try:
        # Create fine-tuner
        logger.info(f"Initializing FLAN-UL2 text trainer with config {args.config}")
        tuner = FlanUL2TextTrainer(
            args.config, 
            use_drive=args.use_drive, 
            drive_base_dir=os.path.join("/content/drive/MyDrive", args.drive_base_dir) if args.use_drive else None
        )
        
        # Start training
        logger.info(f"Starting training with data from {args.data_dir}")
        metrics = tuner.train(args.data_dir)
        
        logger.info(f"Training completed with metrics: {metrics}")
        return 0  # Return success code
    except Exception as e:
        # Catch and log any errors during the training process
        logger.error(f"Training failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Try to save minimal checkpoint information if we have a tuner
        try:
            if 'tuner' in locals() and hasattr(tuner, 'config') and 'output_dir' in tuner.training_config:
                error_log_path = os.path.join(tuner.training_config['output_dir'], 'error_log.txt')
                with open(error_log_path, 'w') as f:
                    f.write(f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}")
                logger.info(f"Saved error log to {error_log_path}")
        except Exception:
            pass
        
        return 1  # Return error code

def process_datasets(config_path, output_dir, use_drive=False, drive_base_dir=None):
    """Process datasets for fine-tuning."""
    import json
    from importlib import import_module
    
    # Load the dataset configuration
    try:
        with open(config_path.replace('training_config_text.json', 'dataset_config_text.json'), 'r') as f:
            dataset_config = json.load(f)
    except Exception as e:
        logger.error(f"Error loading dataset config: {e}")
        return False
    
    # Import dataset processors for text generation
    try:
        processors_module = import_module('src.data.processors.text_processors')
    except ImportError:
        logger.error("Could not import text processors module. Make sure src/data/processors/text_processors.py exists.")
        return False
    
    # Process each dataset
    success = True
    for dataset_name, config in dataset_config.items():
        if config.get('enabled', True):
            logger.info(f"Processing dataset: {dataset_name}")
            
            processor_name = config.get('processor')
            if not hasattr(processors_module, 'PROCESSOR_MAP') or processor_name not in processors_module.PROCESSOR_MAP:
                logger.error(f"No processor found for {dataset_name} with processor {processor_name}")
                success = False
                continue
                
            try:
                # Get the processor function
                processor_fn = processors_module.PROCESSOR_MAP[processor_name]
                
                # Process the dataset
                processor_fn(
                    dataset_path=config.get('path'),
                    output_dir=output_dir,
                    split=config.get('split', 'train'),
                    streaming=True,  # Use streaming for memory efficiency
                    use_cache=False   # Don't use cache to avoid disk space issues
                )
                
                logger.info(f"Successfully processed {dataset_name}")
            except Exception as e:
                logger.error(f"Error processing dataset {dataset_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                success = False
    
    return success

if __name__ == "__main__":
    sys.exit(main()) 