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
    UNSLOTH_AVAILABLE = True
except (ImportError, NotImplementedError) as e:
    print(f"Unsloth not available: {e}")
    print("Training will continue without Unsloth optimizations.")
    UNSLOTH_AVAILABLE = False

# Import the trainer module
try:
    from src.training.text_trainer import FlanUL2TextTrainer
except (ImportError, ModuleNotFoundError):
    print("Error importing FlanUL2TextTrainer. Make sure src/training/text_trainer.py exists.")
    sys.exit(1)

# Import google_drive_manager with fallbacks
try:
    from src.utils.google_drive_manager import (
        drive_manager, 
        test_authentication, 
        test_drive_mounting, 
        configure_sync_method
    )
    GOOGLE_DRIVE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    print("WARNING: google_drive_manager not found. Google Drive functionality will be limited.")
    GOOGLE_DRIVE_AVAILABLE = False
    
    # Create fallback functions
    def test_authentication():
        """Fallback Google Drive authentication check"""
        print("Google Drive authentication not available")
        return False
    
    def test_drive_mounting():
        """Fallback for checking if drive is mounted"""
        print("Google Drive mounting not available")
        return False
    
    def configure_sync_method(base_dir=None):
        """Fallback for configuring sync method"""
        print(f"Cannot configure sync method for Google Drive under {base_dir}")
        return None

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
    parser.add_argument("--no_deepspeed", action="store_true",
                        help="Explicitly disable DeepSpeed even if configured in the config file")
    
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
    
    # Check for HF_TOKEN environment variable
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token and args.push_to_hub:
        logger.warning("HF_TOKEN environment variable not found but --push_to_hub is enabled.")
        logger.warning("You may encounter authentication issues when pushing to Hugging Face Hub.")
        logger.warning("Set the token with: export HF_TOKEN=your_huggingface_token")
    elif hf_token:
        logger.info("HF_TOKEN environment variable found. Will use for Hugging Face Hub operations.")
        # Set as default token for Hugging Face
        try:
            import huggingface_hub
            huggingface_hub.login(token=hf_token)
            logger.info("Logged in to Hugging Face Hub")
        except ImportError:
            logger.warning("huggingface_hub package not found. Install with: pip install huggingface-hub")
    
    # Setup Google Drive if requested
    if args.use_drive and GOOGLE_DRIVE_AVAILABLE:
        logger.info("Attempting to set up Google Drive integration...")
        try:
            # Configure the drive manager with the base directory
            configure_sync_method(base_dir=args.drive_base_dir)
            
            # Test authentication
            if test_authentication():
                logger.info("Successfully authenticated with Google Drive")
                
                # Test drive access
                if test_drive_mounting():
                    logger.info(f"Google Drive integration set up successfully with base directory: {args.drive_base_dir}")
                
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
                    
                    # If --no_deepspeed flag is present, disable DeepSpeed
                    if args.no_deepspeed:
                        logger.info("Explicitly disabling DeepSpeed as requested via --no_deepspeed flag")
                        config["training"]["use_deepspeed"] = False
                    
                    with open(args.config, 'w') as f:
                        json.dump(config, f, indent=2)
                    
                    logger.info(f"Updated config to push to Hub with model ID: {args.hub_model_id or 'auto-generated'}")
                else:
                        logger.warning("Failed to access Google Drive. Using local storage instead.")
                        args.use_drive = False
            else:
                logger.warning("Failed to authenticate with Google Drive. Using local storage instead.")
                args.use_drive = False
        except Exception as e:
            logger.error(f"Error during Google Drive setup: {e}")
            logger.warning("Failed to setup Google Drive. Using local storage instead.")
            args.use_drive = False
    elif args.use_drive and not GOOGLE_DRIVE_AVAILABLE:
        logger.warning("Google Drive integration not available. Using local storage instead.")
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
            drive_base_dir=args.drive_base_dir
        )
        
        # Explicitly disable DeepSpeed if requested
        if args.no_deepspeed and hasattr(tuner, 'training_config') and 'use_deepspeed' in tuner.training_config:
            logger.info("Disabling DeepSpeed as requested via --no_deepspeed flag")
            tuner.training_config['use_deepspeed'] = False
        
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
        dataset_config_path = config_path.replace('training_config_text.json', 'dataset_config_text.json')
        print(f"Loading dataset config from: {dataset_config_path}")
        with open(dataset_config_path, 'r') as f:
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
            
            # Get processor name and dataset path
            processor_name = config.get('processor')
            dataset_path = config.get('path')
            split = config.get('split', 'train')
            
            if not processor_name:
                logger.warning(f"No processor specified for {dataset_name}, skipping")
                continue
                
            if not dataset_path:
                logger.warning(f"No dataset path specified for {dataset_name}, skipping")
                continue
            
            # Get the processor function
            processor_func_name = f"{processor_name}_processor"
            processor_func = getattr(processors_module, processor_func_name, None)
            
            if not processor_func:
                logger.error(f"Processor {processor_func_name} not found in text_processors module")
                success = False
                continue
            
            try:
                # Process the dataset
                logger.info(f"Processing {dataset_name} with {processor_func_name}")
                processor_func(
                    dataset_path=dataset_path,
                    output_dir=output_dir,
                    split=split,
                    streaming=True,  # Use streaming to save memory
                    force_reprocess=True  # Force reprocessing to ensure correct format
                )
                logger.info(f"Successfully processed {dataset_name}")
            except Exception as e:
                logger.error(f"Error processing {dataset_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                success = False
    
    return success

if __name__ == "__main__":
    sys.exit(main()) 