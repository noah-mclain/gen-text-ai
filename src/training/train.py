#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import time  # Added for time tracking

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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    # First try the main implementation
    from src.utils.google_drive_manager import (
        drive_manager, 
        test_authentication, 
        test_drive_mounting, 
        configure_sync_method
    )
    GOOGLE_DRIVE_AVAILABLE = True
    logger.info("Successfully imported Google Drive manager from src.utils")
except (ImportError, ModuleNotFoundError):
    try:
        # Next try the scripts redirect
        from scripts.google_drive_manager import (
            drive_manager,
            test_authentication,
            test_drive_mounting,
            configure_sync_method
        )
        GOOGLE_DRIVE_AVAILABLE = True
        logger.info("Successfully imported Google Drive manager from scripts")
    except (ImportError, ModuleNotFoundError):
        # Last attempt with modified path - but use explicit import
        try:
            # Add utils path to system path
            utils_path = os.path.join(project_root, 'src', 'utils')
            if utils_path not in sys.path:
                sys.path.append(utils_path)
                
            # Use explicit import from src.utils
            try:
                from src.utils.google_drive_manager import (
                    drive_manager,
                    test_authentication,
                    test_drive_mounting,
                    configure_sync_method
                )
                GOOGLE_DRIVE_AVAILABLE = True
                logger.info("Successfully imported Google Drive manager from modified path")
            except (ImportError, ModuleNotFoundError):
                raise  # Re-raise to fall through to the fallback
        except (ImportError, ModuleNotFoundError):
            # Fallback to dummy implementations
            logger.warning("Google Drive manager not found. Using fallback implementations.")
            GOOGLE_DRIVE_AVAILABLE = False
            
            class DummyDriveManager:
                def __init__(self):
                    self.authenticated = False
                    self.folder_ids = {}
                
                def authenticate(self):
                    return False
            
            drive_manager = DummyDriveManager()
            
            def test_authentication():
                """Fallback Google Drive authentication check"""
                logger.warning("Google Drive authentication not available")
                return False
            
            def test_drive_mounting():
                """Fallback for checking if drive is mounted"""
                logger.warning("Google Drive mounting not available")
                return False
            
            def configure_sync_method(base_dir=None):
                """Fallback for configuring sync method"""
                logger.warning(f"Cannot configure sync method for Google Drive under {base_dir}")
                return None

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
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--deepspeed", action="store_true", default=False,
                        help="Enable DeepSpeed (disabled by default)")
    parser.add_argument("--no_deepspeed", action="store_true", 
                        help="Explicitly disable DeepSpeed even if configured in the config file")
    parser.add_argument("--estimate_time", action="store_true", default=True,
                        help="Estimate and report training time (enabled by default)")
    parser.add_argument("--disable_wandb", action="store_true", default=True,
                        help="Disable Weights & Biases logging (enabled by default)")
    
    args = parser.parse_args()
    
    # Set up verbose logging if debug mode is enabled
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("transformers").setLevel(logging.DEBUG)
        logging.getLogger("datasets").setLevel(logging.DEBUG)
    
    # Disable wandb if requested (default is to disable)
    if args.disable_wandb:
        logger.info("Disabling Weights & Biases (wandb) logging")
        os.environ["WANDB_DISABLED"] = "true"
    
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
    
    # Update config file to disable wandb if requested
    if args.disable_wandb and os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            
            if "training" in config:
                config["training"]["report_to"] = "none"
            
            with open(args.config, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info("Updated config to disable wandb reporting")
        except Exception as e:
            logger.warning(f"Error updating config to disable wandb: {e}")
    
    # First check if we're explicitly disabling DeepSpeed
    if args.no_deepspeed:
        logger.info("DeepSpeed is explicitly disabled via --no_deepspeed flag")
        
        # Clean up any existing DeepSpeed environment variables
        logger.info("Cleaning DeepSpeed environment variables...")
        ds_env_vars = [
            "ACCELERATE_USE_DEEPSPEED",
            "ACCELERATE_DEEPSPEED_CONFIG_FILE",
            "ACCELERATE_DEEPSPEED_PLUGIN_TYPE",
            "HF_DS_CONFIG",
            "DEEPSPEED_CONFIG_FILE",
            "DS_ACCELERATOR",
            "DS_OFFLOAD_PARAM",
            "DS_OFFLOAD_OPTIMIZER"
        ]
        for var in ds_env_vars:
            if var in os.environ:
                logger.info(f"Unsetting {var}={os.environ[var]}")
                del os.environ[var]
        
        # Also update the config file to disable DeepSpeed
        if os.path.exists(args.config):
            logger.info("Updating config file to disable DeepSpeed")
            try:
                with open(args.config, 'r') as f:
                    config = json.load(f)
                
                if "training" not in config:
                    config["training"] = {}
                
                config["training"]["use_deepspeed"] = False
                
                with open(args.config, 'w') as f:
                    json.dump(config, f, indent=2)
                
                logger.info("Successfully updated config to disable DeepSpeed")
            except Exception as e:
                logger.warning(f"Failed to update config file: {e}")
        
        # Override the --deepspeed flag
        args.deepspeed = False
    
    # Setup DeepSpeed configuration if enabled
    elif False:
        logger.info("Setting up DeepSpeed environment...")
        
        # Check for existing environment variables
        current_ds_config = os.environ.get("ACCELERATE_DEEPSPEED_CONFIG_FILE", "")
        current_hf_ds_config = os.environ.get("HF_DS_CONFIG", "")
        current_plugin_type = os.environ.get("ACCELERATE_DEEPSPEED_PLUGIN_TYPE", "")
        
        logger.info(f"Current DeepSpeed env vars - ACCELERATE_DEEPSPEED_CONFIG_FILE: {current_ds_config}")
        logger.info(f"Current DeepSpeed env vars - HF_DS_CONFIG: {current_hf_ds_config}")
        logger.info(f"Current DeepSpeed env vars - ACCELERATE_DEEPSPEED_PLUGIN_TYPE: {current_plugin_type}")
        
        # Try to fix DeepSpeed configuration using the dedicated script if available
        ds_fix_script = os.path.join(project_root, "scripts", "fix_deepspeed.py")
        if os.path.exists(ds_fix_script):
            logger.info("Running DeepSpeed configuration fix script")
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("fix_deepspeed", ds_fix_script)
                fix_deepspeed = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(fix_deepspeed)
                fix_deepspeed.fix_deepspeed_config()
                logger.info("DeepSpeed configuration fixed")
            except Exception as e:
                logger.error(f"Failed to run DeepSpeed fix script: {e}")
        
        # Set basic DeepSpeed environment variables if not already set
        if not os.environ.get("ACCELERATE_USE_DEEPSPEED"):
            os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"
            logger.info("Set ACCELERATE_USE_DEEPSPEED to 'true'")
            
        if not os.environ.get("ACCELERATE_DEEPSPEED_PLUGIN_TYPE"):
            os.environ["ACCELERATE_DEEPSPEED_PLUGIN_TYPE"] = "deepspeed"
            logger.info("Set ACCELERATE_DEEPSPEED_PLUGIN_TYPE to 'deepspeed'")
        
        # Check for explicit DeepSpeed config file argument
        if False_config and os.path.exists(args.deepspeed_config):
            logger.info(f"Using specified DeepSpeed config: {args.deepspeed_config}")
            os.environ["ACCELERATE_DEEPSPEED_CONFIG_FILE"] = args.deepspeed_config
            os.environ["HF_DS_CONFIG"] = args.deepspeed_config
        else:
            # Check for common locations for DeepSpeed config
            default_paths = [
                os.path.join(os.getcwd(), "ds_config_a6000.json"),
                os.path.join(project_root, "ds_config_a6000.json"),
                "/notebooks/ds_config_a6000.json" if os.path.exists("/notebooks") else None
            ]
            
            # Filter out None values
            default_paths = [p for p in default_paths if p]
            
            # Use the first valid config
            for path in default_paths:
                if os.path.exists(path):
                    logger.info(f"Found DeepSpeed config at: {path}")
                    os.environ["ACCELERATE_DEEPSPEED_CONFIG_FILE"] = path
                    os.environ["HF_DS_CONFIG"] = path
                    break
        
        # Double check we have a valid configuration
        if not os.environ.get("ACCELERATE_DEEPSPEED_CONFIG_FILE") or not os.path.exists(os.environ["ACCELERATE_DEEPSPEED_CONFIG_FILE"]):
            logger.warning("No valid DeepSpeed config found in environment variables or default locations")
        else:
            logger.info(f"Using DeepSpeed config at: {os.environ.get('ACCELERATE_DEEPSPEED_CONFIG_FILE')}")
    
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
    
    # Try/except around the entire training process to catch and log any errors
    try:
        # Create fine-tuner
        logger.info(f"Initializing fine-tuner with config {args.config}")
        tuner = DeepseekFineTuner(
            args.config, 
            use_drive=args.use_drive, 
            drive_base_dir=args.drive_base_dir
        )
        
        # Load config to estimate training time
        if args.estimate_time:
            try:
                with open(args.config, 'r') as f:
                    config = json.load(f)
                
                # Get training parameters
                train_config = config.get("training", {})
                epochs = train_config.get("num_train_epochs", 3)
                batch_size = train_config.get("per_device_train_batch_size", 1)
                grad_accum = train_config.get("gradient_accumulation_steps", 16)
                effective_batch = batch_size * grad_accum
                
                # Get model name/info
                model_name = train_config.get("model_name_or_path", "unknown")
                
                # Estimate steps and time
                estimated_steps = train_config.get("max_steps", 0)
                if estimated_steps <= 0:  # If max_steps not set, use epochs
                    # Rough estimate - assume 5000 samples per dataset
                    num_datasets = len(config.get("dataset_weights", {}).keys())
                    if num_datasets == 0:
                        num_datasets = 1
                    estimated_samples = 5000 * num_datasets
                    estimated_steps = (estimated_samples * epochs) // effective_batch
                
                # Base estimate - seconds per step depends on model size
                seconds_per_step = 0.5  # Default for small models
                if "deepseek-coder-6.7b" in model_name:
                    seconds_per_step = 1.0
                elif "deepseek-coder-33b" in model_name:
                    seconds_per_step = 4.0
                elif "ul2" in model_name:
                    seconds_per_step = 1.2
                
                estimated_seconds = estimated_steps * seconds_per_step * 1.15  # 15% buffer
                estimated_hours = estimated_seconds / 3600
                
                # Calculate expected completion time
                now = time.time()
                completion_time = now + estimated_seconds
                completion_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(completion_time))
                
                logger.info(f"Estimated training steps: {estimated_steps}")
                logger.info(f"Estimated training time: {estimated_hours:.1f} hours")
                logger.info(f"Expected completion time: {completion_str}")
                
                # Record start time for final calculation
                training_start_time = time.time()
                
            except Exception as e:
                logger.warning(f"Error estimating training time: {e}")
        
        # Start training
        logger.info(f"Starting training with data from {args.data_dir}")
        metrics = tuner.train(args.data_dir)
        
        # Calculate actual training time if we recorded start time
        if args.estimate_time and 'training_start_time' in locals():
            training_end_time = time.time()
            actual_training_time = training_end_time - training_start_time
            actual_hours = actual_training_time / 3600
            logger.info(f"Actual training time: {actual_hours:.2f} hours")
            if 'estimated_hours' in locals():
                diff_pct = (actual_hours / estimated_hours - 1.0) * 100
                logger.info(f"Time estimate {'overestimated' if diff_pct < 0 else 'underestimated'} by {abs(diff_pct):.1f}%")
            
            # Add timing info to metrics
            if metrics is None:
                metrics = {}
            metrics['training_time_hours'] = actual_hours
            if 'estimated_hours' in locals():
                metrics['estimated_hours'] = estimated_hours
        
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
                import json
                error_info = {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "timestamp": datetime.now().isoformat()
                }
                error_file = os.path.join(tuner.training_config["output_dir"], "training_error.json")
                os.makedirs(os.path.dirname(error_file), exist_ok=True)
                with open(error_file, 'w') as f:
                    json.dump(error_info, f, indent=2)
                logger.info(f"Error information saved to {error_file}")
        except Exception as save_error:
            logger.error(f"Failed to save error information: {save_error}")
        
        return 1  # Return error code

if __name__ == "__main__":
    sys.exit(main()) 