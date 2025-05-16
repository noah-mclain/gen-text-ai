#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import time  # Added for time tracking
from pathlib import Path
import json  # Explicitly import json at the top level

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels from src/training to project root
sys.path.append(project_root)

# Add notebooks directory to path if running in Paperspace
if os.path.exists("/notebooks"):
    notebook_root = "/notebooks"
    if notebook_root not in sys.path:
        sys.path.append(notebook_root)
    # Add a symlink if the script is not found in /notebooks
    train_src_path = os.path.join("/notebooks", "scripts", "src", "training")
    if not os.path.exists(train_src_path):
        os.makedirs(train_src_path, exist_ok=True)
        try:
            if not os.path.exists(os.path.join(train_src_path, "train.py")):
                os.symlink(os.path.abspath(__file__), os.path.join(train_src_path, "train.py"))
        except Exception as e:
            logger.warning(f"Failed to create symlink: {e}")
    
    # Also create symlinks for trainer.py if it's not already there
    trainer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trainer.py")
    if os.path.exists(trainer_path):
        trainer_dest = os.path.join(train_src_path, "trainer.py")
        if not os.path.exists(trainer_dest):
            try:
                os.symlink(trainer_path, trainer_dest)
                logger.info(f"Created symlink for trainer.py at {trainer_dest}")
            except Exception as e:
                logger.warning(f"Failed to create trainer.py symlink: {e}")

# Fix import order to ensure Unsloth optimizations are applied correctly
try:
    # Only import unsloth if CUDA is available
    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except:
        pass
    
    if cuda_available:
        from unsloth import FastLanguageModel
        logger.info("Unsloth imported successfully with CUDA support")
    else:
        logger.warning("CUDA not available, skipping Unsloth import")
except ImportError:
    logger.warning("Unsloth not installed. Install with: pip install unsloth")

# Standard imports
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

# Monkey patch DataLoader to always use limited workers
original_dataloader = torch.utils.data.DataLoader
def safe_dataloader(*args, **kwargs):
    # Always use 1 worker to avoid 'Too many workers' error
    kwargs['num_workers'] = int(os.environ.get('DATALOADER_NUM_WORKERS', 1))
    return original_dataloader(*args, **kwargs)
torch.utils.data.DataLoader = safe_dataloader

# Import trainer directly with absolute path reference
try:
    # First check if we're in the same directory as the trainer
    if os.path.exists(os.path.join(os.path.dirname(__file__), "trainer.py")):
        # We're in the same directory, so import directly
        from trainer import DeepseekFineTuner
    else:
        # Try to import from src.training
        from src.training.trainer import DeepseekFineTuner
    
    # Also try to import dataset mapping helper
    try:
        from src.training.dataset_mapping import get_dataset_paths, find_matching_directories
        logger.info("Successfully imported dataset mapping module")
    except (ImportError, ModuleNotFoundError):
        try:
            # Try relative import
            from dataset_mapping import get_dataset_paths, find_matching_directories
            logger.info("Successfully imported dataset mapping module")
        except (ImportError, ModuleNotFoundError):
            logger.warning("Could not import dataset mapping module, datasets may not be found correctly")
            # Define simple fallback function
            def get_dataset_paths(dataset_names):
                return {}
            def find_matching_directories(data_dir, pattern):
                return []
except (ImportError, ModuleNotFoundError) as e:
    logger.error(f"Error importing DeepseekFineTuner: {e}")
    try:
        # Try with direct path
        trainer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trainer.py")
        if os.path.exists(trainer_path):
            # Add the directory to path
            sys.path.append(os.path.dirname(trainer_path))
            from trainer import DeepseekFineTuner
            logger.info("Successfully imported DeepseekFineTuner from direct path")
        else:
            # Last resort: create a simple wrapper class as fallback
            logger.error("WARNING: Could not import DeepseekFineTuner - creating minimal fallback class")
            
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
                        logger.error(f"Error loading config: {str(e)}")
                        self.config = {}
                    
                    logger.info(f"Initialized fallback DeepseekFineTuner with config: {config_path}")
                
                def train(self, data_dir):
                    """Fallback training method"""
                    logger.error(f"WARNING: Using fallback training with data from {data_dir}")
                    logger.error("This is a minimal implementation as the real trainer could not be imported")
                    
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
                                    logger.info(f"Loaded dataset from {path} with {len(dataset)} examples")
                                    return {"success": False, "error": "Training aborted - proper trainer not available"}
                                except Exception:
                                    pass
                    except Exception as e:
                        logger.error(f"Fallback training failed: {str(e)}")
                    
                    return {"success": False, "error": "Training aborted - proper trainer not available"}
    except (ImportError, ModuleNotFoundError):
        logger.error("Failed to import or create DeepseekFineTuner class. Training is not possible.")

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
        from scripts.google_drive.google_drive_manager import (
            drive_manager,
            test_authentication,
            test_drive_mounting,
            configure_sync_method
        )
        GOOGLE_DRIVE_AVAILABLE = True
        logger.info("Successfully imported Google Drive manager from scripts")
    except (ImportError, ModuleNotFoundError):
        # Try to find the google_drive_manager module in various locations
        potential_paths = [
            os.path.join(project_root, 'src', 'utils'),
            os.path.join(project_root, 'scripts', 'google_drive'),
            os.path.join(notebook_root, 'src', 'utils') if 'notebook_root' in locals() else None,
            os.path.join(notebook_root, 'scripts', 'google_drive') if 'notebook_root' in locals() else None
        ]
        
        module_found = False
        for path in potential_paths:
            if path and os.path.exists(path):
                if path not in sys.path:
                    sys.path.append(path)
                try:
                    if os.path.exists(os.path.join(path, 'google_drive_manager.py')):
                        # Try importing from the adjusted path
                        if 'utils' in path:
                            from google_drive_manager import (
                                drive_manager, test_authentication, 
                                test_drive_mounting, configure_sync_method
                            )
                        else:
                            from google_drive_manager import (
                                drive_manager, test_authentication, 
                                test_drive_mounting, configure_sync_method
                            )
                        GOOGLE_DRIVE_AVAILABLE = True
                        module_found = True
                        logger.info(f"Successfully imported Google Drive manager from {path}")
                        break
                except ImportError:
                    continue
        
        if not module_found:
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

# Try to import datasets functions safely
try:
    from datasets import load_from_disk, concatenate_datasets
    logger.info("Successfully imported datasets functions")
except ImportError:
    logger.warning("Could not import datasets module. Some functionality may be limited.")

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
    parser.add_argument("--disable_wandb", action="store_true", default=False,
                        help="Disable Weights & Biases logging (disabled by default)")
    parser.add_argument("--dataloader_workers", type=int, default=1,
                        help="Number of workers for DataLoader")
    parser.add_argument('--features_dir', type=str, default=None,
                    help='Directory containing extracted features for training')
    parser.add_argument('--skip_drive', action="store_true", 
                        help='Skip using Google Drive for datasets')
    
    args = parser.parse_args()
    
    # Set dataloader workers environment variable from argument
    os.environ['DATALOADER_NUM_WORKERS'] = str(args.dataloader_workers)
    logger.info(f"Setting dataloader workers to {args.dataloader_workers}")
    
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
    
    # Store skip_drive argument for later use
    skip_drive = args.skip_drive
    
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
    
    # Apply dataset validation settings to avoid None issues
    if os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            
            # Add monkey patch for dataset collator to handle None values
            logger.info("Adding dataset safety filter to handle None values in datasets")
            
            # Monkey patch DataCollatorForLanguageModeling to handle None values
            original_call = transformers.DataCollatorForLanguageModeling.__call__
            
            def safe_call(self, features):
                try:
                    # Filter out None values
                    valid_features = []
                    for feature in features:
                        if feature is None:
                            continue
                        
                        # Check if any required keys have None values
                        has_none = False
                        for key in ["input_ids", "attention_mask", "labels"]:
                            if key in feature and feature[key] is None:
                                has_none = True
                                break
                        
                        if not has_none:
                            valid_features.append(feature)
                    
                    # If we filtered out items, log a warning
                    if len(valid_features) < len(features):
                        logger.debug(f"Filtered out {len(features) - len(valid_features)} samples with None values")
                    
                    # If no valid features, create a dummy valid batch with a warning
                    if not valid_features and features:
                        logger.warning("No valid features in batch - creating dummy batch")
                        # Use the first feature as template and replace None values with empty values
                        dummy_feature = {}
                        for key, value in features[0].items():
                            if value is None:
                                if key == "input_ids" or key == "labels":
                                    dummy_feature[key] = torch.zeros(1, dtype=torch.long)
                                elif key == "attention_mask":
                                    dummy_feature[key] = torch.ones(1, dtype=torch.long)
                                else:
                                    dummy_feature[key] = value
                            else:
                                dummy_feature[key] = value
                        valid_features = [dummy_feature]
                    
                    # Call the original implementation with valid features
                    return original_call(self, valid_features)
                except Exception as e:
                    logger.warning(f"Error in data collator: {e}")
                    # Return an empty batch as last resort
                    return {"input_ids": torch.zeros((1, 1), dtype=torch.long),
                            "attention_mask": torch.ones((1, 1), dtype=torch.long),
                            "labels": torch.zeros((1, 1), dtype=torch.long)}
            
            # Apply the monkey patch
            transformers.DataCollatorForLanguageModeling.__call__ = safe_call
            
            logger.info("Added safety filter for dataset processing")
            
        except Exception as e:
            logger.warning(f"Error updating config for dataset validation: {e}")
    
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
    
    # Setup Google Drive if requested and not explicitly skipped
    if args.use_drive and not skip_drive and GOOGLE_DRIVE_AVAILABLE:
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
    elif args.use_drive and skip_drive:
        logger.info("Google Drive integration explicitly skipped with --skip_drive flag")
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
        
        # Check for features directory and handle appropriately
        if args.features_dir and os.path.exists(args.features_dir):
            logger.info(f"Loading pre-extracted features from: {args.features_dir}")
            try:
                # Import required functions
                from datasets import load_from_disk, concatenate_datasets
                
                # Check for combined features directory
                combined_features_path = os.path.join(args.features_dir, "combined_features")
                if os.path.exists(combined_features_path):
                    logger.info(f"Loading combined features from {combined_features_path}")
                    train_dataset = load_from_disk(combined_features_path)
                    
                    # Define seed from config or use default
                    seed = 42  # Default seed
                    try:
                        with open(args.config, 'r') as f:
                            config_data = json.load(f)
                            # Try to get seed from training section
                            seed = config_data.get('training', {}).get('seed', 42)
                    except Exception as e:
                        logger.warning(f"Could not load seed from config, using default: {e}")
                    
                    # Create validation split if not already split
                    if "validation" not in train_dataset.keys():
                        logger.info("Creating train/validation split from combined features")
                        datasets = train_dataset.train_test_split(
                            test_size=0.05, 
                            seed=seed
                        )
                        train_dataset = datasets["train"]
                        eval_dataset = datasets["test"]
                    else:
                        train_dataset = train_dataset["train"]
                        eval_dataset = train_dataset["validation"]
                    
                    logger.info(f"Loaded {len(train_dataset)} training examples and {len(eval_dataset)} validation examples")
                else:
                    # Look for feature directories
                    logger.info("Loading individual feature datasets")
                    feature_dirs = [d for d in os.listdir(args.features_dir) 
                                  if os.path.isdir(os.path.join(args.features_dir, d)) and d.endswith('_features')]
                    
                    if not feature_dirs:
                        logger.warning(f"No feature directories found in {args.features_dir}")
                        raise ValueError(f"No feature directories found in {args.features_dir}")
                    
                    logger.info(f"Found {len(feature_dirs)} feature directories")
                    datasets_list = []
                    
                    # Load each feature dataset
                    for feature_dir in feature_dirs:
                        feature_path = os.path.join(args.features_dir, feature_dir)
                        logger.info(f"Loading features from {feature_path}")
                        dataset = load_from_disk(feature_path)
                        datasets_list.append(dataset)
                    
                    logger.info(f"Combining {len(datasets_list)} feature datasets")
                    train_dataset = concatenate_datasets(datasets_list)
                    
                    # Create validation split
                    logger.info("Creating train/validation split")
                    # Define seed from config or use default
                    seed = 42  # Default seed
                    try:
                        with open(args.config, 'r') as f:
                            config_data = json.load(f)
                            # Try to get seed from training section
                            seed = config_data.get('training', {}).get('seed', 42)
                    except Exception as e:
                        logger.warning(f"Could not load seed from config, using default: {e}")
                    
                    datasets = train_dataset.train_test_split(
                        test_size=0.05, 
                        seed=seed
                    )
                    train_dataset = datasets["train"]
                    eval_dataset = datasets["test"]
                    
                    logger.info(f"Combined dataset contains {len(train_dataset)} training examples and {len(eval_dataset)} validation examples")
            except Exception as e:
                logger.error(f"Error loading features: {str(e)}")
                logger.warning("Falling back to standard dataset loading")
                # Continue with standard dataset loading
        
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