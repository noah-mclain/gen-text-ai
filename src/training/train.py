#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import time  # Added for time tracking
from pathlib import Path

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
    try:
        from src.training.trainer import DeepseekFineTuner
        
        # Also try to import dataset mapping helper
        try:
            from src.training.dataset_mapping import get_dataset_paths, find_matching_directories
            logger.info("Successfully imported dataset mapping module")
        except (ImportError, ModuleNotFoundError):
            logger.warning("Could not import dataset mapping module, datasets may not be found correctly")
            # Define simple fallback function
            def get_dataset_paths(dataset_names):
                return {}
            def find_matching_directories(data_dir, pattern):
                return []
    except (ImportError, ModuleNotFoundError):
        # Try direct import from current directory
        from trainer import DeepseekFineTuner
        
        # Also try to import dataset mapping helper from current directory
        try:
            from dataset_mapping import get_dataset_paths, find_matching_directories
            logger.info("Successfully imported dataset mapping module")
        except (ImportError, ModuleNotFoundError):
            logger.warning("Could not import dataset mapping module, datasets may not be found correctly")
            # Define simple fallback function
            def get_dataset_paths(dataset_names):
                return {}
            def find_matching_directories(data_dir, pattern):
                return []
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
        from scripts.google_drive.google_drive_manager import (
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
    parser.add_argument("--disable_wandb", action="store_true", default=False,
                        help="Disable Weights & Biases logging (disabled by default)")
    parser.add_argument("--dataloader_workers", type=int, default=1,
                        help="Number of workers for DataLoader")
    parser.add_argument('--features_dir', type=str, default=None,
                    help='Directory containing extracted features for training')
    
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
        
        # Load dataset - either from processed features or from raw processed data
        if args.features_dir and os.path.exists(args.features_dir):
            logger.info(f"Loading pre-extracted features from: {args.features_dir}")
            try:
                # Check for combined features directory
                combined_features_path = os.path.join(args.features_dir, "combined_features")
                if os.path.exists(combined_features_path):
                    logger.info(f"Loading combined features from {combined_features_path}")
                    train_dataset = load_from_disk(combined_features_path)
                    
                    # Create validation split if not already split
                    if "validation" not in train_dataset.keys():
                        logger.info("Creating train/validation split from combined features")
                        datasets = train_dataset.train_test_split(
                            test_size=0.05, 
                            seed=training_args.seed
                        )
                        train_dataset = datasets["train"]
                        eval_dataset = datasets["test"]
                    else:
                        train_dataset = train_dataset["train"]
                        eval_dataset = train_dataset["validation"]
                    
                    logger.info(f"Loaded {len(train_dataset)} training examples and {len(eval_dataset)} validation examples")
                else:
                    # Load individual feature datasets and combine them
                    logger.info("Loading individual feature datasets")
                    feature_dirs = [d for d in os.listdir(args.features_dir) 
                                  if os.path.isdir(os.path.join(args.features_dir, d)) and d.endswith('_features')]
                    
                    if not feature_dirs:
                        logger.warning(f"No feature directories found in {args.features_dir}")
                        raise ValueError(f"No feature directories found in {args.features_dir}")
                    
                    logger.info(f"Found {len(feature_dirs)} feature directories")
                    datasets_list = []
                    
                    for feature_dir in feature_dirs:
                        feature_path = os.path.join(args.features_dir, feature_dir)
                        logger.info(f"Loading features from {feature_path}")
                        dataset = load_from_disk(feature_path)
                        datasets_list.append(dataset)
                    
                    logger.info(f"Combining {len(datasets_list)} feature datasets")
                    train_dataset = concatenate_datasets(datasets_list)
                    
                    # Create validation split
                    logger.info("Creating train/validation split")
                    datasets = train_dataset.train_test_split(
                        test_size=0.05, 
                        seed=training_args.seed
                    )
                    train_dataset = datasets["train"]
                    eval_dataset = datasets["test"]
                    
                    logger.info(f"Combined dataset contains {len(train_dataset)} training examples and {len(eval_dataset)} validation examples")
            except Exception as e:
                logger.error(f"Error loading features: {str(e)}")
                logger.warning("Falling back to standard dataset loading")
                # Continue with standard dataset loading
        else:
            # Standard dataset loading code (keep the existing implementation)
            logger.info(f"Loading datasets from {args.data_dir}")
            
            # First, try to load datasets from config weights if available
            try:
                with open(args.config, 'r') as f:
                    config = json.load(f)
                
                dataset_weights = config.get('dataset_weights', {})
                dataset_paths = config.get('dataset_paths', {})
                
                if dataset_weights:
                    logger.info(f"Found {len(dataset_weights)} datasets in config")
                    datasets_to_load = []
                    
                    # First try to load from dataset_paths if available
                    if dataset_paths:
                        for dataset_name, path in dataset_paths.items():
                            if dataset_name in dataset_weights and os.path.exists(path):
                                datasets_to_load.append((dataset_name, path, dataset_weights[dataset_name]))
                                logger.info(f"Using path from config for {dataset_name}: {path}")
                    
                    # Then look for datasets in data_dir that match names in dataset_weights
                    for dataset_name in dataset_weights.keys():
                        if dataset_name not in [d[0] for d in datasets_to_load]:
                            # Try different common naming patterns
                            potential_paths = [
                                os.path.join(args.data_dir, f"{dataset_name}_processed"),
                                os.path.join(args.data_dir, f"{dataset_name}_processed_interim_final"),
                                os.path.join(args.data_dir, dataset_name)
                            ]
                            
                            # Also check language-specific variants for code datasets
                            if "codesearchnet" in dataset_name:
                                for lang in ["python", "java", "javascript", "php", "ruby", "go"]:
                                    potential_paths.append(os.path.join(args.data_dir, f"codesearchnet_{lang}_processed"))
                                    potential_paths.append(os.path.join(args.data_dir, f"codesearchnet_all_{lang}_processed"))
                            
                            # Use the first path that exists
                            for path in potential_paths:
                                if os.path.exists(path):
                                    datasets_to_load.append((dataset_name, path, dataset_weights[dataset_name]))
                                    logger.info(f"Found dataset directory for {dataset_name}: {path}")
                                    break
                    
                    if not datasets_to_load:
                        logger.warning("No matching dataset directories found for configured datasets")
                else:
                    logger.warning("No dataset_weights found in config, will search for datasets in data_dir")
            except Exception as e:
                logger.warning(f"Error loading dataset configuration: {e}")
                logger.warning("Will search for datasets in data_dir")
            
            # If we couldn't find datasets from config, search the data directory
            if 'datasets_to_load' not in locals() or not datasets_to_load:
                logger.info(f"Searching for dataset directories in {args.data_dir}")
                datasets_to_load = []
                
                if os.path.exists(args.data_dir):
                    # Look for directories with _processed suffix
                    processed_dirs = [d for d in os.listdir(args.data_dir) 
                                     if os.path.isdir(os.path.join(args.data_dir, d)) 
                                     and (d.endswith('_processed') or '_processed_' in d)]
                    
                    for dir_name in processed_dirs:
                        path = os.path.join(args.data_dir, dir_name)
                        # Get dataset name by removing _processed suffix
                        dataset_name = dir_name.split('_processed')[0]
                        # Add with default weight 1.0
                        datasets_to_load.append((dataset_name, path, 1.0))
                        logger.info(f"Found dataset directory: {path}")
            
            # Load the datasets
            loaded_datasets = []
            for dataset_name, path, weight in datasets_to_load:
                try:
                    logger.info(f"Loading dataset {dataset_name} from {path} (weight: {weight})")
                    
                    # Try to load the dataset
                    from datasets import load_from_disk
                    dataset = load_from_disk(path)
                    
                    # Check if this is a valid dataset with the expected format
                    if dataset is None or len(dataset) == 0:
                        logger.warning(f"Dataset {dataset_name} is empty or invalid, skipping")
                        continue
                    
                    # Make sure we have the expected 'text' column
                    if 'text' not in dataset.column_names and 'processed_text' not in dataset.column_names:
                        logger.warning(f"Dataset {dataset_name} is missing 'text' or 'processed_text' column, skipping")
                        continue
                    
                    # Rename 'processed_text' to 'text' if needed
                    if 'processed_text' in dataset.column_names and 'text' not in dataset.column_names:
                        dataset = dataset.rename_column('processed_text', 'text')
                    
                    # Apply tokenization if needed
                    # Check if the dataset needs to be tokenized (doesn't have input_ids and attention_mask)
                    needs_tokenization = ('input_ids' not in dataset.column_names or 
                                         'attention_mask' not in dataset.column_names)
                    
                    if needs_tokenization:
                        logger.info(f"Tokenizing dataset {dataset_name}")
                        
                        # Load tokenizer from config if available
                        try:
                            # Get model name from config
                            with open(args.config, 'r') as f:
                                config = json.load(f)
                            
                            model_name = (config.get('training', {}).get('model_name_or_path') or 
                                         config.get('model', {}).get('base_model'))
                            
                            if model_name:
                                logger.info(f"Loading tokenizer for {model_name}")
                                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                                
                                # Set padding token if not already set
                                if tokenizer.pad_token is None:
                                    if tokenizer.eos_token is not None:
                                        tokenizer.pad_token = tokenizer.eos_token
                                    else:
                                        logger.warning("No padding token available, using a default")
                                        tokenizer.pad_token = tokenizer.eos_token = "</s>"
                                
                                # Tokenize the dataset
                                def tokenize_function(examples):
                                    return tokenizer(
                                        examples['text'],
                                        padding='max_length',
                                        truncation=True,
                                        max_length=2048  # Default value, can be adjusted
                                    )
                                
                                # Apply tokenization
                                dataset = dataset.map(
                                    tokenize_function,
                                    batched=True,
                                    remove_columns=dataset.column_names
                                )
                                
                                logger.info(f"Tokenized dataset {dataset_name}")
                            else:
                                logger.warning("Model name not found in config, skipping tokenization")
                                needs_tokenization = False
                        except Exception as e:
                            logger.warning(f"Error loading tokenizer or tokenizing dataset: {e}")
                            needs_tokenization = False
                    
                    # Set the weight attribute for combining datasets with different weights
                    dataset = dataset.with_format("torch")
                    
                    # Add to loaded datasets
                    loaded_datasets.append({
                        'name': dataset_name,
                        'dataset': dataset,
                        'weight': weight
                    })
                    
                    logger.info(f"Successfully loaded dataset {dataset_name} with {len(dataset)} examples")
                
                except Exception as e:
                    logger.error(f"Error loading dataset {dataset_name}: {e}")
            
            # Ensure we have at least one valid dataset
            if not loaded_datasets:
                raise ValueError(f"No valid datasets found in {args.data_dir}")
            
            # Combine the datasets if multiple
            if len(loaded_datasets) > 1:
                logger.info(f"Combining {len(loaded_datasets)} datasets")
                
                # Perform weighted sampling if needed
                if any(d['weight'] != 1.0 for d in loaded_datasets):
                    # Use dataset concatenation with sampling weights
                    from datasets import concatenate_datasets
                    
                    # Extract datasets and prepare sampling weights
                    datasets_list = [d['dataset'] for d in loaded_datasets]
                    weights = [d['weight'] for d in loaded_datasets]
                    
                    # Normalize weights to sum to 1.0
                    total_weight = sum(weights)
                    normalized_weights = [w / total_weight for w in weights]
                    
                    # Calculate number of samples to take from each dataset
                    total_samples = sum(len(d) for d in datasets_list)
                    samples_per_dataset = [int(w * total_samples) for w in normalized_weights]
                    
                    # Ensure at least one sample from each dataset
                    samples_per_dataset = [max(1, min(s, len(datasets_list[i]))) for i, s in enumerate(samples_per_dataset)]
                    
                    # Take samples from each dataset
                    sampled_datasets = []
                    for i, dataset in enumerate(datasets_list):
                        if samples_per_dataset[i] < len(dataset):
                            # Take a random subset
                            indices = torch.randperm(len(dataset))[:samples_per_dataset[i]].tolist()
                            sampled_datasets.append(dataset.select(indices))
                        else:
                            # Take the whole dataset
                            sampled_datasets.append(dataset)
                    
                    # Concatenate all sampled datasets
                    combined_dataset = concatenate_datasets(sampled_datasets)
                    logger.info(f"Created weighted combined dataset with {len(combined_dataset)} examples")
                else:
                    # Simple concatenation
                    from datasets import concatenate_datasets
                    combined_dataset = concatenate_datasets([d['dataset'] for d in loaded_datasets])
                    logger.info(f"Created combined dataset with {len(combined_dataset)} examples")
                
                # Shuffle the combined dataset
                combined_dataset = combined_dataset.shuffle(seed=42)
                
                # Split into train/validation
                splits = combined_dataset.train_test_split(test_size=0.05, seed=42)
                train_dataset = splits['train']
                eval_dataset = splits['test']
                
                logger.info(f"Final training dataset has {len(train_dataset)} examples")
                logger.info(f"Final validation dataset has {len(eval_dataset)} examples")
            else:
                # Only one dataset, just use it directly
                dataset = loaded_datasets[0]['dataset']
                
                # Split into train/validation if needed
                if hasattr(dataset, 'train_test_split'):
                    splits = dataset.train_test_split(test_size=0.05, seed=42)
                    train_dataset = splits['train']
                    eval_dataset = splits['test']
                    
                    logger.info(f"Split dataset into {len(train_dataset)} training and {len(eval_dataset)} validation examples")
                else:
                    # Use the whole dataset for training
                    train_dataset = dataset
                    # Create a small validation dataset
                    eval_size = min(100, max(1, int(len(dataset) * 0.05)))
                    
                    # Take random samples for validation
                    indices = torch.randperm(len(dataset))
                    train_indices = indices[eval_size:].tolist()
                    eval_indices = indices[:eval_size].tolist()
                    
                    train_dataset = dataset.select(train_indices)
                    eval_dataset = dataset.select(eval_indices)
                    
                    logger.info(f"Created train/eval split with {len(train_dataset)} training and {len(eval_dataset)} validation examples")
        
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