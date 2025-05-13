# Fix import order to ensure Unsloth optimizations are applied correctly
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    print("Unsloth not installed. Install with: pip install unsloth")
    UNSLOTH_AVAILABLE = False

import os
import json
import logging
import torch
import sys
from typing import Dict, List, Union, Optional, Any, Tuple
from datasets import Dataset, DatasetDict, concatenate_datasets, IterableDataset
import numpy as np
from datetime import datetime
import pkg_resources

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.data.dataset_utils import load_from_disk

# Get transformers version for compatibility handling
TRANSFORMERS_VERSION = "0.0.0"  # Default version for compatibility
try:
    TRANSFORMERS_VERSION = pkg_resources.get_distribution("transformers").version
    major_version, minor_version = map(int, TRANSFORMERS_VERSION.split('.')[:2])
    TRANSFORMERS_MAJOR = major_version
    TRANSFORMERS_MINOR = minor_version
except (ImportError, pkg_resources.DistributionNotFound, ValueError, IndexError):
    TRANSFORMERS_MAJOR = 4
    TRANSFORMERS_MINOR = 0
    print(f"Warning: Could not determine transformers version. Assuming compatibility with 4.0")

# Add project root to path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import other libraries after unsloth
import transformers
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed
)
from transformers.trainer_callback import TrainerCallback, EarlyStoppingCallback
from huggingface_hub import login
import peft
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# Try multiple import paths for dataset utilities
try:
    # Absolute import
    from src.data.dataset_utils import load_processed_datasets, combine_datasets, create_train_val_test_split
except (ImportError, ModuleNotFoundError):
    try:
        # Relative import from project root
        from data.dataset_utils import load_processed_datasets, combine_datasets, create_train_val_test_split
    except (ImportError, ModuleNotFoundError):
        try:
            # Relative import within package
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from data.dataset_utils import load_processed_datasets, combine_datasets, create_train_val_test_split
        except (ImportError, ModuleNotFoundError):
            print("Warning: dataset_utils not found. Creating fallback functions.")
            
            # Create minimal fallback functions to prevent crashes
            def load_processed_datasets(data_dir, max_samples=None):
                """Fallback function for loading datasets"""
                print(f"Using fallback dataset loader for directory: {data_dir}")
                from datasets import load_from_disk
                datasets = {}
                
                if not os.path.exists(data_dir):
                    print(f"Warning: Data directory {data_dir} does not exist.")
                    return {}
                
                for entry in os.listdir(data_dir):
                    path = os.path.join(data_dir, entry)
                    if os.path.isdir(path) and entry.endswith("_processed"):
                        name = entry.replace("_processed", "")
                        try:
                            dataset = load_from_disk(path)
                            if isinstance(dataset, Dataset) or isinstance(dataset, DatasetDict):
                                datasets[name] = dataset
                                print(f"Loaded dataset {name} from {path} with {len(dataset)} examples")
                        except Exception as e:
                            print(f"Failed to load dataset {name}: {str(e)}")
                
                return datasets
            
            def combine_datasets(datasets, weights=None, seed=42):
                """Fallback function to combine datasets"""
                if not datasets:
                    print("No datasets to combine")
                    return None
                    
                dataset_list = list(datasets.values())
                combined = dataset_list[0]
                for ds in dataset_list[1:]:
                    combined = concatenate_datasets([combined, ds])
                
                return combined
                
            def create_train_val_test_split(dataset, train_size=0.8, val_size=0.1, test_size=0.1, seed=42):
                """Fallback function to split dataset"""
                if not dataset:
                    return None, None, None
                    
                splits = dataset.train_test_split(test_size=val_size + test_size, seed=seed)
                train_dataset = splits['train']
                
                # Further split the test into validation and test
                remaining_size = val_size / (val_size + test_size)
                val_test_splits = splits['test'].train_test_split(test_size=remaining_size, seed=seed)
                
                return train_dataset, val_test_splits['test'], val_test_splits['train']

# Try multiple import paths for Google Drive utilities
try:
    # First try the new path structure
    from src.utils.google_drive_manager import drive_manager, sync_to_drive, sync_from_drive
    
    def save_model_to_drive(model_path, remote_dir=None):
        """Save model to Google Drive using the new utility functions"""
        try:
            return sync_to_drive(model_path, remote_dir or "models")
        except Exception as e:
            logger.error(f"Error saving model to Drive: {e}")
            return False
            
    def get_drive_path(local_path, drive_base_path, fallback_path=None):
        """Get the equivalent path on Google Drive"""
        return drive_base_path + "/" + os.path.basename(local_path) if drive_manager.authenticated else (fallback_path or local_path)
        
    def is_drive_mounted():
        """Check if Google Drive is authenticated and available"""
        return drive_manager.authenticated
        
except (ImportError, ModuleNotFoundError):
    # Try the legacy imports
    try:
        from scripts.google_drive_manager import drive_manager, sync_to_drive, sync_from_drive
        
        def save_model_to_drive(model_path, remote_dir=None):
            """Save model to Google Drive using the new utility functions"""
            try:
                return sync_to_drive(model_path, remote_dir or "models")
            except Exception as e:
                logger.error(f"Error saving model to Drive: {e}")
                return False
                
        def get_drive_path(local_path, drive_base_path, fallback_path=None):
            """Get the equivalent path on Google Drive"""
            return drive_base_path + "/" + os.path.basename(local_path) if drive_manager.authenticated else (fallback_path or local_path)
            
        def is_drive_mounted():
            """Check if Google Drive is authenticated and available"""
            return drive_manager.authenticated
    except (ImportError, ModuleNotFoundError):
        logger.warning("google_drive_manager not found. Creating fallback functions.")
        
        def save_model_to_drive(*args, **kwargs):
            """Fallback function when drive_utils is not available"""
            logger.warning("Google Drive integration not available - model not saved to drive.")
            return False
            
        def get_drive_path(local_path, drive_base_path, fallback_path=None):
            """Fallback for getting Google Drive path"""
            return fallback_path or local_path
            
        def is_drive_mounted():
            """Fallback for checking if drive is mounted"""
            return False

# Define custom callback classes
class CodeEvalCallback(TrainerCallback):
    """Callback for code evaluation during training"""
    def __init__(self, eval_steps=500):
        self.eval_steps = eval_steps
        
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == 0:
            # Implement code evaluation here
            pass

class TensorLoggingCallback(TrainerCallback):
    """Callback for logging tensor values during training"""
    def on_step_end(self, args, state, control, **kwargs):
        # Implement tensor logging here
        pass

class DeepseekFineTuner:
    def __init__(self, config_path: str, use_drive: bool = False, drive_base_dir: Optional[str] = None):
        """
        Initialize the DeepseekFineTuner with a configuration file.
        
        Args:
            config_path: Path to the training configuration file
            use_drive: Whether to use Google Drive for storage
            drive_base_dir: Base directory on Google Drive (if using Drive)
        """
        self.config = self._load_config(config_path)
        self.model_config = self.config["model"]
        self.peft_config = self.config["peft"]
        self.training_config = self.config["training"]
        self.dataset_config = self.config["dataset"]
        
        # Flag for LoRA adapter
        self.use_lora = self.peft_config.get("use_lora", True)
        
        # Google Drive integration
        self.use_drive = use_drive
        self.drive_base_dir = drive_base_dir
        if use_drive and drive_base_dir and not is_drive_mounted():
            logger.warning("Google Drive is not mounted, but use_drive is set to True. Using local paths instead.")
            self.use_drive = False
        
        # Convert output directory to Google Drive path if needed
        if self.use_drive and "output_dir" in self.training_config:
            self.training_config["output_dir"] = get_drive_path(
                self.training_config["output_dir"],
                os.path.join(self.drive_base_dir, "models")
            )
            
            # Set hub model ID if pushing to hub
            if self.training_config.get("push_to_hub", False):
                if "hub_model_id" not in self.training_config:
                    model_name = os.path.basename(self.training_config["output_dir"])
                    self.training_config["hub_model_id"] = model_name
        
        # Set seed for reproducibility
        set_seed(self.training_config.get("seed", 42))
        
        # HuggingFace login if pushing to hub
        if self.training_config.get("push_to_hub", False):
            self._login_huggingface()
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config["base_model"],
            trust_remote_code=True,
            use_auth_token=os.environ.get("HF_TOKEN")
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def _login_huggingface(self):
        """Login to HuggingFace Hub using token."""
        token = os.environ.get("HF_TOKEN")
        if token:
            try:
                login(token=token)
                logger.info("Successfully logged in to HuggingFace Hub")
            except Exception as e:
                logger.error(f"Error logging in to HuggingFace Hub: {str(e)}")
        else:
            logger.warning("HF_TOKEN environment variable not set. Pushing to Hub may fail.")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from a JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _check_parameter_compatibility(self, args_obj, param_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check parameter compatibility with an object and filter out incompatible parameters.
        
        Args:
            args_obj: Object to check parameter compatibility with
            param_dict: Dictionary of parameters to check
            
        Returns:
            Dictionary with compatible parameters only
        """
        compatible_params = {}
        
        # Define version-specific parameter mappings
        renames = {
            # Parameters that changed names across transformers versions
            "evaluation_strategy": ["eval_strategy"],
            "save_strategy": ["save_steps_strategy"],
            "logging_strategy": ["log_strategy"]
        }
        
        # Handle base_model_prefix differently based on version
        import inspect
        
        # Get the signature of the args_obj constructor
        if hasattr(args_obj, "__init__"):
            signature = inspect.signature(args_obj.__init__)
            valid_params = list(signature.parameters.keys())
            
            # Filter parameters
            for key, value in param_dict.items():
                # Check if the parameter is valid
                if key in valid_params:
                    compatible_params[key] = value
                else:
                    # Try renamed parameters
                    renamed = False
                    if key in renames:
                        for alt_key in renames[key]:
                            if alt_key in valid_params:
                                compatible_params[alt_key] = value
                                renamed = True
                                break
                    
                    if not renamed:
                        logger.debug(f"Parameter '{key}' is not compatible with {args_obj.__class__.__name__}")
        else:
            # If we can't inspect, just pass through all parameters
            compatible_params = param_dict
        
        return compatible_params
    
    def _load_and_prepare_datasets(self, data_dir: str) -> Dict[str, Dataset]:
        """Load and prepare datasets for training."""
        logger.info(f"Loading datasets from {data_dir}")
        
        # Determine if we should use streaming and caching
        streaming = self.dataset_config.get("streaming", False)
        use_cache = self.dataset_config.get("use_cache", True)
        
        # Load processed datasets
        datasets = load_processed_datasets(
            data_dir, 
            streaming=streaming, 
            use_cache=use_cache
        )
        if not datasets:
            raise ValueError(f"No datasets found in {data_dir}")
        
        # Get dataset weights
        weights = self.dataset_config.get("dataset_weights", {})
        
        # Combine datasets
        logger.info("Combining datasets")
        combined_dataset = combine_datasets(
            datasets, 
            weights=weights,
            interleave_prob=self.dataset_config.get("interleave_prob"),
            seed=self.training_config.get("seed", 42)
        )
        logger.info(f"Combined dataset size: {len(combined_dataset)} examples")
        
        # Create train/val/test splits
        logger.info("Creating train/val/test splits")
        train_dataset, val_dataset, test_dataset = create_train_val_test_split(
            combined_dataset,
            train_size=self.dataset_config.get("train_size", 0.9),
            val_size=self.dataset_config.get("val_size", 0.05),
            test_size=self.dataset_config.get("test_size", 0.05),
            seed=self.training_config.get("seed", 42),
            streaming=streaming
        )
        
        # Safely check lengths for logging
        try:
            logger.info(f"Train size: {len(train_dataset) if hasattr(train_dataset, '__len__') else 'unknown'} examples")
            logger.info(f"Validation size: {len(val_dataset) if hasattr(val_dataset, '__len__') else 'unknown'} examples")
            logger.info(f"Test size: {len(test_dataset) if hasattr(test_dataset, '__len__') else 'unknown'} examples")
        except Exception as e:
            logger.warning(f"Could not determine dataset sizes: {str(e)}")
        
        # Return as dictionary for trainer
        return {
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        }
        
    def _load_model(self):
        """Load model with appropriate quantization and optimization."""
        use_unsloth = self.model_config.get("use_unsloth", False) and UNSLOTH_AVAILABLE
        use_4bit = self.model_config.get("use_4bit", True)
        use_8bit = self.model_config.get("use_8bit", False)
        
        logger.info(f"Loading model {self.model_config['base_model']}")
        logger.info(f"Use Unsloth: {use_unsloth}, Use 4-bit: {use_4bit}, Use 8-bit: {use_8bit}")
        
        # Calculate optimal batch size and gradient accumulation for A6000 GPU
        # A6000 has 48GB VRAM, we'll target ~40GB usage to leave room for overhead
        # Deepseek-Coder 6.7b with 4-bit quantization needs ~10GB base memory
        # Each sequence takes approximately (seq_len * 2 * 4) / 4 = 2*seq_len bytes in 4-bit
        max_length = self.dataset_config.get("max_length", 2048)
        memory_per_seq = max_length * 2  # Approximation in bytes for 4-bit quantization
        
        # For A6000 with 48GB (minus 8GB overhead)
        available_memory = 40 * 1024 * 1024 * 1024  # 40GB in bytes
        
        # Calculate max batch size (adjust for model type and precision)
        max_batch_size = min(16, available_memory // (memory_per_seq * 1.5))
        per_device_batch_size = min(
            max_batch_size, 
            self.training_config.get("per_device_train_batch_size", 4)
        )
        
        # Update training config with calculated values
        self.training_config["per_device_train_batch_size"] = per_device_batch_size
        self.training_config["gradient_accumulation_steps"] = max(
            1, 
            32 // per_device_batch_size  # Target an effective batch size of 32
        )
        
        logger.info(f"Using batch size: {per_device_batch_size}, " +
                   f"grad accumulation: {self.training_config['gradient_accumulation_steps']}")
        
        # Configure attention implementation
        if torch.cuda.get_device_capability()[0] >= 8:  # For A6000 (Ampere) and newer
            if self.model_config.get("xformers_attention", True):
                attn_implementation = "xformers"
                logger.info(f"Using xformers attention for faster training")
            else:
                attn_implementation = "eager"
        else:
            attn_implementation = "eager"
        
        if use_unsloth:
            model, _ = FastLanguageModel.from_pretrained(
                model_name=self.model_config["base_model"],
                max_seq_length=self.dataset_config.get("max_length", 2048),
                dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                load_in_4bit=use_4bit,
                load_in_8bit=use_8bit,
                token=os.environ.get("HF_TOKEN"),
                attn_implementation=attn_implementation,
            )
            # Add LoRA adapters
            model = FastLanguageModel.get_peft_model(
                model,
                r=self.peft_config.get("r", 16),
                target_modules=self.peft_config.get("target_modules"),
                lora_alpha=self.peft_config.get("lora_alpha", 16),
                lora_dropout=self.peft_config.get("lora_dropout", 0.05),
                bias=self.peft_config.get("bias", "none"),
                use_gradient_checkpointing=self.model_config.get("use_gradient_checkpointing", True),
                random_state=self.training_config.get("seed", 42),
            )
        else:
            # Standard loading with PEFT
            if use_4bit:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_config["base_model"],
                    quantization_config={
                        "load_in_4bit": True,
                        "bnb_4bit_compute_dtype": getattr(torch, self.model_config.get("bnb_4bit_compute_dtype", "float16")),
                        "bnb_4bit_use_double_quant": self.model_config.get("use_nested_quant", True),
                        "bnb_4bit_quant_type": "nf4",
                    },
                    device_map="auto",
                    trust_remote_code=True,
                    token=os.environ.get("HF_TOKEN"),
                    attn_implementation=attn_implementation,
                )
            elif use_8bit:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_config["base_model"],
                    load_in_8bit=True,
                    device_map="auto",
                    trust_remote_code=True,
                    token=os.environ.get("HF_TOKEN"),
                    attn_implementation=attn_implementation,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_config["base_model"],
                    device_map="auto",
                    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    trust_remote_code=True,
                    token=os.environ.get("HF_TOKEN"),
                    attn_implementation=attn_implementation,
                )
            
            if use_4bit or use_8bit:
                model = prepare_model_for_kbit_training(model)
            
            if self.model_config.get("use_gradient_checkpointing", True):
                model.gradient_checkpointing_enable()
                
            # Add LoRA adapters
            lora_config = LoraConfig(
                r=self.peft_config.get("r", 16),
                lora_alpha=self.peft_config.get("lora_alpha", 16),
                lora_dropout=self.peft_config.get("lora_dropout", 0.05),
                bias=self.peft_config.get("bias", "none"),
                task_type=TaskType.CAUSAL_LM,
                target_modules=self.peft_config.get("target_modules")
            )
            model = get_peft_model(model, lora_config)
        
        model.config.use_cache = False  # Required for gradient checkpointing
        
        # Print some debug information
        logger.info(f"Model loaded with {model.num_parameters() / 1e6:.2f}M parameters")
        logger.info(f"Trainable parameters: {model.num_parameters(True) / 1e6:.2f}M")
        
        return model
    
    def _combine_datasets(self, processed_datasets):
        """
        Combine multiple datasets with their respective weights.
        
        Args:
            processed_datasets: Dictionary of datasets with their weights
            
        Returns:
            Combined dataset
        """
        logger.info("Combining datasets")
        
        if not processed_datasets:
            logger.error("No datasets to combine")
            return None
            
        # If there's only one dataset, return it directly
        if len(processed_datasets) == 1:
            dataset_info = next(iter(processed_datasets.values()))
            return dataset_info["dataset"]
            
        # Get all datasets
        datasets = [info["dataset"] for info in processed_datasets.values()]
        weights = [info["weight"] for info in processed_datasets.values()]
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        logger.info(f"Combining {len(datasets)} datasets with weights: {normalized_weights}")
        
        # For streaming datasets, we need to interleave them
        is_streaming = any(isinstance(ds, IterableDataset) for ds in datasets)
        
        if is_streaming:
            # Import necessary function
            from datasets import interleave_datasets
            
            try:
                combined = interleave_datasets(
                    datasets, 
                    probabilities=normalized_weights,
                    seed=self.training_config.get("seed", 42)
                )
                return combined
            except Exception as e:
                logger.error(f"Error combining streaming datasets: {e}")
                # Fallback to first dataset
                return datasets[0]
        else:
            # For regular datasets, we sample and concatenate
            try:
                # Import concatenate_datasets
                from datasets import concatenate_datasets
                
                # Determine sample sizes
                total_samples = sum(len(ds) for ds in datasets)
                sample_sizes = [int(w * total_samples) for w in normalized_weights]
                
                # Ensure at least one sample from each dataset
                sample_sizes = [max(1, size) for size in sample_sizes]
                
                # Sample from each dataset
                sampled_datasets = []
                for ds, size in zip(datasets, sample_sizes):
                    sampled = ds.shuffle(seed=self.training_config.get("seed", 42)).select(range(min(len(ds), size)))
                    sampled_datasets.append(sampled)
                
                # Concatenate all sampled datasets
                combined = concatenate_datasets(sampled_datasets)
                combined = combined.shuffle(seed=self.training_config.get("seed", 42))
                
                return combined
            except Exception as e:
                logger.error(f"Error combining datasets: {e}")
                # Fallback to first dataset
                return datasets[0]
    
    def _configure_lora(self, model):
        """
        Configure LoRA adapters for the model.
        
        Args:
            model: The base model to add LoRA adapters to
            
        Returns:
            Model with LoRA adapters
        """
        logger.info("Configuring LoRA adapters")
        
        try:
            # Use the LoraConfig from the peft library
            lora_config = LoraConfig(
                r=self.peft_config.get("r", 16),
                lora_alpha=self.peft_config.get("lora_alpha", 16),
                lora_dropout=self.peft_config.get("lora_dropout", 0.05),
                bias=self.peft_config.get("bias", "none"),
                task_type=TaskType.CAUSAL_LM,
                target_modules=self.peft_config.get("target_modules")
            )
            
            # Apply LoRA to the model
            model = get_peft_model(model, lora_config)
            logger.info(f"LoRA adapters configured with rank={lora_config.r}")
            
            return model
        except Exception as e:
            logger.error(f"Error configuring LoRA: {e}")
            logger.info("Continuing without LoRA")
            return model
    
    def train(self, data_dir: str) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            data_dir: Directory containing preprocessed datasets
            
        Returns:
            Dict of training metrics
        """
        logger.info("Loading datasets from %s", data_dir)
        
        # Initialize streaming flag
        self.is_streaming = False
        
        # Check for streaming datasets first (since user doesn't save locally)
        streaming_datasets = {}
        logger.info("Checking for streaming datasets")
        
        for dataset_name, weight in self.dataset_config.get("dataset_weights", {}).items():
            try:
                # Import dataset loader functions
                from src.data.streaming import load_streaming_dataset
                
                # Load streaming dataset
                dataset = load_streaming_dataset(
                    dataset_name,
                    self.tokenizer,
                    self.dataset_config.get("streaming_config", {}),
                    num_workers=torch.cuda.device_count() if torch.cuda.is_available() else 1
                )
                
                # Add to the list of streaming datasets
                streaming_datasets[dataset_name] = {
                    "dataset": dataset,
                    "weight": weight
                }
                
                logger.info(f"Loaded streaming dataset {dataset_name} with weight {weight}")
            except Exception as e:
                logger.info(f"Streaming dataset {dataset_name} not available: {e}")
                
        # If streaming datasets are available, use them
        processed_datasets = {}
        if streaming_datasets:
            processed_datasets = streaming_datasets
            self.is_streaming = True
            logger.info("Using streaming datasets")
            
            # Update training args to specify max_steps for streaming datasets
            if "max_steps" not in self.training_config:
                # Set a default value based on num_train_epochs if available
                epochs = self.training_config.get("num_train_epochs", 1)
                self.training_config["max_steps"] = int(100000 * epochs)  # A reasonable default
                logger.info(f"Setting max_steps to {self.training_config['max_steps']} for streaming datasets")
        else:
            # Only check local files if streaming is not available
            for dataset_name, weight in self.dataset_config.get("dataset_weights", {}).items():
                dataset_processed_path = os.path.join(data_dir, f"{dataset_name}_processed")
                
                if os.path.exists(dataset_processed_path):
                    try:
                        dataset = load_from_disk(dataset_processed_path)
                        
                        # Add to the list of processed datasets
                        processed_datasets[dataset_name] = {
                            "dataset": dataset,
                            "weight": weight
                        }
                        
                        logger.info(f"Loaded dataset {dataset_name} with {len(dataset)} examples and weight {weight}")
                    except Exception as e:
                        logger.info(f"Error loading dataset {dataset_name}: {e}")
            
        if not processed_datasets:
            logger.error("No datasets found for training!")
            return {"error": "No datasets found"}
        
        # Create a combined dataset
        train_dataset = self._combine_datasets(processed_datasets)
        
        if train_dataset is None:
            logger.error("Failed to create combined dataset")
            return {"error": "Failed to create combined dataset"}
        
        # Prepare model for training
        model = self._load_model()
        
        # Configure LoRA if enabled
        if self.use_lora:
            logger.info("Configuring LoRA")
            model = self._configure_lora(model)
        
        logger.info("Creating training arguments")
        
        # Ensure max_steps is set for streaming datasets
        if self.is_streaming and "max_steps" not in self.training_config:
            # Set a reasonable default
            self.training_config["max_steps"] = 100000
            logger.info(f"Set max_steps to {self.training_config['max_steps']} for streaming datasets")
            
        # Filter incompatible training arguments based on the Transformers version
        filtered_training_config = self._check_parameter_compatibility(
            TrainingArguments, 
            {k: v for k, v in self.training_config.items() if k != "output_dir"}
        )
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=self.training_config.get("output_dir", "models/deepseek-coder-finetune"),
            **filtered_training_config
        )
        
        logger.info("Creating data collator")
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        logger.info("Creating trainer")
        try:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=None,  # We'll handle evaluation separately
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                callbacks=[
                    CodeEvalCallback(self.training_config.get("eval_steps", 500))
                ] if self.training_config.get("eval_during_training", False) else None
            )
            
            # Add early stopping callback if configured
            if self.training_config.get("early_stopping_patience"):
                trainer.add_callback(
                    EarlyStoppingCallback(
                        early_stopping_patience=self.training_config.get("early_stopping_patience")
                    )
                )
            
            # Add tensor logging callback if configured
            if self.training_config.get("log_tensors", False):
                trainer.add_callback(TensorLoggingCallback())
            
        except Exception as e:
            logger.error(f"Error creating trainer: {e}")
            raise
        
        # Train the model
        logger.info("Starting training")
        train_result = trainer.train()
        
        # Save model
        logger.info(f"Saving model to {training_args.output_dir}")
        try:
            trainer.save_model()
            self.tokenizer.save_pretrained(training_args.output_dir)
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            # Don't raise here so we can still try to evaluate
        
        # Evaluate model on test set
        logger.info("Evaluating model on test set")
        try:
            metrics = trainer.evaluate(None)
            logger.info(f"Test metrics: {metrics}")
            
            # Save metrics
            with open(os.path.join(training_args.output_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f)
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            metrics = {"error": str(e)}
        
        # Push to HuggingFace Hub if enabled
        if self.training_config.get("push_to_hub", False):
            try:
                logger.info(f"Pushing model to HuggingFace Hub as {self.training_config.get('hub_model_id')}")
                trainer.push_to_hub()
            except Exception as e:
                logger.error(f"Error pushing to hub: {str(e)}")
        
        return metrics 