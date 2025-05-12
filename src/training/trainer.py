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
from datasets import Dataset, DatasetDict, concatenate_datasets
import numpy as np
from datetime import datetime
import pkg_resources

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
    from src.utils.drive_utils import save_model_to_drive, get_drive_path, is_drive_mounted
except (ImportError, ModuleNotFoundError):
    try:
        from utils.drive_utils import save_model_to_drive, get_drive_path, is_drive_mounted
    except (ImportError, ModuleNotFoundError):
        print("Warning: drive_utils not found. Creating fallback functions.")
        
        def save_model_to_drive(*args, **kwargs):
            """Fallback function when drive_utils is not available"""
            print("Google Drive integration not available - model not saved to drive.")
            return False
            
        def get_drive_path(local_path, drive_base_path, fallback_path=None):
            """Fallback for getting Google Drive path"""
            return fallback_path or local_path
            
        def is_drive_mounted():
            """Fallback for checking if drive is mounted"""
            return False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
            "eval_steps": ["evaluation_steps"],
            "save_strategy": ["checkpoint_strategy"],
            "lr_scheduler_type": ["scheduler_type"],
            "gradient_accumulation_steps": ["accumulate_grad_batches"],
            "group_by_length": ["length_column_name"],
            "ddp_find_unused_parameters": ["find_unused_parameters"],
            "optim": ["optimizer"],
        }
        
        # Transformers 4.0 specific mappings
        if TRANSFORMERS_MAJOR == 4 and TRANSFORMERS_MINOR < 6:
            # In very old versions (pre-4.6), use these mappings
            strategy_map = {
                "steps": "eval_steps",
                "epoch": "epoch",
                "no": "no"
            }
        
        # Set output_dir directly
        if "output_dir" in param_dict:
            compatible_params["output_dir"] = param_dict["output_dir"]
        
        # Log what parameters are available in the TrainingArguments class
        available_attrs = dir(args_obj)
        logger.debug(f"Available TrainingArguments parameters: {[attr for attr in available_attrs if not attr.startswith('_')]}")
        
        # Process each parameter
        for key, value in param_dict.items():
            if key == "output_dir":
                continue  # already handled
                
            # Direct parameter match
            if hasattr(args_obj, key):
                compatible_params[key] = value
                continue
            
            # Try known renames
            found_rename = False
            if key in renames:
                for alt_name in renames[key]:
                    if hasattr(args_obj, alt_name):
                        compatible_params[alt_name] = value
                        logger.info(f"Mapped parameter '{key}' to '{alt_name}'")
                        found_rename = True
                        break
            
            # Handle special case for evaluation_strategy
            if key == "evaluation_strategy" and not found_rename and TRANSFORMERS_MAJOR == 4 and TRANSFORMERS_MINOR < 6:
                # In older versions, this was split into multiple flags
                if value == "steps" and hasattr(args_obj, "evaluate_during_training"):
                    compatible_params["evaluate_during_training"] = True
                    if "eval_steps" in param_dict:
                        if hasattr(args_obj, "eval_steps"):
                            compatible_params["eval_steps"] = param_dict["eval_steps"]
                    found_rename = True
            
            # Log if we couldn't find a compatible parameter
            if not found_rename and key not in ["evaluation_strategy", "eval_steps", "save_strategy"]:
                # Don't warn about known problematic parameters that we'll handle separately
                logger.warning(f"Parameter '{key}' not found in training arguments and no suitable rename found")
        
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
    
    def train(self, data_dir: str):
        """
        Train the model using the datasets in the specified directory.
        
        Args:
            data_dir: Directory containing processed datasets
        """
        # Update data directory with Google Drive path if needed
        if self.use_drive:
            data_dir = get_drive_path(data_dir, os.path.join(self.drive_base_dir, "data/processed"), data_dir)
        
        # Validate the data directory exists
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory {data_dir} does not exist")
            
        # Load datasets with error handling
        try:
            logger.info(f"Loading and preparing datasets from {data_dir}")
            datasets = self._load_and_prepare_datasets(data_dir)
            
            # Validate datasets
            if not datasets or not all(key in datasets for key in ["train", "validation", "test"]):
                raise ValueError("Failed to properly prepare train/validation/test datasets")
                
            # Validate that datasets have content
            for split_name, dataset in datasets.items():
                if dataset is None:
                    raise ValueError(f"{split_name} dataset is None")
                    
                # Try to check if datasets are empty (when possible)
                try:
                    if hasattr(dataset, '__len__') and len(dataset) == 0:
                        logger.warning(f"{split_name} dataset is empty - training may fail")
                except Exception:
                    logger.warning(f"Could not check length of {split_name} dataset")
        except Exception as e:
            logger.error(f"Error preparing datasets: {str(e)}")
            raise
        
        # Load model
        logger.info("Loading and preparing model")
        try:
            model = self._load_model()
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
        
        # Create training arguments
        training_args_dict = {k: v for k, v in self.training_config.items() if k not in ["seed"]}
        
        # Ensure shuffling is enabled during training
        if "shuffle_dataset" not in training_args_dict:
            training_args_dict["shuffle_dataset"] = True
        
        # Set CUDA optimization flags
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
        
        # Handle parameter compatibility for different versions of transformers
        # Map deprecated or renamed parameters
        param_mapping = {
            "evaluation_strategy": "eval_strategy",
            "eval_steps": "eval_steps",
            "save_strategy": "save_strategy",
        }
        
        # Check transformers version and adjust parameters accordingly
        for old_param, new_param in param_mapping.items():
            if old_param in training_args_dict:
                # For older versions, use the new parameter name
                if hasattr(TrainingArguments, new_param) and not hasattr(TrainingArguments, old_param):
                    training_args_dict[new_param] = training_args_dict.pop(old_param)
                # For newer versions, keep the old parameter name if it exists and the new one doesn't
                elif hasattr(TrainingArguments, old_param) and not hasattr(TrainingArguments, new_param):
                    pass
                # If both exist or neither exist, use the version-appropriate one
                else:
                    try:
                        # Attempt to get the correct parameter by inspecting signatures
                        import inspect
                        params = inspect.signature(TrainingArguments.__init__).parameters
                        if new_param in params and old_param not in params:
                            training_args_dict[new_param] = training_args_dict.pop(old_param)
                        elif old_param in params and new_param not in params:
                            pass
                        else:
                            # Default to using the original param
                            pass
                    except (ImportError, AttributeError):
                        pass
        
        # Remove any parameters that don't exist in the current TrainingArguments
        valid_params = set()
        try:
            import inspect
            valid_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
        except (ImportError, AttributeError):
            pass
        
        if valid_params:
            # Filter out parameters that don't exist in the current version
            training_args_dict = {k: v for k, v in training_args_dict.items() if k in valid_params or k == 'output_dir'}
        
        # Configure DeepSpeed if enabled
        if self.training_config.get("use_deepspeed", False):
            # Basic DeepSpeed config optimized for single-GPU LoRA training
            ds_config = {
                "fp16": {
                    "enabled": "fp16" in self.training_config and self.training_config["fp16"],
                },
                "bf16": {
                    "enabled": "bf16" in self.training_config and self.training_config["bf16"],
                },
                "zero_optimization": {
                    "stage": 2,
                    "offload_optimizer": {
                        "device": "cpu",
                        "pin_memory": True
                    },
                    "contiguous_gradients": True,
                    "overlap_comm": True
                },
                "optimizer": {
                    "type": "Adam",
                    "params": {
                        "lr": self.training_config.get("learning_rate", 2e-4),
                        "weight_decay": self.training_config.get("weight_decay", 0.01)
                    }
                },
                "scheduler": {
                    "type": "WarmupDecayLR",
                    "params": {
                        "warmup_min_lr": 0,
                        "warmup_max_lr": self.training_config.get("learning_rate", 2e-4),
                        "warmup_num_steps": int(self.training_config.get("warmup_ratio", 0.03) * 
                                              self.training_config.get("num_train_epochs", 3) * 
                                              (len(datasets["train"]) if hasattr(datasets["train"], '__len__') else 1000) / 
                                              self.training_config.get("per_device_train_batch_size", 2)),
                        "total_num_steps": int(self.training_config.get("num_train_epochs", 3) * 
                                            (len(datasets["train"]) if hasattr(datasets["train"], '__len__') else 1000) / 
                                            self.training_config.get("per_device_train_batch_size", 2))
                    }
                },
                "gradient_accumulation_steps": self.training_config.get("gradient_accumulation_steps", 4),
                "gradient_clipping": self.training_config.get("max_grad_norm", 1.0),
                "train_batch_size": self.training_config.get("per_device_train_batch_size", 2),
                "train_micro_batch_size_per_gpu": self.training_config.get("per_device_train_batch_size", 2),
            }
            
            # Add DeepSpeed config to training args
            ds_config_path = os.path.join(os.path.dirname(self.training_config["output_dir"]), "ds_config.json")
            os.makedirs(os.path.dirname(ds_config_path), exist_ok=True)
            with open(ds_config_path, "w") as f:
                json.dump(ds_config, f, indent=4)
            
            if "deepspeed" in valid_params:
                training_args_dict["deepspeed"] = ds_config_path
        
        # Create training arguments
        logger.info("Creating training arguments")
        try:
            # First create a minimal version with just output_dir to avoid errors
            minimal_args = {"output_dir": self.training_config["output_dir"]}
            training_args = TrainingArguments(**minimal_args)
            
            # Filter the parameters for compatibility using our helper
            filtered_args = self._check_parameter_compatibility(training_args, training_args_dict)
            
            # Then set the rest of the arguments dynamically
            for key, value in filtered_args.items():
                try:
                    setattr(training_args, key, value)
                except Exception as e:
                    logger.warning(f"Failed to set parameter {key}: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error creating training arguments: {str(e)}")
            # Try with only essential parameters as fallback
            try:
                logger.info("Trying fallback with essential parameters only")
                essential_args = {
                    "output_dir": self.training_config["output_dir"],
                    "per_device_train_batch_size": self.training_config.get("per_device_train_batch_size", 2),
                    "num_train_epochs": self.training_config.get("num_train_epochs", 3)
                }
                training_args = TrainingArguments(**essential_args)
            except Exception as e2:
                logger.error(f"Fallback also failed: {str(e2)}")
                raise
        
        # Create data collator
        logger.info("Creating data collator")
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=False
        )
        
        # Create trainer
        logger.info("Creating trainer")
        try:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=datasets["train"],
                eval_dataset=datasets["validation"],
                tokenizer=self.tokenizer,
                data_collator=data_collator
            )
        except Exception as e:
            logger.error(f"Error creating trainer: {str(e)}")
            raise
        
        # Train model
        logger.info("Starting training")
        try:
            trainer.train()
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
        
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
            metrics = trainer.evaluate(datasets["test"])
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