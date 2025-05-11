# Fix import order to ensure Unsloth optimizations are applied correctly
try:
    from unsloth import FastLanguageModel
except ImportError:
    print("Unsloth not installed. Install with: pip install unsloth")

import os
import json
import logging
import torch
import sys
from typing import Dict, List, Union, Optional, Any, Tuple
from datasets import Dataset, DatasetDict, concatenate_datasets
import numpy as np
from datetime import datetime

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
    BitsAndBytesConfig
)
import peft
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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
    from src.utils.drive_utils import save_model_to_drive
except (ImportError, ModuleNotFoundError):
    try:
        from utils.drive_utils import save_model_to_drive
    except (ImportError, ModuleNotFoundError):
        print("Warning: drive_utils not found. Creating fallback functions.")
        
        def save_model_to_drive(*args, **kwargs):
            """Fallback function when drive_utils is not available"""
            print("Google Drive integration not available - model not saved to drive.")
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
    
    def _load_and_prepare_datasets(self, data_dir: str) -> DatasetDict:
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
        splits = create_train_val_test_split(
            combined_dataset,
            train_size=self.dataset_config.get("train_size", 0.9),
            val_size=self.dataset_config.get("val_size", 0.05),
            test_size=self.dataset_config.get("test_size", 0.05),
            seed=self.training_config.get("seed", 42),
            streaming=streaming
        )
        
        logger.info(f"Train size: {len(splits['train'])} examples")
        logger.info(f"Validation size: {len(splits['validation'])} examples")
        logger.info(f"Test size: {len(splits['test'])} examples")
        
        return splits
        
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
        
        # Configure Flash Attention if available
        if torch.cuda.get_device_capability()[0] >= 8:  # For A6000 (Ampere) and newer
            attn_implementation = "flash_attention_2"
            logger.info(f"Using Flash Attention 2 for faster training")
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
        
        # Load datasets
        datasets = self._load_and_prepare_datasets(data_dir)
        
        # Load model
        model = self._load_model()
        
        # Create training arguments
        training_args_dict = {k: v for k, v in self.training_config.items() if k not in ["seed"]}
        
        # Ensure shuffling is enabled during training
        if "shuffle_dataset" not in training_args_dict:
            training_args_dict["shuffle_dataset"] = True
        
        # Set CUDA optimization flags
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
        
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
                                              len(datasets["train"]) / 
                                              self.training_config.get("per_device_train_batch_size", 2)),
                        "total_num_steps": int(self.training_config.get("num_train_epochs", 3) * 
                                            len(datasets["train"]) / 
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
            
            training_args_dict["deepspeed"] = ds_config_path
        
        # Create training arguments
        training_args = TrainingArguments(**training_args_dict)
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        
        # Train model
        logger.info("Starting training")
        trainer.train()
        
        # Save model
        logger.info(f"Saving model to {training_args.output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)
        
        # Evaluate model on test set
        logger.info("Evaluating model on test set")
        metrics = trainer.evaluate(datasets["test"])
        logger.info(f"Test metrics: {metrics}")
        
        # Save metrics
        with open(os.path.join(training_args.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f)
        
        # Push to HuggingFace Hub if enabled
        if self.training_config.get("push_to_hub", False):
            logger.info(f"Pushing model to HuggingFace Hub as {self.training_config.get('hub_model_id')}")
            trainer.push_to_hub()
        
        return metrics 