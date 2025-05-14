# Fix import order to ensure Unsloth optimizations are applied correctly
# Check if the variable has been set in the main script
if 'UNSLOTH_AVAILABLE' not in globals():
    try:
        from unsloth import FastLanguageModel
        UNSLOTH_AVAILABLE = True
    except (ImportError, NotImplementedError):
        print("Unsloth not installed or not compatible. Using standard model loading.")
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

# Add project root to path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import other libraries after unsloth
import transformers
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
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
    from src.utils.google_drive_manager import sync_to_drive as save_model_to_drive, get_drive_path, drive_manager
    
    def is_drive_mounted():
        """Check if Google Drive is authenticated and available"""
        return drive_manager.authenticated if drive_manager else False
        
except (ImportError, ModuleNotFoundError):
    try:
        from scripts.google_drive_manager import drive_manager
        
        def save_model_to_drive(model_path, remote_dir=None):
            """Save model to Google Drive"""
            try:
                if drive_manager and drive_manager.authenticated:
                    return drive_manager.upload_folder(model_path, remote_dir or "models")
                return False
            except Exception as e:
                print(f"Error saving model to Drive: {e}")
                return False
                
        def get_drive_path(local_path, drive_base_path, fallback_path=None):
            """Get the equivalent path on Google Drive"""
            if drive_manager and drive_manager.authenticated:
                return drive_base_path + "/" + os.path.basename(local_path)
            return fallback_path or local_path
            
        def is_drive_mounted():
            """Check if Google Drive is authenticated and available"""
            return drive_manager.authenticated if drive_manager else False
    except (ImportError, ModuleNotFoundError):
        print("Warning: Google Drive integration not available. Creating fallback functions.")
        
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

class FlanUL2TextTrainer:
    def __init__(self, config_path: str, use_drive: bool = False, drive_base_dir: Optional[str] = None):
        """
        Initialize the FlanUL2TextTrainer with a configuration file.
        
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
            token=os.environ.get("HF_TOKEN")
        )
        
        # Set PAD token if not already defined
        if self.tokenizer.pad_token is None:
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
    
    def _load_and_prepare_datasets(self, data_dir: str) -> Dict[str, Dataset]:
        """Load and prepare datasets for training."""
        logger.info(f"Loading datasets from {data_dir}")
        
        # Determine if we should use streaming and caching
        streaming = self.dataset_config.get("streaming", False)
        use_cache = self.dataset_config.get("use_cache", True)
        
        # Get max samples configuration
        max_samples = self.dataset_config.get("max_samples", {})
        
        # Load processed datasets
        datasets = load_processed_datasets(
            data_dir, 
            streaming=streaming, 
            use_cache=use_cache,
            max_samples=max_samples
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
        logger.info(f"Combined dataset size: {len(combined_dataset) if hasattr(combined_dataset, '__len__') else 'unknown'} examples")
        
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
        """Load and prepare the model for training."""
        logger.info(f"Loading model {self.model_config['base_model']}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config["base_model"], 
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN")
        )
        
        # Add special tokens if needed
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set quantization configuration
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=self.model_config.get("use_4bit", True),
            use_nested_quantization=self.model_config.get("use_nested_quant", True),
            bnb_4bit_quant_type="nf4"
        )
        
        # Add robust error handling around model loading
        try:
            # Unsloth doesn't support T5/FLAN model which are encoder-decoder
            logger.info("Using standard HuggingFace model loading (Unsloth doesn't support T5/FLAN models)")
            
            # Standard HuggingFace loading
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_config["base_model"],
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                token=os.environ.get("HF_TOKEN")
            )
            
            # Apply LoRA to the model using PEFT
            if self.model_config.get("use_4bit", False):
                model = prepare_model_for_kbit_training(model)
            
            # Configure LoRA
            lora_config = LoraConfig(
                r=self.peft_config.get("r", 32),
                lora_alpha=self.peft_config.get("lora_alpha", 16),
                target_modules=self.peft_config.get("target_modules", ["q", "k", "v", "o", "wi", "wo"]),
                lora_dropout=self.peft_config.get("lora_dropout", 0.05),
                bias=self.peft_config.get("bias", "none"),
                task_type=TaskType.SEQ_TO_SEQ_LM
            )
            
            # Apply LoRA
            model = get_peft_model(model, lora_config)
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
        
        # Enable gradient checkpointing if requested
        if self.model_config.get("use_gradient_checkpointing", True) and not UNSLOTH_AVAILABLE:
            model.gradient_checkpointing_enable()
        
        # Enable xformers attention if available and requested
        if self.model_config.get("xformers_attention", True) and hasattr(model, "config"):
            try:
                import xformers
                # Enable memory efficient attention
                model.config._attn_implementation = "xformers"
                logger.info("Using xformers memory efficient attention")
            except ImportError:
                logger.warning("xformers not installed. Falling back to default attention.")
        
        # Log model parameters
        logger.info("Model initialized with the following configuration:")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}% of {total_params:,} total)")
        
        return model
    
    def _tokenize_datasets(self, datasets: Dict[str, Dataset]) -> Dict[str, Dataset]:
        """Tokenize datasets for training."""
        logger.info("Tokenizing datasets for sequence-to-sequence training")
        
        max_length = self.dataset_config.get("max_length", 1024)
        
        def tokenize_function(examples):
            # Extract instructions and responses if available
            instructions = examples.get("instruction", [None] * len(examples["text"]))
            responses = examples.get("response", [None] * len(examples["text"]))
            
            # If instruction/response format is available, use it
            if instructions[0] is not None and responses[0] is not None:
                inputs = instructions
                targets = responses
            else:
                # Otherwise, split the text around a separator or use the full text as input
                # For T5/FLAN models, we need an input and a target
                texts = examples["text"]
                # Try to split by common separators like "\n\n", "###", etc.
                inputs = []
                targets = []
                
                for text in texts:
                    if "\n\n" in text:
                        parts = text.split("\n\n", 1)
                        inputs.append(parts[0])
                        targets.append(parts[1])
                    elif "###" in text:
                        parts = text.split("###", 1)
                        inputs.append(parts[0])
                        targets.append(parts[1])
                    else:
                        # No clear separator, use the first half as input
                        mid = len(text) // 2
                        inputs.append(text[:mid])
                        targets.append(text[mid:])
            
            # Tokenize inputs and targets
            model_inputs = self.tokenizer(
                inputs,
                max_length=max_length // 2,  # Allow space for target
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Tokenize targets
            labels = self.tokenizer(
                targets,
                max_length=max_length // 2,
                padding="max_length", 
                truncation=True,
                return_tensors="pt"
            )
            
            # Set the labels in the model inputs
            model_inputs["labels"] = labels["input_ids"]
            
            return model_inputs
        
        # Tokenize each split
        tokenized_datasets = {}
        for split, dataset in datasets.items():
            if dataset is not None:
                logger.info(f"Tokenizing {split} split")
                tokenized_datasets[split] = dataset.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=["instruction", "response", "text", "language", "source"]
                    if "instruction" in dataset.column_names else None
                )
                
                # Add dataset format for the Trainer
                if not hasattr(tokenized_datasets[split], "set_format"):
                    logger.warning(f"{split} dataset doesn't support set_format, skipping format setting")
                else:
                    tokenized_datasets[split].set_format("torch")
        
        return tokenized_datasets
    
    def _configure_training_arguments(self) -> TrainingArguments:
        """Configure training arguments for the Trainer."""
        # Ensure output directory exists
        os.makedirs(self.training_config["output_dir"], exist_ok=True)
        
        # Create a unique run name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"flan_ul2_text_{timestamp}"
        
        # Build training arguments
        args = TrainingArguments(
            output_dir=self.training_config["output_dir"],
            overwrite_output_dir=True,
            num_train_epochs=self.training_config.get("num_train_epochs", 3),
            per_device_train_batch_size=self.training_config.get("per_device_train_batch_size", 2),
            per_device_eval_batch_size=self.training_config.get("per_device_eval_batch_size", 2),
            gradient_accumulation_steps=self.training_config.get("gradient_accumulation_steps", 16),
            evaluation_strategy=self.training_config.get("evaluation_strategy", "steps"),
            eval_steps=self.training_config.get("eval_steps", 200),
            save_strategy=self.training_config.get("save_strategy", "steps"),
            save_steps=self.training_config.get("save_steps", 500),
            save_total_limit=self.training_config.get("save_total_limit", 3),
            learning_rate=self.training_config.get("learning_rate", 1e-4),
            weight_decay=self.training_config.get("weight_decay", 0.01),
            warmup_ratio=self.training_config.get("warmup_ratio", 0.03),
            lr_scheduler_type=self.training_config.get("lr_scheduler_type", "cosine"),
            logging_steps=self.training_config.get("logging_steps", 50),
            report_to=self.training_config.get("report_to", ["wandb", "tensorboard"]),
            run_name=run_name,
            seed=self.training_config.get("seed", 42),
            fp16=self.training_config.get("fp16", True),
            bf16=self.training_config.get("bf16", False),
            push_to_hub=self.training_config.get("push_to_hub", False),
            hub_model_id=self.training_config.get("hub_model_id", None),
            dataloader_num_workers=self.training_config.get("dataloader_num_workers", 8),
            group_by_length=self.training_config.get("group_by_length", True),
            optim=self.training_config.get("optim", "adamw_torch"),
            max_grad_norm=self.training_config.get("max_grad_norm", 1.0),
            ddp_find_unused_parameters=self.training_config.get("ddp_find_unused_parameters", False)
        )
        
        return args
    
    def train(self, data_dir: str) -> Dict[str, Any]:
        """
        Train the FLAN-UL2 model with the datasets at data_dir.
        
        Args:
            data_dir: Directory containing processed datasets
            
        Returns:
            Dictionary with training metrics and results
        """
        # Start timing
        start_time = datetime.now()
        logger.info(f"Starting training at {start_time}")
        
        try:
            # Load model
            model = self._load_model()
            if model is None:
                return {
                    "success": False,
                    "error": "Model loading failed",
                    "traceback": "Model returned None - check logs for details"
                }
            
            # Load and prepare datasets
            datasets = self._load_and_prepare_datasets(data_dir)
            
            # Set up data collator for language modeling
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=model
            )
            
            # Tokenize datasets
            tokenized_datasets = self._tokenize_datasets(datasets)
            
            # Set up training arguments
            training_args = self._configure_training_arguments()
            
            # Log total training steps
            if hasattr(tokenized_datasets["train"], "__len__"):
                total_steps = int((len(tokenized_datasets["train"]) / 
                                   (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)) * 
                                   training_args.num_train_epochs)
                logger.info(f"Total training steps: {total_steps}")
            
            # Initialize Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                tokenizer=self.tokenizer,
                data_collator=data_collator
            )
            
            # Enable torch compile if requested
            if self.training_config.get("torch_compile", False) and hasattr(torch, "compile"):
                logger.info("Enabling torch.compile()")
                torch._dynamo.config.suppress_errors = True
                trainer.model = torch.compile(trainer.model)
            
            # Train the model
            logger.info("Starting training")
            train_result = trainer.train()
            
            # Save the final model
            logger.info("Training completed, saving model")
            trainer.save_model(training_args.output_dir)
            
            # Save to Google Drive if requested
            if self.use_drive and self.drive_base_dir:
                logger.info("Saving model to Google Drive")
                save_model_to_drive(
                    training_args.output_dir,
                    os.path.join(self.drive_base_dir, "models"),
                    move=False
                )
            
            # Compute and log metrics
            metrics = train_result.metrics
            metrics["train_samples"] = len(tokenized_datasets["train"]) if hasattr(tokenized_datasets["train"], "__len__") else "unknown"
            
            # Run evaluation on test set if available
            if "test" in tokenized_datasets and tokenized_datasets["test"] is not None:
                logger.info("Evaluating on test set")
                eval_metrics = trainer.evaluate(tokenized_datasets["test"], metric_key_prefix="test")
                metrics.update(eval_metrics)
            
            # Log metrics
            logger.info(f"Training metrics: {metrics}")
            
            # Calculate duration
            end_time = datetime.now()
            duration = end_time - start_time
            metrics["training_duration"] = str(duration)
            logger.info(f"Total training time: {duration}")
            
            # Return metrics
            metrics_path = os.path.join(training_args.output_dir, "training_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            
            return {
                "success": True,
                "metrics": metrics,
                "output_dir": training_args.output_dir
            }
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Return error information
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            } 