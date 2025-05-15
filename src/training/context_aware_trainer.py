"""
Context-Aware Text Trainer for FLAN-UT 20B

This module provides a specialized trainer for fine-tuning the FLAN-UT 20B model
with context and intent awareness to enhance conversational capabilities.
"""

import os
import json
import logging
import torch
import sys
from typing import Dict, List, Union, Optional, Any, Tuple
from datasets import Dataset, DatasetDict, concatenate_datasets
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project root to path for absolute imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import transformer libraries
import transformers
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    set_seed,
    AutoConfig,
    T5ForConditionalGeneration
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# Try to import dataset utilities
try:
    from src.data.dataset_utils import load_processed_datasets, combine_datasets, create_train_val_test_split
except (ImportError, ModuleNotFoundError):
    from data.dataset_utils import load_processed_datasets, combine_datasets, create_train_val_test_split

# Try to import the ConversationMemory and analyzers for inference
try:
    from src.data.processors.context_aware_processor import (
        ConversationMemory, 
        IntentAnalyzer, 
        ContextExtractor
    )
except (ImportError, ModuleNotFoundError):
    try:
        from data.processors.context_aware_processor import (
            ConversationMemory, 
            IntentAnalyzer, 
            ContextExtractor
        )
    except (ImportError, ModuleNotFoundError):
        logger.warning("Could not import context-aware components. Some functionality may be limited.")
        ConversationMemory = None
        IntentAnalyzer = None
        ContextExtractor = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContextAwareTrainer:
    """
    Trainer for fine-tuning FLAN-UT 20B with context and intent awareness.
    """
    
    def __init__(self, config_path: str, use_drive: bool = False, drive_base_dir: Optional[str] = None, pretrained_model_path: Optional[str] = None):
        """
        Initialize the Context-Aware Trainer with a configuration file.
        
        Args:
            config_path: Path to the training configuration file
            use_drive: Whether to use Google Drive for storage
            drive_base_dir: Base directory on Google Drive (if using Drive)
            pretrained_model_path: Path to a previously fine-tuned model to use as base
        """
        self.config = self._load_config(config_path)
        self.model_config = self.config["model"]
        self.peft_config = self.config["peft"]
        self.training_config = self.config["training"]
        self.dataset_config = self.config["dataset"]
        
        # Store pretrained model path
        self.pretrained_model_path = pretrained_model_path
        
        # Set the model paths and directories
        self.output_dir = self.training_config.get("output_dir", "models/flan-ut-20b-context-aware")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize context-aware components for inference
        self.conversation_memory = ConversationMemory() if ConversationMemory else None
        self.intent_analyzer = IntentAnalyzer() if IntentAnalyzer else None
        self.context_extractor = ContextExtractor() if ContextExtractor else None
        
        # Initialize tokenizer and model to None (will be loaded later)
        self.tokenizer = None
        self.model = None
        
        # Ensure we have access to Hugging Face Hub
        self._login_huggingface()
    
    def _login_huggingface(self):
        """Login to Hugging Face Hub if token is available."""
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            try:
                from huggingface_hub import login
                login(token=hf_token)
                logger.info("Successfully logged in to Hugging Face Hub")
            except Exception as e:
                logger.warning(f"Failed to login to Hugging Face Hub: {str(e)}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from a JSON file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    def _load_and_prepare_datasets(self, data_dir: str) -> Dict[str, Dataset]:
        """
        Load and prepare datasets for training.
        
        Args:
            data_dir: Directory containing the processed datasets
            
        Returns:
            Dictionary mapping dataset names to datasets
        """
        logger.info(f"Loading datasets from {data_dir}")
        
        # Get dataset weights from config (fallback to default weights if not provided)
        dataset_weights = self.dataset_config.get("dataset_weights", {
            "context_aware_conversation": 1.0,
            "synthetic_persona": 0.8,
            "gpteacher_general": 0.6,
            "writingprompts": 0.4,
            "pile": 0.2
        })
        
        # Get max samples per dataset
        max_samples = self.dataset_config.get("max_samples", {})
        
        # Load all available datasets in the directory
        datasets = load_processed_datasets(data_dir, max_samples)
        
        # Log available datasets with their sizes
        logger.info(f"Loaded {len(datasets)} datasets:")
        for name, dataset in datasets.items():
            logger.info(f"  - {name}: {len(dataset)} examples")
            
            # Check if dataset has context and intent columns
            if name == "context_aware_conversation":
                columns = dataset.column_names
                logger.info(f"    Columns: {', '.join(columns)}")
                if "context" in columns and "intent" in columns:
                    logger.info(f"    Context and intent columns found in {name}")
        
        # Filter to only include datasets listed in the weights
        filtered_datasets = {name: dataset for name, dataset in datasets.items() if name in dataset_weights}
        if len(filtered_datasets) < len(datasets):
            logger.info(f"Filtered from {len(datasets)} to {len(filtered_datasets)} datasets based on weights")
        
        # For datasets not in filtered_datasets but in weights, log a warning
        for name in dataset_weights:
            if name not in filtered_datasets:
                logger.warning(f"Dataset {name} specified in weights but not found in {data_dir}")
        
        # Apply weights to filtered datasets
        weights = {name: dataset_weights.get(name, 1.0) for name in filtered_datasets}
        
        # Combine datasets with appropriate weighting
        combined_dataset = combine_datasets(filtered_datasets, weights)
        
        # Create train/validation/test splits
        train_size = self.dataset_config.get("train_size", 0.9)
        val_size = self.dataset_config.get("val_size", 0.05)
        test_size = self.dataset_config.get("test_size", 0.05)
        
        train_dataset, val_dataset, test_dataset = create_train_val_test_split(
            combined_dataset, train_size, val_size, test_size
        )
        
        # Add a dataset type field to each split
        return {
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        }
    
    def _load_model(self):
        """Load and prepare the FLAN-UT 20B model for fine-tuning."""
        # Extract model configuration
        base_model = self.model_config.get("base_model", "google/flan-t5-xxl") # Default to FLAN-T5-XXL if FLAN-UT-20B not specified
        use_4bit = self.model_config.get("use_4bit", True)
        use_nested_quant = self.model_config.get("use_nested_quant", True)
        bnb_4bit_compute_dtype = self.model_config.get("bnb_4bit_compute_dtype", "bfloat16")
        use_gradient_checkpointing = self.model_config.get("use_gradient_checkpointing", True)
        
        compute_dtype_mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }
        
        compute_dtype = compute_dtype_mapping.get(bnb_4bit_compute_dtype, torch.bfloat16)
        
        # If we have a pretrained model path, use that instead of the base model
        model_path = self.pretrained_model_path if self.pretrained_model_path else base_model
        
        logger.info(f"Loading model: {model_path}")
        logger.info(f"Quantization: 4-bit={use_4bit}, Nested={use_nested_quant}, Compute dtype={bnb_4bit_compute_dtype}")
        
        # First load the tokenizer
        tokenizer_path = self.pretrained_model_path if self.pretrained_model_path else base_model
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            padding_side="right",
            use_fast=True
        )
        
        # Configure the quantization settings (if needed)
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=use_nested_quant,
                bnb_4bit_quant_type="nf4"
            )
        else:
            quantization_config = None
        
        # Load the base model
        device_map = "auto"  # Let the library decide on the optimal device map
        
        # Load model with appropriate settings
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=compute_dtype if not use_4bit else None
        )
        
        # Apply gradient checkpointing if needed
        if use_gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        # Prepare the model for PEFT/LoRA
        if self.peft_config:
            logger.info("Setting up LoRA fine-tuning")
            
            # Extract LoRA configuration
            lora_alpha = self.peft_config.get("lora_alpha", 16)
            lora_dropout = self.peft_config.get("lora_dropout", 0.05)
            r = self.peft_config.get("r", 16)
            bias = self.peft_config.get("bias", "none")
            task_type = self.peft_config.get("task_type", "SEQ_TO_SEQ_LM")
            target_modules = self.peft_config.get("target_modules", ["q", "k", "v", "o", "wi", "wo"])
            
            # Map string task type to TaskType enum
            task_type_mapping = {
                "SEQ_TO_SEQ_LM": TaskType.SEQ_TO_SEQ_LM,
                "CAUSAL_LM": TaskType.CAUSAL_LM,
                "SEQ_CLS": TaskType.SEQ_CLS
            }
            task_type_enum = task_type_mapping.get(task_type, TaskType.SEQ_TO_SEQ_LM)
            
            # Prepare the model for PEFT if using 4-bit quantization
            if use_4bit:
                self.model = prepare_model_for_kbit_training(self.model)
            
            # Create LoRA configuration
            peft_config = LoraConfig(
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias=bias,
                task_type=task_type_enum,
                target_modules=target_modules
            )
            
            # Apply LoRA to the model
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        
        logger.info(f"Model loaded successfully: {model_path}")
        return self.model, self.tokenizer
    
    def _tokenize_datasets(self, datasets: Dict[str, Dataset]) -> Dict[str, Dataset]:
        """
        Tokenize datasets for training.
        
        Args:
            datasets: Dictionary mapping split names to datasets
            
        Returns:
            Dictionary of tokenized datasets
        """
        if not self.tokenizer:
            logger.error("Tokenizer not loaded. Load the model first.")
            return datasets
        
        max_length = self.dataset_config.get("max_length", 1024)
        logger.info(f"Tokenizing datasets with max_length={max_length}")
        
        def tokenize_function(examples):
            # Extract instructions and responses
            instructions = examples.get("instruction", [""] * len(examples["text"]))
            responses = examples.get("response", [""] * len(examples["text"]))
            
            # Process context and intent if available
            context_list = examples.get("context", ["{}" for _ in range(len(examples["text"]))])
            intent_list = examples.get("intent", ["" for _ in range(len(examples["text"]))])
            
            # Enhance instructions with context and intent where available
            enhanced_instructions = []
            for i, (instruction, context_str, intent) in enumerate(zip(instructions, context_list, intent_list)):
                try:
                    # Try to parse context as JSON (it might be a string representation)
                    if isinstance(context_str, str) and context_str.strip().startswith("{"):
                        context = json.loads(context_str)
                    else:
                        context = {}
                except (json.JSONDecodeError, TypeError):
                    context = {}
                
                # Build enhanced instruction with context and intent
                enhanced = instruction
                
                # If instruction doesn't already mention being an assistant, add it
                if "You are a helpful AI assistant" not in enhanced:
                    enhanced = "You are a helpful AI assistant. " + enhanced
                
                # If context has entities or topics and they're not already in the instruction
                if (context.get("entities") or context.get("topics")) and "Entities" not in enhanced and "Topics" not in enhanced:
                    enhanced += "\n\nContext information:"
                    if context.get("entities"):
                        entities = context["entities"][:5] if isinstance(context["entities"], list) else []
                        if entities:
                            enhanced += f"\nEntities: {', '.join(entities)}"
                    if context.get("topics"):
                        topics = context["topics"][:3] if isinstance(context["topics"], list) else []
                        if topics:
                            enhanced += f"\nTopics: {', '.join(topics)}"
                
                # Add intent if available and not already in the instruction
                if intent and "intent" not in enhanced.lower():
                    enhanced += f"\nUser intent: {intent}"
                
                enhanced_instructions.append(enhanced)
            
            # Tokenize inputs
            model_inputs = self.tokenizer(
                enhanced_instructions,
                max_length=max_length,
                padding="max_length",
                truncation=True
            )
            
            # Tokenize targets/responses
            labels = self.tokenizer(
                responses,
                max_length=max_length,
                padding="max_length",
                truncation=True
            )
            
            # Set the labels in the model inputs
            model_inputs["labels"] = labels["input_ids"]
            
            return model_inputs
        
        # Apply tokenization to each dataset
        tokenized_datasets = {}
        for split, dataset in datasets.items():
            logger.info(f"Tokenizing {split} dataset with {len(dataset)} examples")
            tokenized_datasets[split] = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names
            )
            
            # Set the tensor format to PyTorch
            tokenized_datasets[split].set_format("torch")
            
            logger.info(f"Tokenized {split} dataset: {len(tokenized_datasets[split])} examples")
        
        return tokenized_datasets
    
    def _configure_training_arguments(self) -> TrainingArguments:
        """Configure training arguments for the trainer."""
        # Extract training configuration
        output_dir = self.training_config.get("output_dir", "models/flan-ut-20b-context-aware")
        num_train_epochs = self.training_config.get("num_train_epochs", 3)
        per_device_train_batch_size = self.training_config.get("per_device_train_batch_size", 2)
        per_device_eval_batch_size = self.training_config.get("per_device_eval_batch_size", 2)
        gradient_accumulation_steps = self.training_config.get("gradient_accumulation_steps", 16)
        evaluation_strategy = self.training_config.get("evaluation_strategy", "steps")
        eval_steps = self.training_config.get("eval_steps", 200)
        save_strategy = self.training_config.get("save_strategy", "steps")
        save_steps = self.training_config.get("save_steps", 500)
        save_total_limit = self.training_config.get("save_total_limit", 3)
        learning_rate = self.training_config.get("learning_rate", 1e-4)
        weight_decay = self.training_config.get("weight_decay", 0.01)
        warmup_ratio = self.training_config.get("warmup_ratio", 0.03)
        lr_scheduler_type = self.training_config.get("lr_scheduler_type", "cosine")
        logging_steps = self.training_config.get("logging_steps", 50)
        report_to = self.training_config.get("report_to", ["tensorboard"])
        seed = self.training_config.get("seed", 42)
        fp16 = self.training_config.get("fp16", False)
        bf16 = self.training_config.get("bf16", True)
        max_grad_norm = self.training_config.get("max_grad_norm", 1.0)
        
        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            evaluation_strategy=evaluation_strategy,
            eval_steps=eval_steps,
            save_strategy=save_strategy,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type,
            logging_steps=logging_steps,
            report_to=report_to,
            seed=seed,
            fp16=fp16,
            bf16=bf16,
            push_to_hub=self.training_config.get("push_to_hub", False),
            max_grad_norm=max_grad_norm,
            dataloader_num_workers=self.training_config.get("dataloader_num_workers", 2),
            group_by_length=self.training_config.get("group_by_length", True),
            optim=self.training_config.get("optim", "adamw_torch"),
            ddp_find_unused_parameters=self.training_config.get("ddp_find_unused_parameters", False),
            load_best_model_at_end=self.training_config.get("load_best_model_at_end", True),
            metric_for_best_model=self.training_config.get("metric_for_best_model", "eval_loss"),
            greater_is_better=self.training_config.get("greater_is_better", False)
        )
        
        return training_args
    
    def train(self, data_dir: str) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            data_dir: Directory containing the processed datasets
            
        Returns:
            Training metrics
        """
        # Load the model and tokenizer
        if not self.model or not self.tokenizer:
            self._load_model()
        
        # Load and prepare datasets
        datasets = self._load_and_prepare_datasets(data_dir)
        
        # Tokenize datasets
        tokenized_datasets = self._tokenize_datasets(datasets)
        
        # Configure training arguments
        training_args = self._configure_training_arguments()
        
        # Set up the data collator for padding
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding="max_length",
            max_length=self.dataset_config.get("max_length", 1024),
            label_pad_token_id=-100
        )
        
        # Initialize the trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        
        # Train the model
        logger.info("Starting training...")
        train_result = trainer.train(
            resume_from_checkpoint=self.training_config.get("resume_from_checkpoint", True)
        )
        
        # Save the model
        logger.info(f"Saving model to {training_args.output_dir}")
        trainer.save_model(training_args.output_dir)
        self.tokenizer.save_pretrained(training_args.output_dir)
        
        # Save training metrics
        metrics = train_result.metrics
        trainer.save_metrics("train", metrics)
        
        # Evaluate on the test set
        if tokenized_datasets["test"]:
            logger.info("Evaluating on test set...")
            test_metrics = trainer.evaluate(tokenized_datasets["test"], metric_key_prefix="test")
            trainer.save_metrics("test", test_metrics)
            metrics.update(test_metrics)
        
        logger.info(f"Training complete. Metrics: {metrics}")
        return metrics
    
    def generate_with_context(
        self,
        prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate text with context and intent awareness.
        
        Args:
            prompt: The user prompt
            conversation_history: Previous conversation turns
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Generated response
        """
        if not self.model or not self.tokenizer:
            logger.error("Model not loaded. Load the model first.")
            return "Error: Model not loaded."
        
        # Initialize components if they don't exist
        if not self.conversation_memory and ConversationMemory:
            self.conversation_memory = ConversationMemory()
        if not self.intent_analyzer and IntentAnalyzer:
            self.intent_analyzer = IntentAnalyzer()
        if not self.context_extractor and ContextExtractor:
            self.context_extractor = ContextExtractor()
        
        # If no memory or analyzers are available, just do simple generation
        if not self.conversation_memory or not self.intent_analyzer or not self.context_extractor:
            logger.warning("Context-aware components not available. Performing simple generation.")
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
            
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0
            )
            
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Process conversation history
        if conversation_history:
            # Clear any existing memory
            self.conversation_memory.clear()
            
            # Add history to memory
            for turn in conversation_history:
                user_text = turn.get("user", "")
                assistant_text = turn.get("assistant", "")
                
                if user_text and assistant_text:
                    context = self.context_extractor.extract_context(user_text)
                    intent = self.intent_analyzer.get_primary_intent(user_text)
                    self.conversation_memory.add_turn(user_text, assistant_text, context, intent)
        
        # Process current prompt
        context = self.context_extractor.extract_context(prompt)
        intent = self.intent_analyzer.get_primary_intent(prompt)
        
        # Prepare enhanced prompt with context and intent
        enhanced_prompt = "You are a helpful AI assistant. Respond to the following message using the provided context:\n\n"
        
        # Add conversation history if available
        if self.conversation_memory.conversation_history:
            enhanced_prompt += "Previous conversation:\n"
            enhanced_prompt += self.conversation_memory.get_formatted_history() + "\n\n"
        
        # Add context information
        if context.get("entities"):
            enhanced_prompt += f"Entities mentioned: {', '.join(context['entities'][:5])}\n"
        if context.get("topics"):
            enhanced_prompt += f"Topics: {', '.join(context['topics'])}\n"
        
        # Add intent information
        enhanced_prompt += f"The user's intent is: {intent}\n\n"
        
        # Add the current prompt
        enhanced_prompt += f"User message: {prompt}"
        
        # Generate response
        inputs = self.tokenizer.encode(enhanced_prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Update memory with this turn
        self.conversation_memory.add_turn(prompt, response, context, intent)
        
        return response 