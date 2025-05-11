import os
import re
import json
import logging
from typing import Dict, List, Optional, Union, Callable
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, tokenizer_path: str = "deepseek-ai/deepseek-coder-6.7b-base", max_length: int = 2048):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            use_auth_token=os.environ.get("HF_TOKEN")
        )
        self.max_length = max_length
        self.begin_token = "<｜begin▁of▁sentence｜>"
        
    def common_preprocessing(self, examples: Dict, prompt_field: str, completion_field: str,
                            lowercase: bool = False, streaming: bool = False) -> Dict:
        """Apply common preprocessing steps to all datasets."""
        
        # Clean whitespace, strip leading/trailing spaces and newlines
        prompts = [re.sub(r'\s+', ' ', str(p)).strip() for p in examples[prompt_field]]
        completions = [re.sub(r'\s+', ' ', str(c)).strip() for c in examples[completion_field]]
        
        # Optionally lowercase all text
        if lowercase:
            prompts = [p.lower() for p in prompts]
            completions = [c.lower() for c in completions]
            
        # Format each sample with the required template
        formatted_texts = [
            f"{self.begin_token}User: {prompt}\nAssistant: {completion}"
            for prompt, completion in zip(prompts, completions)
        ]
        
        # Filter out empty, null, or very short examples
        valid_indices = [i for i, text in enumerate(formatted_texts) 
                        if len(text) > 10 and "None" not in text]
        
        filtered_texts = [formatted_texts[i] for i in valid_indices]
        
        # For streaming mode, process in smaller batches to save memory
        if streaming:
            batch_size = min(len(filtered_texts), 32)  # Process in small batches
            all_lengths = []
            
            for i in range(0, len(filtered_texts), batch_size):
                batch = filtered_texts[i:i+batch_size]
                # Tokenize and truncate to max length
                tokenized = self.tokenizer(
                    batch,
                    truncation=True,
                    max_length=self.max_length,
                    return_overflowing_tokens=False,
                    return_length=True
                )
                all_lengths.extend(tokenized["length"])
                
                # Clear GPU cache after processing batch
                if hasattr(self.tokenizer, "cache_clear"):
                    self.tokenizer.cache_clear()
                
            return {"processed_text": filtered_texts, "length": all_lengths}
        else:
            # Tokenize and truncate to max length for non-streaming mode
            tokenized = self.tokenizer(
                filtered_texts,
                truncation=True,
                max_length=self.max_length,
                return_overflowing_tokens=False,
                return_length=True
            )
            
            return {"processed_text": filtered_texts, "length": tokenized["length"]}
    
    def process_codesearchnet(self, dataset: Union[Dataset, DatasetDict], 
                             streaming: bool = False) -> Union[Dataset, DatasetDict]:
        """Process CodeSearchNet dataset."""
        logger.info("Processing CodeSearchNet dataset...")
        
        def process_sample(examples):
            prompts = [f"'''{docstring}'''\n# Write a Python function" for docstring in examples["docstring"]]
            return self.common_preprocessing(
                {"prompt": prompts, "completion": examples["code"]},
                "prompt", "completion",
                streaming=streaming
            )
        
        return dataset.map(
            process_sample,
            batched=True, 
            remove_columns=dataset.column_names,
            batch_size=100 if streaming else None
        )
    
    def process_code_alpaca(self, dataset: Union[Dataset, DatasetDict],
                           streaming: bool = False) -> Union[Dataset, DatasetDict]:
        """Process CodeAlpaca-20K dataset."""
        logger.info("Processing CodeAlpaca dataset...")
        
        return dataset.map(
            lambda examples: self.common_preprocessing(
                {"prompt": examples["instruction"], "completion": examples["output"]},
                "prompt", "completion",
                streaming=streaming
            ),
            batched=True,
            remove_columns=dataset.column_names,
            batch_size=100 if streaming else None
        )
    
    def process_instruct_code(self, dataset: Union[Dataset, DatasetDict],
                             streaming: bool = False) -> Union[Dataset, DatasetDict]:
        """Process InstructCode dataset."""
        logger.info("Processing InstructCode dataset...")
        
        return dataset.map(
            lambda examples: self.common_preprocessing(
                {"prompt": examples["prompt"], "completion": examples["completion"]},
                "prompt", "completion",
                streaming=streaming
            ),
            batched=True,
            remove_columns=dataset.column_names,
            batch_size=100 if streaming else None
        )
    
    def process_mbpp(self, dataset: Union[Dataset, DatasetDict],
                    streaming: bool = False) -> Union[Dataset, DatasetDict]:
        """Process MBPP dataset."""
        logger.info("Processing MBPP dataset...")
        
        return dataset.map(
            lambda examples: self.common_preprocessing(
                {"prompt": examples["text"], "completion": examples["code"]},
                "prompt", "completion",
                streaming=streaming
            ),
            batched=True,
            remove_columns=dataset.column_names,
            batch_size=100 if streaming else None
        )
    
    def process_ds1000(self, dataset: Union[Dataset, DatasetDict],
                      streaming: bool = False) -> Union[Dataset, DatasetDict]:
        """Process DS-1000 dataset."""
        logger.info("Processing DS-1000 dataset...")
        
        return dataset.map(
            lambda examples: self.common_preprocessing(
                {"prompt": examples["prompt"], "completion": examples["canonical_solution"]},
                "prompt", "completion",
                streaming=streaming
            ),
            batched=True,
            remove_columns=dataset.column_names,
            batch_size=100 if streaming else None
        )
    
    def process_codeparrot(self, dataset: Union[Dataset, DatasetDict],
                          streaming: bool = False) -> Union[Dataset, DatasetDict]:
        """Process CodeParrot Clean dataset."""
        logger.info("Processing CodeParrot dataset...")
        
        def process_sample(examples):
            # Add a synthetic prompt for instruction-based format
            prompts = ["# Write a Python function" for _ in examples["code"]]
            return self.common_preprocessing(
                {"prompt": prompts, "completion": examples["code"]},
                "prompt", "completion",
                streaming=streaming
            )
        
        return dataset.map(
            process_sample,
            batched=True,
            remove_columns=dataset.column_names,
            batch_size=100 if streaming else None
        )
    
    def process_the_stack(self, dataset: Union[Dataset, DatasetDict], language: str = "python",
                         streaming: bool = False) -> Union[Dataset, DatasetDict]:
        """Process The Stack dataset."""
        logger.info(f"Processing The Stack dataset for {language}...")
        
        # Filter for permissive licenses
        permissive_licenses = ["mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "cc0-1.0", "isc"]
        filtered_dataset = dataset.filter(lambda x: x.get("license", "").lower() in permissive_licenses)
        
        def process_sample(examples):
            # Generate synthetic prompts
            prompts = [f"# Implement a function in {language}" for _ in examples["content"]]
            return self.common_preprocessing(
                {"prompt": prompts, "completion": examples["content"]},
                "prompt", "completion",
                streaming=streaming
            )
        
        return filtered_dataset.map(
            process_sample,
            batched=True,
            remove_columns=filtered_dataset.column_names,
            batch_size=100 if streaming else None
        )
    
    def process_humaneval(self, dataset: Union[Dataset, DatasetDict],
                         streaming: bool = False) -> Union[Dataset, DatasetDict]:
        """Process HumanEval dataset."""
        logger.info("Processing HumanEval dataset...")
        
        return dataset.map(
            lambda examples: self.common_preprocessing(
                {"prompt": examples["prompt"], "completion": examples["canonical_solution"]},
                "prompt", "completion",
                streaming=streaming
            ),
            batched=True,
            remove_columns=dataset.column_names,
            batch_size=100 if streaming else None
        )
    
    def load_and_process_all_datasets(self, dataset_config: Dict, save_path: str) -> Dict[str, Dataset]:
        """Load and process all configured datasets."""
        processed_datasets = {}
        
        for dataset_name, config in dataset_config.items():
            logger.info(f"Loading dataset: {dataset_name}")
            
            try:
                # Get streaming and caching options
                streaming = config.get("streaming", False)
                use_cache = config.get("use_cache", True)
                
                # Set cache directory if needed
                if not use_cache:
                    os.environ["HF_DATASETS_CACHE"] = "no"
                
                # Load the dataset
                dataset = load_dataset(
                    config["path"], 
                    name=config.get("name"), 
                    split=config.get("split"),
                    streaming=streaming,
                    use_auth_token=os.environ.get("HF_TOKEN"),
                    trust_remote_code=True
                )
                
                # Apply the appropriate processing function with streaming flag
                processor_func = getattr(self, f"process_{config['processor']}")
                processor_args = {"streaming": streaming}
                
                # Add language arg for the_stack processor
                if config['processor'] == "the_stack" and "language" in config:
                    processor_args["language"] = config["language"]
                
                processed_dataset = processor_func(dataset, **processor_args)
                
                # Save the processed dataset
                os.makedirs(save_path, exist_ok=True)
                save_path_for_dataset = os.path.join(save_path, f"{dataset_name}_processed")
                
                # For streaming datasets, we need to materialize them before saving
                if streaming:
                    logger.info(f"Materializing streaming dataset {dataset_name} before saving...")
                    processed_dataset = processed_dataset.take(config.get("max_samples", 100000))
                
                processed_dataset.save_to_disk(save_path_for_dataset)
                logger.info(f"Saved processed dataset to {save_path_for_dataset}")
                
                processed_datasets[dataset_name] = processed_dataset
                logger.info(f"Successfully processed and saved {dataset_name}")
            
            except Exception as e:
                logger.error(f"Error processing {dataset_name}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        return processed_datasets 