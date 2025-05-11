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
        self.begin_token = ""
        
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
            # Ensure batch_size is never zero to avoid range() error
            batch_size = max(1, min(len(filtered_texts), 32))  # At minimum 1, max 32
            all_lengths = []
            
            # Handle empty filtered_texts
            if not filtered_texts:
                return {"processed_text": [], "length": []}
            
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
            # Handle empty filtered_texts for non-streaming mode too
            if not filtered_texts:
                return {"processed_text": [], "length": []}
                
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
                             streaming: bool = False, language: str = None) -> Union[Dataset, DatasetDict]:
        """Process CodeSearchNet dataset."""
        if language:
            logger.info(f"Processing CodeSearchNet dataset for {language}...")
            # Filter dataset by language if specified
            if "language" in dataset.column_names:
                dataset = dataset.filter(lambda x: x["language"].lower() == language.lower())
        else:
            logger.info("Processing CodeSearchNet dataset for all languages...")
        
        def process_sample(examples):
            # Handle different field names depending on the dataset structure
            docstrings = []
            code_snippets = []
            languages = examples.get("language", [])
            has_language = len(languages) > 0
            
            # Try to extract the docstring and code from various field structures
            if "function" in examples and isinstance(examples["function"], list):
                for i, item in enumerate(examples["function"]):
                    if isinstance(item, dict):
                        docstrings.append(item.get("docstring", ""))
                        code_snippets.append(item.get("function", ""))
            elif "docstring" in examples and "code" in examples:
                docstrings = examples["docstring"]
                code_snippets = examples["code"]
            elif "docstring" in examples and "function" in examples:
                docstrings = examples["docstring"]
                code_snippets = examples["function"]
            else:
                # Fallback: create simple prompts
                if has_language:
                    docstrings = [f"Write a {lang} function" for lang in languages]
                else:
                    docstrings = ["Write a function" for _ in range(len(examples.get("code", examples.get("function", []))))]
                code_snippets = examples.get("code", examples.get("function", []))
            
            # Create language-specific prompts if language information is available
            if has_language:
                prompts = [f"'''{doc}'''\n# Write a {lang} function" 
                          for doc, lang in zip(docstrings, languages)]
            else:
                prompts = [f"'''{doc}'''\n# Write a function" for doc in docstrings]
            
            return self.common_preprocessing(
                {"prompt": prompts, "completion": code_snippets},
                "prompt", "completion",
                streaming=streaming
            )
        
        try:
            return dataset.map(
                process_sample,
                batched=True, 
                remove_columns=dataset.column_names if not streaming else None,
                batch_size=100 if streaming else None
            )
        except Exception as e:
            logger.warning(f"Error removing columns in streaming mode: {e}")
            return dataset.map(
                process_sample,
                batched=True,
                batch_size=100 if streaming else None
            )
    
    def process_code_alpaca(self, dataset: Union[Dataset, DatasetDict],
                           streaming: bool = False) -> Union[Dataset, DatasetDict]:
        """Process CodeAlpaca-20K dataset."""
        logger.info("Processing CodeAlpaca dataset...")
        
        # For streaming mode, process one example at a time to avoid issues
        if streaming:
            processed_examples = []
            
            # Process each example individually
            for i, example in enumerate(dataset):
                if i >= 10000:  # Limit to prevent processing too many examples
                    break
                    
                try:
                    # Check for field names that might be present
                    instruction = None
                    output = None
                    
                    if isinstance(example, dict):
                        if "instruction" in example:
                            instruction = example["instruction"]
                        elif "input" in example:
                            instruction = example["input"]
                            
                        if "output" in example:
                            output = example["output"]
                        elif "completion" in example:
                            output = example["completion"]
                        elif "response" in example:
                            output = example["response"]
                    
                    # Skip if missing required fields
                    if not instruction or not output:
                        continue
                        
                    processed = self.common_preprocessing(
                        {"prompt": instruction, "completion": output},
                        "prompt", "completion",
                        streaming=streaming
                    )
                    processed_examples.append(processed)
                except Exception as e:
                    logger.warning(f"Error processing example {i}: {e}")
                    continue
                
            return processed_examples
        else:
            # For non-streaming mode
            try:
                # Try to determine the field names from a sample
                sample_example = next(iter(dataset))
                
                # Find appropriate field names
                prompt_field = "instruction"
                completion_field = "output"
                
                if isinstance(sample_example, dict):
                    if "instruction" not in sample_example and "input" in sample_example:
                        prompt_field = "input"
                    
                    if "output" not in sample_example:
                        if "completion" in sample_example:
                            completion_field = "completion"
                        elif "response" in sample_example:
                            completion_field = "response"
                
                return dataset.map(
                    lambda examples: self.common_preprocessing(
                        {"prompt": examples[prompt_field], "completion": examples[completion_field]},
                        "prompt", "completion",
                        streaming=streaming
                    ),
                    batched=True,
                    remove_columns=dataset.column_names if not streaming else None,
                    batch_size=100 if streaming else None
                )
            except Exception as e:
                logger.warning(f"Error in batch processing mode: {e}")
                # Fallback to simple processing with expected field names
                return dataset.map(
                    lambda examples: self.common_preprocessing(
                        {"prompt": examples.get("instruction", examples.get("input", "")), 
                         "completion": examples.get("output", examples.get("completion", examples.get("response", "")))},
                        "prompt", "completion",
                        streaming=streaming
                    ),
                    batched=True,
                    batch_size=100 if streaming else None
                )
    
    def process_instruct_code(self, dataset: Union[Dataset, DatasetDict],
                             streaming: bool = False) -> Union[Dataset, DatasetDict]:
        """Process InstructCode dataset."""
        logger.info("Processing InstructCode dataset...")
        
        def process_sample(examples):
            # Check for different field structures
            prompts = []
            completions = []
            
            # Try to extract from common field names
            if "instruction" in examples and "output" in examples:
                prompts = examples["instruction"]
                completions = examples["output"]
            elif "instructions" in examples and "response" in examples:
                prompts = examples["instructions"]
                completions = examples["response"]
            elif "conversations" in examples:
                # Handle conversation format
                for conv_list in examples["conversations"]:
                    if isinstance(conv_list, list) and len(conv_list) >= 2:
                        # Extract first user message as prompt and first assistant message as completion
                        user_msgs = [msg.get("value", "") for msg in conv_list if msg.get("role", "") == "user"]
                        assistant_msgs = [msg.get("value", "") for msg in conv_list if msg.get("role", "") == "assistant"]
                        
                        if user_msgs and assistant_msgs:
                            prompts.append(user_msgs[0])
                            completions.append(assistant_msgs[0])
            elif "messages" in examples:
                # Handle messages format
                for msg_list in examples["messages"]:
                    if isinstance(msg_list, list) and len(msg_list) >= 2:
                        user_msgs = [msg.get("content", "") for msg in msg_list if msg.get("role", "") == "user"]
                        assistant_msgs = [msg.get("content", "") for msg in msg_list if msg.get("role", "") == "assistant"]
                        
                        if user_msgs and assistant_msgs:
                            prompts.append(user_msgs[0])
                            completions.append(assistant_msgs[0])
            
            if not prompts or not completions:
                # Fallback: try to find any text fields to use
                for key in examples:
                    if "prompt" in key.lower() or "instruction" in key.lower() or "input" in key.lower():
                        prompts = examples[key]
                        break
                        
                for key in examples:
                    if "completion" in key.lower() or "response" in key.lower() or "output" in key.lower() or "answer" in key.lower():
                        completions = examples[key]
                        break
            
            # Ensure we have matching lengths
            min_len = min(len(prompts), len(completions))
            prompts = prompts[:min_len]
            completions = completions[:min_len]
            
            return self.common_preprocessing(
                {"prompt": prompts, "completion": completions},
                "prompt", "completion",
                streaming=streaming
            )
        
        try:
            return dataset.map(
                process_sample,
                batched=True,
                remove_columns=dataset.column_names if not streaming else None,
                batch_size=100 if streaming else None
            )
        except Exception as e:
            logger.warning(f"Error removing columns in streaming mode: {e}")
            return dataset.map(
                process_sample,
                batched=True,
                batch_size=100 if streaming else None
            )
    
    def process_mbpp(self, dataset: Union[Dataset, DatasetDict],
                    streaming: bool = False) -> Union[Dataset, DatasetDict]:
        """Process MBPP dataset."""
        logger.info("Processing MBPP dataset...")
        
        # For streaming mode, process one example at a time to avoid issues
        if streaming:
            processed_examples = []
            
            # Process each example individually
            for i, example in enumerate(dataset):
                if i >= 10000:  # Limit to prevent processing too many examples
                    break
                    
                try:
                    # Check for field names that might be present
                    prompt = None
                    code = None
                    
                    if isinstance(example, dict):
                        if "text" in example:
                            prompt = example["text"]
                        elif "prompt" in example:
                            prompt = example["prompt"]
                        elif "problem" in example:
                            prompt = example["problem"]
                        elif "task_id" in example:
                            prompt = f"Solve task {example['task_id']}"
                            
                        if "code" in example:
                            code = example["code"]
                        elif "solution" in example:
                            code = example["solution"]
                        elif "canonical_solution" in example:
                            code = example["canonical_solution"]
                    
                    # Skip if missing required fields
                    if not prompt or not code:
                        logger.warning(f"Skipping example {i}, missing fields")
                        continue
                        
                    processed = self.common_preprocessing(
                        {"prompt": prompt, "completion": code},
                        "prompt", "completion",
                        streaming=streaming
                    )
                    processed_examples.append(processed)
                except Exception as e:
                    logger.warning(f"Error processing example {i}: {e}")
                    continue
                
            return processed_examples
        else:
            # For non-streaming mode
            try:
                # Try to determine the field names from a sample
                sample_example = next(iter(dataset))
                
                # Find appropriate field names
                prompt_field = "text" 
                code_field = "code"
                
                if isinstance(sample_example, dict):
                    if "text" not in sample_example:
                        if "prompt" in sample_example:
                            prompt_field = "prompt"
                        elif "problem" in sample_example:
                            prompt_field = "problem"
                    
                    if "code" not in sample_example:
                        if "solution" in sample_example:
                            code_field = "solution"
                        elif "canonical_solution" in sample_example:
                            code_field = "canonical_solution"
                
                return dataset.map(
                    lambda examples: self.common_preprocessing(
                        {"prompt": examples[prompt_field], "completion": examples[code_field]},
                        "prompt", "completion",
                        streaming=streaming
                    ),
                    batched=True,
                    remove_columns=dataset.column_names if not streaming else None,
                    batch_size=100 if streaming else None
                )
            except Exception as e:
                logger.warning(f"Error in batch processing mode: {e}")
                # Fallback to simple processing with expected field names
                return dataset.map(
                    lambda examples: self.common_preprocessing(
                        {"prompt": examples.get("text", examples.get("prompt", examples.get("problem", "Write code"))), 
                         "completion": examples.get("code", examples.get("solution", examples.get("canonical_solution", "")))},
                        "prompt", "completion",
                        streaming=streaming
                    ),
                    batched=True,
                    batch_size=100 if streaming else None
                )
    
    def process_ds1000(self, dataset: Union[Dataset, DatasetDict],
                      streaming: bool = False) -> Union[Dataset, DatasetDict]:
        """Process DS-1000 dataset."""
        logger.info("Processing DS-1000 dataset...")
        
        try:
            return dataset.map(
                lambda examples: self.common_preprocessing(
                    {"prompt": examples["prompt"], "completion": examples["canonical_solution"]},
                    "prompt", "completion",
                    streaming=streaming
                ),
                batched=True,
                remove_columns=dataset.column_names if not streaming else None,
                batch_size=100 if streaming else None
            )
        except Exception as e:
            logger.warning(f"Error removing columns in streaming mode: {e}")
            return dataset.map(
                lambda examples: self.common_preprocessing(
                    {"prompt": examples["prompt"], "completion": examples["canonical_solution"]},
                    "prompt", "completion",
                    streaming=streaming
                ),
                batched=True,
                batch_size=100 if streaming else None
            )
    
    def process_codeparrot(self, dataset: Union[Dataset, DatasetDict],
                          streaming: bool = False) -> Union[Dataset, DatasetDict]:
        """Process CodeParrot Clean dataset."""
        logger.info("Processing CodeParrot dataset...")
        
        # For streaming mode, process individually to avoid issues
        if streaming:
            processed_examples = []
            
            for i, example in enumerate(dataset):
                if i >= 10000:  # Limit to prevent processing too many examples
                    break
                
                try:
                    code = None
                    
                    if isinstance(example, dict):
                        # Try different possible field names
                        for field_name in ["code", "content", "source", "text"]:
                            if field_name in example and example[field_name]:
                                code = example[field_name]
                                break
                    else:
                        # If the example is directly a string
                        code = str(example)
                    
                    if not code:
                        logger.warning(f"Skipping example {i}, no code found")
                        continue
                    
                    prompt = "# Write a Python function"
                    processed = self.common_preprocessing(
                        {"prompt": prompt, "completion": code},
                        "prompt", "completion",
                        streaming=streaming
                    )
                    processed_examples.append(processed)
                except Exception as e:
                    logger.warning(f"Error processing example {i}: {e}")
                    continue
            
            return processed_examples
        else:
            # For batch processing
            try:
                # Try to determine the structure from a sample
                sample_example = next(iter(dataset))
                
                # Find the field with code
                code_field = None
                if isinstance(sample_example, dict):
                    for field_name in ["code", "content", "source", "text"]:
                        if field_name in sample_example:
                            code_field = field_name
                            break
                
                if not code_field:
                    logger.warning("Could not find a code field in the dataset")
                    code_field = "code"  # Default fallback
                
                def process_sample(examples):
                    try:
                        # Add a synthetic prompt for instruction-based format
                        if code_field in examples and len(examples[code_field]) > 0:
                            prompts = ["# Write a Python function" for _ in examples[code_field]]
                            return self.common_preprocessing(
                                {"prompt": prompts, "completion": examples[code_field]},
                                "prompt", "completion",
                                streaming=streaming
                            )
                        else:
                            logger.warning(f"Missing code field: {code_field}")
                            return {"processed_text": [], "length": []}
                    except Exception as e:
                        logger.error(f"Error in process_sample: {e}")
                        return {"processed_text": [], "length": []}
                
                return dataset.map(
                    process_sample,
                    batched=True,
                    remove_columns=dataset.column_names if not streaming else None,
                    batch_size=100 if streaming else None
                )
            except Exception as e:
                logger.warning(f"Error in batch processing: {e}")
                
                # Fallback to simple item-by-item processing
                processed_examples = []
                count = 0
                
                for example in dataset:
                    if count >= 10000:
                        break
                    
                    try:
                        code = str(example)
                        prompt = "# Write a Python function"
                        
                        processed = self.common_preprocessing(
                            {"prompt": prompt, "completion": code},
                            "prompt", "completion",
                            streaming=True  # Force streaming mode for this fallback
                        )
                        processed_examples.append(processed)
                        count += 1
                    except:
                        continue
                
                return processed_examples
    
    def process_the_stack(self, dataset: Union[Dataset, DatasetDict], language: str = None,
                         streaming: bool = False) -> Union[Dataset, DatasetDict]:
        """Process The Stack dataset."""
        if language:
            logger.info(f"Processing The Stack dataset for {language}...")
        else:
            logger.info("Processing The Stack dataset for all languages...")
        
        # Filter for permissive licenses
        permissive_licenses = ["mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "cc0-1.0", "isc"]
        filtered_dataset = dataset.filter(lambda x: x.get("license", "").lower() in permissive_licenses)
        
        # Filter by language if specified
        if language and "lang" in filtered_dataset.column_names:
            filtered_dataset = filtered_dataset.filter(lambda x: x["lang"].lower() == language.lower())
        
        def process_sample(examples):
            languages = []
            
            # Try to get language information
            if "lang" in examples:
                languages = examples["lang"]
            
            # Generate synthetic prompts with language-specific text if available
            if len(languages) > 0:
                prompts = [f"# Implement a function in {lang}" for lang in languages]
            else:
                # Default to generic prompt if language info isn't available
                prompts = [f"# Implement a function" for _ in examples["content"]]
                
            return self.common_preprocessing(
                {"prompt": prompts, "completion": examples["content"]},
                "prompt", "completion",
                streaming=streaming
            )
        
        try:
            return filtered_dataset.map(
                process_sample,
                batched=True,
                remove_columns=filtered_dataset.column_names if not streaming else None,
                batch_size=100 if streaming else None
            )
        except Exception as e:
            logger.warning(f"Error removing columns in streaming mode: {e}")
            return filtered_dataset.map(
                process_sample,
                batched=True,
                batch_size=100 if streaming else None
            )
    
    def process_humaneval(self, dataset: Union[Dataset, DatasetDict],
                         streaming: bool = False) -> Union[Dataset, DatasetDict]:
        """Process HumanEval dataset."""
        logger.info("Processing HumanEval dataset...")
        
        # For HumanEval, we need special handling for streaming mode
        if streaming:
            processed_examples = []
            
            # Process each example individually
            for example in dataset:
                # Handle both dictionary structure and direct access
                if isinstance(example, dict):
                    prompt = example.get("prompt", "")
                    solution = example.get("canonical_solution", "")
                    # Some versions use different field names
                    if not solution and "test" in example:
                        solution = example.get("test", "")
                    if not solution and "solution" in example:
                        solution = example.get("solution", "")
                else:
                    # For the case where each example is directly a string
                    try:
                        prompt = "Write a Python function"
                        solution = str(example)
                    except:
                        logger.warning(f"Could not process example: {example}")
                        continue
                
                processed = self.common_preprocessing(
                    {"prompt": prompt, "completion": solution},
                    "prompt", "completion",
                    streaming=streaming
                )
                processed_examples.append(processed)
                
            return processed_examples
        else:
            try:
                # Determine the actual field names in the dataset
                sample_item = next(iter(dataset))
                prompt_field = "prompt"
                solution_field = "canonical_solution"
                
                # Check field names
                if isinstance(sample_item, dict):
                    if "prompt" not in sample_item and "task_id" in sample_item:
                        prompt_field = "task_id"
                    if "canonical_solution" not in sample_item:
                        if "test" in sample_item:
                            solution_field = "test"
                        elif "solution" in sample_item:
                            solution_field = "solution"
                
                return dataset.map(
                    lambda examples: self.common_preprocessing(
                        {"prompt": examples[prompt_field], "completion": examples[solution_field]},
                        "prompt", "completion",
                        streaming=streaming
                    ),
                    batched=True,
                    remove_columns=dataset.column_names if not streaming else None,
                    batch_size=100 if streaming else None
                )
            except Exception as e:
                logger.warning(f"Error processing HumanEval: {e}")
                return dataset.map(
                    lambda examples: self.common_preprocessing(
                        {"prompt": "Write a Python function", "completion": str(examples)},
                        "prompt", "completion",
                        streaming=streaming
                    ),
                    batched=True
                )
    
    def load_and_process_all_datasets(self, dataset_config: Dict, save_path: str) -> Dict[str, Dataset]:
        """Load and process all configured datasets."""
        processed_datasets = {}
        
        # Check for HF_TOKEN environment variable
        if os.environ.get("HF_TOKEN") is None:
            logger.warning("HF_TOKEN environment variable not set. Some gated datasets might be inaccessible.")
            logger.warning("You may need to set it using: export HF_TOKEN=your_huggingface_token")
        else:
            logger.info("Using Hugging Face token from environment for authentication")
            
        for dataset_name, config in dataset_config.items():
            # Skip datasets that are explicitly disabled
            if config.get("enabled") is False:
                logger.info(f"Skipping disabled dataset: {dataset_name}")
                continue
                
            logger.info(f"Loading dataset: {dataset_name}")
            
            try:
                # Get streaming and caching options
                streaming = config.get("streaming", False)
                use_cache = config.get("use_cache", True)
                max_samples = config.get("max_samples", 10000)
                
                # Set cache directory if needed
                if not use_cache:
                    os.environ["HF_DATASETS_CACHE"] = "no"
                
                # Load the dataset with proper error handling
                try:
                    logger.info(f"Attempting to load dataset {config['path']}")
                    
                    # Use token parameter instead of deprecated use_auth_token
                    # For compatibility with both older and newer versions of the library, try both methods
                    hf_token = os.environ.get("HF_TOKEN")
                    try:
                        # Try with the newer 'token' parameter first
                        dataset = load_dataset(
                            config["path"], 
                            name=config.get("name"), 
                            split=config.get("split"),
                            streaming=streaming,
                            token=hf_token,  # New parameter name
                            trust_remote_code=True
                        )
                    except TypeError as e:
                        # If the newer parameter didn't work, fall back to the older one
                        if "got an unexpected keyword argument 'token'" in str(e):
                            logger.warning("Falling back to deprecated 'use_auth_token' parameter")
                            dataset = load_dataset(
                                config["path"], 
                                name=config.get("name"), 
                                split=config.get("split"),
                                streaming=streaming,
                                use_auth_token=hf_token,  # Old parameter name
                                trust_remote_code=True
                            )
                        else:
                            raise
                    
                    logger.info(f"Successfully loaded dataset {dataset_name}")
                    
                except (ImportError, ModuleNotFoundError) as e:
                    logger.error(f"Missing dependency for loading dataset {dataset_name}: {str(e)}")
                    logger.error("Try installing additional dependencies if needed")
                    continue
                
                except Exception as e:
                    error_msg = str(e)
                    if "is a gated dataset" in error_msg:
                        logger.error(f"Dataset {config['path']} requires authentication. Make sure your HF_TOKEN has proper access rights.")
                        logger.error("Visit the dataset page on Hugging Face and request access.")
                    elif "doesn't exist on the Hub" in error_msg:
                        logger.error(f"Dataset {config['path']} was not found on Hugging Face Hub.")
                        logger.error("Check the dataset name and make sure it's correct.")
                    else:
                        logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
                    
                    logger.info(f"Skipping dataset {dataset_name}")
                    continue
                
                # Apply the appropriate processing function with streaming flag
                processor_func = getattr(self, f"process_{config['processor']}")
                processor_args = {"streaming": streaming}
                
                # Add language arg for processors that support it
                if "language" in config:
                    if config['processor'] in ["the_stack", "codesearchnet"]:
                        processor_args["language"] = config["language"]
                    elif config.get("languages") is None:  # Only store single language if no languages list is provided
                        config["languages"] = [config["language"]]
                
                # Add support for multiple languages
                if "languages" in config and isinstance(config["languages"], list) and config["languages"]:
                    logger.info(f"Processing dataset {dataset_name} for multiple languages: {', '.join(config['languages'])}")
                    # For processors that have built-in language support, pass language=None to process all languages
                    if config['processor'] in ["the_stack", "codesearchnet"]:
                        processor_args["language"] = None
                
                # Process the dataset
                processed_dataset = processor_func(dataset, **processor_args)
                
                # Create save directory
                os.makedirs(save_path, exist_ok=True)
                save_path_for_dataset = os.path.join(save_path, f"{dataset_name}_processed")
                
                # For streaming datasets we need to materialize and convert to non-streaming format
                if streaming:
                    logger.info(f"Materializing streaming dataset {dataset_name} (taking {max_samples} samples)...")
                    
                    # Create standard format dataset
                    from datasets import Dataset as HFDataset
                    
                    # Collect all examples in memory
                    collected_data = {"processed_text": [], "length": []}
                    count = 0
                    errors = 0
                    
                    # Handle different return types from processor
                    if isinstance(processed_dataset, list):
                        # For HumanEval or other datasets that return a list
                        for example in processed_dataset:
                            if count >= max_samples:
                                break
                            
                            try:
                                if "processed_text" in example and "length" in example:
                                    collected_data["processed_text"].append(example["processed_text"])
                                    collected_data["length"].append(example["length"])
                                    count += 1
                            except Exception as e:
                                logger.warning(f"Error collecting example: {e}")
                                errors += 1
                                if errors > 100:
                                    logger.error("Too many errors, stopping collection")
                                    break
                    else:
                        # For regular streaming datasets
                        for example in processed_dataset:
                            if count >= max_samples:
                                break
                                
                            try:
                                if "processed_text" in example and "length" in example:
                                    collected_data["processed_text"].append(example["processed_text"])
                                    collected_data["length"].append(example["length"])
                                    count += 1
                            except Exception as e:
                                logger.warning(f"Error collecting example: {e}")
                                errors += 1
                                if errors > 100:
                                    logger.error("Too many errors, stopping collection")
                                    break
                    
                    if count == 0:
                        logger.error(f"No examples could be collected for {dataset_name}")
                        continue
                        
                    # Convert to HF Dataset
                    materialized_dataset = HFDataset.from_dict(collected_data)
                    
                    # Save the materialized dataset
                    materialized_dataset.save_to_disk(save_path_for_dataset)
                    logger.info(f"Saved materialized dataset ({len(materialized_dataset)} examples) to {save_path_for_dataset}")
                    
                    # Store for return
                    processed_datasets[dataset_name] = materialized_dataset
                    
                else:
                    # For non-streaming datasets, save directly
                    processed_dataset.save_to_disk(save_path_for_dataset)
                    logger.info(f"Saved processed dataset to {save_path_for_dataset}")
                    processed_datasets[dataset_name] = processed_dataset
                
                logger.info(f"Successfully processed and saved {dataset_name}")
            
            except Exception as e:
                logger.error(f"Error processing {dataset_name}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        return processed_datasets 