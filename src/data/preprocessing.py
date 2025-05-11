import os
import re
import json
import logging
import traceback
import gc
import psutil
import time
import sys
from typing import Dict, List, Optional, Union, Callable, Tuple, Iterator, Any, Set
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from itertools import islice
from tqdm import tqdm
import random

# For language detection
try:
    import langid
    LANGID_AVAILABLE = True
except ImportError:
    LANGID_AVAILABLE = False
    
# For GPU memory management
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    # Define popular programming languages - include those in CodeSearchNet plus popular ones
    POPULAR_PROGRAMMING_LANGUAGES = {
        'python', 'javascript', 'java', 'go', 'php', 'ruby',  # CodeSearchNet languages
        'c', 'cpp', 'csharp', 'c++', 'c#',                    # C family
        'typescript', 'rust', 'kotlin', 'swift', 'scala',     # Other popular langs
        'html', 'css', 'sql', 'bash', 'shell',                # Web/scripting
    }
    
    # Natural languages to include
    ALLOWED_NATURAL_LANGUAGES = {'en', 'ar'}  # English and Arabic
    
    def __init__(self, tokenizer_path: str = "deepseek-ai/deepseek-coder-6.7b-base", max_length: int = 2048):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            use_auth_token=os.environ.get("HF_TOKEN")
        )
        self.max_length = max_length
        self.begin_token = ""
        self._last_memory_check = 0
        self._memory_check_interval = 1000  # Check memory every 1000 examples
        
        # Try to load langid for language detection
        if not LANGID_AVAILABLE:
            logger.warning("langid not installed - language filtering for comments will be limited. " 
                          "Install with: pip install langid")
                          
        # Check if we can use GPU
        if TORCH_AVAILABLE and torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024):.2f} GB")
        
    def detect_language(self, text: str) -> str:
        """Detect language of text. Returns language code."""
        if not text or not isinstance(text, str) or len(text.strip()) < 10:
            return "unknown"  # Too short to detect
            
        # Use langid if available
        if LANGID_AVAILABLE:
            try:
                lang, _ = langid.classify(text)
                return lang
            except Exception as e:
                logger.warning(f"Language detection failed: {e}")
                
        # Fallback to basic heuristics
        arabic_chars_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
        if arabic_chars_pattern.search(text):
            return "ar"
            
        # Default to English if we can't detect language or no special characters
        return "en"
        
    def should_include_by_language(self, text, allowed_languages=None):
        """
        Check if text should be included based on natural language detection.
        By default, only keep English content, but allow for specifying other languages.
        
        Args:
            text: Text to check
            allowed_languages: List of language codes to allow (e.g., ['en', 'ar']), or None for all
            
        Returns:
            Boolean indicating if this text should be included
        """
        # If no language restrictions, include everything
        if not allowed_languages:
            return True
        
        # Skip empty text
        if not text or len(text.strip()) < 10:
            return False
        
        try:
            # Detect language with langdetect
            lang_code = detect(text[:1000])  # Only use first 1000 chars for speed
            
            # Convert allowed_languages to lowercase for case-insensitive matching
            allowed_languages_lower = [lang.lower() for lang in allowed_languages]
            
            # Include if language is in allowed list
            return lang_code.lower() in allowed_languages_lower
        except Exception as e:
            # In case of detection error, include by default
            logger.debug(f"Language detection error: {str(e)}")
            return True
    
    def _check_resources(self, force: bool = False) -> Dict[str, float]:
        """Monitor system resources and manage memory if needed."""
        current_time = time.time()
        
        # Only check periodically to avoid performance impact
        if not force and (current_time - self._last_memory_check < 60):  # Check max once per minute
            return {}
            
        self._last_memory_check = current_time
        
        try:
            # Get memory info
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
            memory_percent = process.memory_percent()
            
            # Check for GPU memory if available
            gpu_memory_info = {}
            
            try:
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    gpu_memory_info = {
                        "gpu_allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
                        "gpu_reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
                        "gpu_max_mb": torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                    }
                    
                    # Calculate GPU memory percentage
                    if gpu_memory_info["gpu_max_mb"] > 0:
                        gpu_memory_info["gpu_percent"] = (gpu_memory_info["gpu_allocated_mb"] / 
                                                         gpu_memory_info["gpu_max_mb"]) * 100
            except Exception as e:
                # If torch is not available or there's any error, continue without GPU monitoring
                logger.debug(f"Error checking GPU memory: {str(e)}")
            
            # Log resource usage if it's high
            if force or memory_percent > 75 or (gpu_memory_info and gpu_memory_info.get("gpu_percent", 0) > 75):
                logger.info(f"Memory usage: {memory_mb:.1f} MB ({memory_percent:.1f}% of system memory)")
                
                if gpu_memory_info:
                    logger.info(f"GPU memory: {gpu_memory_info.get('gpu_allocated_mb', 0):.1f} MB "
                               f"({gpu_memory_info.get('gpu_percent', 0):.1f}% of GPU memory)")
                
                # Do some cleanup if memory usage is very high
                if memory_percent > 85 or (gpu_memory_info and gpu_memory_info.get("gpu_percent", 0) > 80):
                    logger.warning("High memory usage detected, performing cleanup")
                    self._cleanup_memory()
            
            # Combine system and GPU info
            result = {
                "memory_mb": memory_mb,
                "memory_percent": memory_percent,
            }
            
            # Add GPU info if available
            if gpu_memory_info:
                result.update(gpu_memory_info)
                
            return result
            
        except Exception as e:
            logger.error(f"Error monitoring resources: {str(e)}")
            return {}
    
    def _cleanup_memory(self):
        """Release memory when usage is high."""
        # Clear GPU memory first if torch is available
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                logger.info("Clearing CUDA cache")
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Error clearing CUDA cache: {str(e)}")
            
        # Force garbage collection
        collected = gc.collect()
        logger.info(f"Garbage collected {collected} objects")
    
    def _safe_dataset_iterator(self, dataset: Union[Dataset, DatasetDict, List], 
                              max_samples: int = 10000) -> Iterator[Any]:
        """Safely iterate through a dataset with proper error handling and progress tracking."""
        count = 0
        errors = 0
        max_errors = 100  # Maximum number of errors before stopping iteration
        
        # Create iterator with progress tracking
        try:
            iterator = iter(dataset)
            
            # Use leave=True to maintain only a single progress bar
            with tqdm(total=min(max_samples, len(dataset) if hasattr(dataset, '__len__') else max_samples), 
                     desc="Processing examples", leave=True, position=0) as pbar:
                while count < max_samples:
                    try:
                        example = next(iterator)
                        yield example
                        count += 1
                        pbar.update(1)
                        
                        # Periodically check memory usage and collect garbage if needed
                        # Don't log during normal operation to keep terminal clean
                        if count % self._memory_check_interval == 0:
                            self._check_resources(force=False)
                    except StopIteration:
                        break
                    except Exception as e:
                        errors += 1
                        # Log only every 10th error to reduce spam
                        if errors == 1 or errors % 10 == 0:
                            logger.warning(f"Error processing example ({errors} total errors): {str(e)}")
                        if errors >= max_errors:
                            logger.error(f"Too many errors ({errors}), stopping iteration")
                            break
        except Exception as e:
            logger.error(f"Failed to create iterator: {str(e)}")
            return
            yield from []
        
    def common_preprocessing(self, examples: Dict, prompt_field: str, completion_field: str,
                            lowercase: bool = False, streaming: bool = False) -> Dict:
        """Apply common preprocessing steps to all datasets."""
        
        # Input validation
        if not isinstance(examples, dict):
            logger.warning(f"Expected dictionary for examples, got {type(examples)}")
            return {"processed_text": [], "length": [], "duplicates_removed": 0}
            
        if prompt_field not in examples or completion_field not in examples:
            fields = list(examples.keys())
            logger.warning(f"Missing required fields. Expected {prompt_field} and {completion_field}, got {fields}")
            return {"processed_text": [], "length": [], "duplicates_removed": 0}
            
        # Get the raw data, ensuring we have lists
        raw_prompts = examples[prompt_field]
        raw_completions = examples[completion_field]
        
        if not isinstance(raw_prompts, list):
            raw_prompts = [raw_prompts]
        if not isinstance(raw_completions, list):
            raw_completions = [raw_completions]
            
        # Ensure equal lengths by truncating
        min_len = min(len(raw_prompts), len(raw_completions))
        if min_len == 0:
            return {"processed_text": [], "length": [], "duplicates_removed": 0}
            
        prompts = raw_prompts[:min_len]
        completions = raw_completions[:min_len]
        
        # Clean whitespace, strip leading/trailing spaces and newlines
        # Convert all inputs to strings, handle None values
        prompts = [re.sub(r'\s+', ' ', str(p or "")).strip() for p in prompts]
        completions = [re.sub(r'\s+', ' ', str(c or "")).strip() for c in completions]
        
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
        filtered_prompts = [prompts[i] for i in valid_indices]
        filtered_completions = [completions[i] for i in valid_indices]
        
        # Deduplicate based on prompt+completion pairs
        unique_pairs = {}
        deduplicated_texts = []
        
        for i, (text, prompt, completion) in enumerate(zip(filtered_texts, filtered_prompts, filtered_completions)):
            # Create a unique key from the prompt and completion
            pair_key = f"{prompt}__SEP__{completion}"
            if pair_key not in unique_pairs:
                unique_pairs[pair_key] = True
                deduplicated_texts.append(text)
        
        # Calculate duplicates removed but don't log it (will be aggregated later)
        duplicates_removed = len(filtered_texts) - len(deduplicated_texts)
        
        # For streaming mode, process in smaller batches to save memory
        if streaming:
            # Ensure batch_size is never zero to avoid range() error
            batch_size = max(1, min(len(deduplicated_texts), 32))  # At minimum 1, max 32
            all_lengths = []
            
            # Handle empty filtered_texts
            if not deduplicated_texts:
                return {"processed_text": [], "length": [], "duplicates_removed": duplicates_removed}
            
            for i in range(0, len(deduplicated_texts), batch_size):
                batch = deduplicated_texts[i:i+batch_size]
                # Tokenize and truncate to max length
                try:
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
                except Exception as e:
                    logger.warning(f"Tokenization error: {str(e)}. Skipping batch.")
                    # Add placeholder lengths for skipped batch to maintain indices
                    all_lengths.extend([0] * len(batch))
                
            # Ensure output has matching lengths
            if len(all_lengths) < len(deduplicated_texts):
                all_lengths.extend([0] * (len(deduplicated_texts) - len(all_lengths)))
                
            return {"processed_text": deduplicated_texts, "length": all_lengths, "duplicates_removed": duplicates_removed}
        else:
            # Handle empty filtered_texts for non-streaming mode too
            if not deduplicated_texts:
                return {"processed_text": [], "length": [], "duplicates_removed": duplicates_removed}
                
            # Tokenize and truncate to max length for non-streaming mode
            try:
                tokenized = self.tokenizer(
                    deduplicated_texts,
                    truncation=True,
                    max_length=self.max_length,
                    return_overflowing_tokens=False,
                    return_length=True
                )
                
                return {"processed_text": deduplicated_texts, "length": tokenized["length"], "duplicates_removed": duplicates_removed}
            except Exception as e:
                logger.error(f"Tokenization error in non-streaming mode: {str(e)}")
                # Return with dummy length values as fallback
                return {"processed_text": deduplicated_texts, "length": [0] * len(deduplicated_texts), "duplicates_removed": duplicates_removed}
    
    def process_codesearchnet(self, dataset: Union[Dataset, DatasetDict], 
                             streaming: bool = False, language: str = None) -> Union[Dataset, DatasetDict]:
        """Process CodeSearchNet dataset."""
        if language:
            logger.info(f"Processing CodeSearchNet dataset for {language}...")
            # Filter dataset by language if specified
            if hasattr(dataset, "column_names") and "language" in dataset.column_names:
                dataset = dataset.filter(lambda x: x["language"].lower() == language.lower())
                if isinstance(dataset, Dataset) and len(dataset) == 0:
                    logger.warning(f"No samples found for language {language}. Check if the language name is correct.")
                    return dataset
        else:
            logger.info("Processing CodeSearchNet dataset for all languages...")
        
        # For streaming mode, process examples one at a time to avoid issues
        if streaming:
            processed_examples = []
            
            # Process each example individually using our safe iterator
            for example in self._safe_dataset_iterator(dataset):
                try:
                    # Extract docstring and code
                    doc = example.get("func_documentation_string", "")
                    code = example.get("func_code_string", "")
                    lang = example.get("language", "")
                    
                    # Skip if missing required fields
                    if not doc or not code:
                        continue
                    
                    # Create a prompt with language information if available
                    if lang:
                        prompt = f"'''{doc}'''\n# Write a {lang} function"
                    else:
                        prompt = f"'''{doc}'''\n# Write a function"
                    
                    processed = self.common_preprocessing(
                        {"prompt": prompt, "completion": code},
                        "prompt", "completion",
                        streaming=True
                    )
                    processed_examples.append(processed)
                except Exception as e:
                    logger.warning(f"Error processing example: {e}")
                    continue
            
            return processed_examples
        
        # For non-streaming mode
        def process_sample(examples):
            # Handle different field names depending on the dataset structure
            if not isinstance(examples, dict):
                logger.warning(f"Expected dictionary, got {type(examples)}")
                return {"processed_text": [], "length": [], "duplicates_removed": 0}
                
            # Extract relevant fields
            docstrings = examples.get("func_documentation_string", [])
            code_snippets = examples.get("func_code_string", [])
            languages = examples.get("language", [])
            
            # Ensure all fields are lists
            if not isinstance(docstrings, list):
                docstrings = [docstrings]
            if not isinstance(code_snippets, list):
                code_snippets = [code_snippets]
            if not isinstance(languages, list):
                languages = [languages]
                
            # Ensure we have matching lengths
            min_len = min(len(docstrings), len(code_snippets))
            if min_len == 0:
                return {"processed_text": [], "length": [], "duplicates_removed": 0}
            
            # Truncate to matching length
            docstrings = docstrings[:min_len]
            code_snippets = code_snippets[:min_len]
            
            # Create language-specific prompts if language information is available
            if len(languages) >= min_len:
                languages = languages[:min_len]
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
            logger.warning(f"Error processing CodeSearchNet: {e}")
            try:
                # Try a simpler approach with smaller batch size
                return dataset.map(
                    process_sample,
                    batched=True,
                    batch_size=20  # Smaller batch size
                )
            except Exception as e2:
                logger.error(f"Failed second attempt to process CodeSearchNet: {e2}")
                logger.error(traceback.format_exc())
                return dataset  # Return original dataset if processing fails
    
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
    
    def _extract_fields(self, example: Dict, field_names_map: Dict[str, List[str]]) -> Dict[str, Any]:
        """Extract fields from example using multiple possible field names."""
        result = {}
        
        for target_field, possible_names in field_names_map.items():
            for name in possible_names:
                if name in example and example[name]:
                    result[target_field] = example[name]
                    break
                    
        return result
        
    def process_instruct_code(self, dataset: Union[Dataset, DatasetDict],
                             streaming: bool = False) -> Union[Dataset, DatasetDict]:
        """Process InstructCode dataset."""
        logger.info("Processing InstructCode dataset...")
        
        # Define field mappings for different dataset structures
        field_mappings = {
            "prompt": ["problem", "instruction", "instructions", "input", "query"],
            "completion": ["solution", "output", "response", "answer", "code"]
        }
        
        # For streaming mode, process examples one at a time to avoid issues
        if streaming:
            processed_examples = []
            
            # Process each example individually using our safe iterator
            for example in self._safe_dataset_iterator(dataset):
                try:
                    # Extract fields using our utility method
                    fields = self._extract_fields(example, field_mappings)
                    
                    # Skip if missing required fields
                    if "prompt" not in fields or "completion" not in fields:
                        # Try to handle conversation format if present
                        if "conversations" in example and isinstance(example["conversations"], list):
                            prompt = ""
                            completion = ""
                            
                            for i, turn in enumerate(example["conversations"]):
                                if isinstance(turn, dict):
                                    role = turn.get("role", turn.get("from", "")).lower()
                                    content = turn.get("content", turn.get("value", ""))
                                    
                                    if role in ["user", "human"] and not prompt:
                                        prompt = content
                                    elif role in ["assistant", "bot", "gpt"] and not completion:
                                        completion = content
                            
                            if prompt and completion:
                                fields = {"prompt": prompt, "completion": completion}
                            else:
                                continue
                        else:
                            continue
                    
                    processed = self.common_preprocessing(
                        {"prompt": fields["prompt"], "completion": fields["completion"]},
                        "prompt", "completion",
                        streaming=True
                    )
                    processed_examples.append(processed)
                    
                except Exception as e:
                    logger.warning(f"Error processing example: {e}")
                    continue
            
            # Log progress occasionally
            if len(processed_examples) % 1000 == 0 and processed_examples:
                logger.info(f"Processed {len(processed_examples)} examples from instruct_code dataset")
                
            if not processed_examples:
                logger.warning("No valid examples processed from instruct_code dataset")
                
            return processed_examples
        
        # For non-streaming mode, process in batches
        def process_sample(examples):
            # Handle different field structures
            if not isinstance(examples, dict):
                logger.warning(f"Expected dictionary, got {type(examples)}")
                return {"processed_text": [], "length": [], "duplicates_removed": 0}
            
            # Determine which fields to use for this batch
            batch_size = len(next(iter(examples.values()))) if examples else 0
            if batch_size == 0:
                return {"processed_text": [], "length": [], "duplicates_removed": 0}
                
            prompts = []
            completions = []
            
            # First try to extract fields for each example individually
            for i in range(batch_size):
                # Extract a single example from the batch
                example = {k: v[i] if isinstance(v, list) and i < len(v) else v for k, v in examples.items()}
                
                # Extract fields using our utility method
                fields = self._extract_fields(example, field_mappings)
                
                if "prompt" in fields and "completion" in fields:
                    prompts.append(fields["prompt"])
                    completions.append(fields["completion"])
                elif "conversations" in example and isinstance(example["conversations"], list):
                    # Try to handle conversation format
                    prompt = ""
                    completion = ""
                    
                    for turn in example["conversations"]:
                        if isinstance(turn, dict):
                            role = turn.get("role", turn.get("from", "")).lower()
                            content = turn.get("content", turn.get("value", ""))
                            
                            if role in ["user", "human"] and not prompt:
                                prompt = content
                            elif role in ["assistant", "bot", "gpt"] and not completion:
                                completion = content
                    
                    if prompt and completion:
                        prompts.append(prompt)
                        completions.append(completion)
            
            # Skip if no valid examples found
            if not prompts or not completions:
                return {"processed_text": [], "length": [], "duplicates_removed": 0}
            
            # Ensure all entries are strings
            valid_prompts = []
            valid_completions = []
            for i in range(min(len(prompts), len(completions))):
                if prompts[i] and completions[i] and isinstance(prompts[i], str) and isinstance(completions[i], str):
                    valid_prompts.append(prompts[i])
                    valid_completions.append(completions[i])
            
            if not valid_prompts or not valid_completions:
                return {"processed_text": [], "length": [], "duplicates_removed": 0}
            
            return self.common_preprocessing(
                {"prompt": valid_prompts, "completion": valid_completions},
                "prompt", "completion",
                streaming=streaming
            )
        
        try:
            # Try to get column names from dataset if available
            column_names = dataset.column_names if hasattr(dataset, "column_names") else None
            
            return dataset.map(
                process_sample,
                batched=True,
                remove_columns=column_names if not streaming else None,
                batch_size=100 if streaming else None
            )
        except Exception as e:
            logger.warning(f"Error processing InstructCode: {e}")
            logger.error(traceback.format_exc())
            try:
                # Try a simpler approach with smaller batch size
                return dataset.map(
                    process_sample,
                    batched=True,
                    batch_size=20  # Smaller batch size
                )
            except Exception as backup_error:
                logger.error(f"Backup processing also failed: {backup_error}")
                # Return an empty processed dataset to prevent pipeline failure
                return {"processed_text": [], "length": [], "duplicates_removed": 0}
    
    def process_mbpp(self, dataset: Union[Dataset, DatasetDict],
                     streaming: bool = False) -> Union[Dataset, DatasetDict]:
        """Process MBPP dataset."""
        logger.info("Processing MBPP dataset...")
        
        # Define field mappings for different dataset structures
        field_mappings = {
            "prompt": ["text", "prompt", "problem", "description", "task_id"],
            "completion": ["code", "solution", "canonical_solution", "answer"]
        }
        
        # For streaming mode, process one example at a time to avoid issues
        if streaming:
            processed_examples = []
            
            # Process each example individually using our safe iterator
            for example in self._safe_dataset_iterator(dataset):
                try:
                    # Extract fields using our utility method
                    fields = self._extract_fields(example, field_mappings)
                    
                    # Handle task_id special case
                    if "prompt" not in fields and "task_id" in example:
                        fields["prompt"] = f"Solve task {example['task_id']}"
                    
                    # Skip if missing required fields
                    if "prompt" not in fields or "completion" not in fields:
                        logger.warning(f"Skipping example, missing fields")
                        continue
                        
                    processed = self.common_preprocessing(
                        {"prompt": fields["prompt"], "completion": fields["completion"]},
                        "prompt", "completion",
                        streaming=True
                    )
                    processed_examples.append(processed)
                except Exception as e:
                    logger.warning(f"Error processing example: {e}")
                    continue
                
            if not processed_examples:
                logger.warning("No valid examples processed from MBPP dataset")
                
            return processed_examples
        else:
            # For non-streaming mode
            def process_sample(examples):
                # Ensure we have valid inputs
                if not isinstance(examples, dict):
                    logger.warning(f"Expected dictionary, got {type(examples)}")
                    return {"processed_text": [], "length": [], "duplicates_removed": 0}
                
                # Get batch size
                batch_size = len(next(iter(examples.values()))) if examples else 0
                if batch_size == 0:
                    return {"processed_text": [], "length": [], "duplicates_removed": 0}
                
                prompts = []
                completions = []
                
                # Process each example in the batch
                for i in range(batch_size):
                    # Extract a single example from the batch
                    example = {k: v[i] if isinstance(v, list) and i < len(v) else v for k, v in examples.items()}
                    
                    # Extract fields using our utility method
                    fields = self._extract_fields(example, field_mappings)
                    
                    # Handle task_id special case
                    if "prompt" not in fields and "task_id" in example:
                        fields["prompt"] = f"Solve task {example['task_id']}"
                    
                    # Add to batch if we have valid fields
                    if "prompt" in fields and "completion" in fields:
                        prompts.append(fields["prompt"])
                        completions.append(fields["completion"])
                
                # Skip if we couldn't extract any valid examples
                if not prompts or not completions:
                    return {"processed_text": [], "length": [], "duplicates_removed": 0}
                
                return self.common_preprocessing(
                    {"prompt": prompts, "completion": completions},
                    "prompt", "completion",
                    streaming=streaming
                )
            
            try:
                # Try to get column names from dataset if available
                column_names = dataset.column_names if hasattr(dataset, "column_names") else None
                
                return dataset.map(
                    process_sample,
                    batched=True,
                    remove_columns=column_names if not streaming else None,
                    batch_size=100 if streaming else None
                )
            except Exception as e:
                logger.warning(f"Error in batch processing mode: {e}")
                logger.error(traceback.format_exc())
                # Try with smaller batch size
                return dataset.map(
                    process_sample,
                    batched=True,
                    batch_size=20  # Smaller batch size
                )
            except Exception as e:
                logger.error(f"Error processing MBPP dataset: {e}")
                logger.error(traceback.format_exc())
                # Return a simple fallback processing
                return dataset.map(
                    lambda examples: self.common_preprocessing(
                        {"prompt": "Write Python code", "completion": str(examples)},
                        "prompt", "completion", 
                        streaming=streaming
                    ),
                    batched=True
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
                         streaming: bool = True, callback: Optional[Callable] = None, 
                         intermediate_save: bool = True, save_every: int = 10000,
                         natural_languages: List[str] = None, sampling_ratio: float = 1.0,
                         max_samples: int = None) -> Iterator[Dict]:
        """
        Process The Stack dataset.
        
        Args:
            dataset: The Stack dataset (or a similar dataset with code)
            language: Filter by programming language
            streaming: Whether to stream results
            callback: Optional callback function to process examples
            intermediate_save: Whether to save intermediate results
            save_every: How often to save intermediate results
            natural_languages: List of natural language codes to filter on (e.g., ['en', 'ar'])
            sampling_ratio: Ratio of examples to sample (0.0-1.0)
            max_samples: Maximum number of samples to process
            
        Returns:
            Iterator of processed examples
        """
        # Ensure we're processing a streaming dataset for memory efficiency
        if not streaming and not isinstance(dataset, Dataset):
            logger.warning("Converting dataset to streaming mode for memory efficiency")
            streaming = True
            
        # Set up resources monitoring
        resources = self._check_resources(force=True)
        
        # Explicitly mention available VRAM if detected
        if "gpu_max_mb" in resources:
            logger.info(f"GPU VRAM available: {resources['gpu_max_mb']:.1f} MB")
            logger.info(f"Current GPU usage: {resources.get('gpu_allocated_mb', 0):.1f} MB " + 
                       f"({resources.get('gpu_percent', 0):.1f}%)")
            
        # Log sampling settings if applied
        if sampling_ratio < 1.0:
            logger.info(f"Sampling {sampling_ratio*100:.1f}% of examples")
        if max_samples:
            logger.info(f"Processing up to {max_samples} examples")
        if natural_languages:
            logger.info(f"Filtering for natural languages: {', '.join(natural_languages)}")
        
        # Define format to include code + docstring
        def format_example(example):
            try:
                # Extract relevant fields
                content = example.get('content', '')
                language_name = example.get('lang', '').lower()
                
                # Apply programming language filtering
                # If a specific language was requested, filter for that
                if language and language_name != language.lower():
                    return None
                    
                # Filter for natural language in comments
                # Extract comments for language detection
                comment_pattern = r'(?:\/\/.*?$|\/\*[\s\S]*?\*\/|#.*?$|\'\'\'[\s\S]*?\'\'\'|"""[\s\S]*?""")'
                comments = re.findall(comment_pattern, content, re.MULTILINE)
                
                # Check natural language if we have comments and natural_languages is specified
                if comments and natural_languages:
                    comment_text = ' '.join(comments)
                    if not self.should_include_by_language(comment_text, natural_languages):
                        return None
                
                # Format the example
                formatted_text = f"{content}"
                
                # Check if we need to truncate
                if len(formatted_text) > self.max_length * 10:  # Rough estimate
                    logger.warning(f"Very long example ({len(formatted_text)} chars), truncating")
                    formatted_text = formatted_text[:self.max_length * 10]
                
                # Return the processed example
                return {
                    "processed_text": formatted_text,
                    "language": language_name
                }
                
            except Exception as e:
                logger.error(f"Error processing example: {str(e)}")
                return None
                
        # Process the dataset with efficient memory usage
        batches_processed = 0
        examples_processed = 0
        stall_counter = 0
        last_progress_time = time.time()
        last_example_count = 0
        
        # Initialize random number generator for sampling
        random.seed(42)  # For reproducibility
        
        # Batch processing logic
        for i, example in enumerate(dataset):
            try:
                # Apply sampling - skip examples based on sampling ratio
                if sampling_ratio < 1.0 and random.random() > sampling_ratio:
                    continue
                    
                # Process the example
                result = format_example(example)
                
                # Only yield valid results
                if result:
                    examples_processed += 1
                    
                    # Check if we reached max_samples
                    if max_samples and examples_processed > max_samples:
                        logger.info(f"Reached maximum sample count of {max_samples}. Stopping.")
                        break
                        
                    yield result
                    
                    # Call callback if provided
                    if callback:
                        callback(result)
                    
                    # Check for stalled processing
                    current_time = time.time()
                    if current_time - last_progress_time > 60:  # Check every minute
                        if examples_processed == last_example_count:
                            stall_counter += 1
                            logger.warning(f"Processing appears stalled: {stall_counter} minutes without progress")
                            
                            # If stalled for too long, clear memory
                            if stall_counter > 5:
                                logger.warning("Processing stalled for too long, cleaning up memory")
                                self._cleanup_memory()
                                stall_counter = 0
                        else:
                            # Reset stall counter if making progress
                            stall_counter = 0
                            
                        # Update progress tracking
                        last_progress_time = current_time
                        last_example_count = examples_processed
                        
                        # Log progress more frequently
                        logger.info(f"Processed {examples_processed} examples from The Stack")
                        
                        # Check resources periodically
                        resources = self._check_resources()
                
                # Periodic intermediate saving if enabled
                if intermediate_save and examples_processed > 0 and examples_processed % save_every == 0:
                    logger.info(f"Processed {examples_processed} examples so far")
                    
                    # Clear CUDA cache after processing batch
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
            except Exception as e:
                logger.error(f"Error processing example {i}: {str(e)}")
                logger.error(traceback.format_exc())
                continue
                
            # Do periodic cleanup
            if examples_processed % 5000 == 0:
                self._cleanup_memory()
                
            # Show more frequent progress updates
            if examples_processed % 1000 == 0:
                current_time = time.time()
                elapsed = current_time - last_progress_time
                rate = 1000 / max(elapsed, 1) if elapsed > 0 else 0
                logger.info(f"Processed {examples_processed} examples from The Stack (rate: {rate:.1f} examples/sec)")
                last_progress_time = current_time
                
        # Final progress update
        logger.info(f"Completed processing {examples_processed} examples from The Stack")
        
        # Final resource check
        self._check_resources(force=True)
        
        # Try to release VRAM if using PyTorch
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    
    def process_humaneval(self, dataset: Union[Dataset, DatasetDict],
                         streaming: bool = False) -> Union[Dataset, DatasetDict]:
        """Process HumanEval dataset."""
        logger.info("Processing HumanEval dataset...")
        
        # Define field mappings for different dataset structures
        field_mappings = {
            "prompt": ["prompt", "task_id", "question", "instruction"],
            "completion": ["canonical_solution", "solution", "test", "answer", "code"]
        }
        
        # For HumanEval, we need special handling for streaming mode
        if streaming:
            processed_examples = []
            
            # Process each example individually using our safe iterator
            for example in self._safe_dataset_iterator(dataset):
                try:
                    # Extract fields using our utility method
                    fields = self._extract_fields(example, field_mappings)
                    
                    # Handle case where we have neither prompt nor completion
                    if "prompt" not in fields or "completion" not in fields:
                        # For the case where each example is directly a string
                        if not isinstance(example, dict):
                            try:
                                fields = {
                                    "prompt": "Write a Python function",
                                    "completion": str(example)
                                }
                            except:
                                logger.warning(f"Could not process example: {type(example)}")
                                continue
                        
                        # Handle combined format with entry_point and test
                        elif "entry_point" in example and "test" in example:
                            entry_point = example.get("entry_point", "")
                            test_code = example.get("test", "")
                            
                            fields = {
                                "prompt": f"Implement the function {entry_point}",
                                "completion": test_code
                            }
                        else:
                            logger.warning("Skipping example with missing prompt or solution")
                            continue
                    
                    processed = self.common_preprocessing(
                        {"prompt": fields["prompt"], "completion": fields["completion"]},
                        "prompt", "completion",
                        streaming=True
                    )
                    processed_examples.append(processed)
                except Exception as e:
                    logger.warning(f"Error processing example: {e}")
                    continue
            
            if not processed_examples:
                logger.warning("No valid examples processed from HumanEval dataset")
            
            return processed_examples
        else:
            # For non-streaming mode, we can use map with batch processing
            try:
                def process_sample(examples):
                    # Ensure we have valid inputs
                    if not isinstance(examples, dict):
                        logger.warning(f"Expected dictionary, got {type(examples)}")
                        return {"processed_text": [], "length": [], "duplicates_removed": 0}
                    
                    # Get batch size
                    batch_size = len(next(iter(examples.values()))) if examples else 0
                    if batch_size == 0:
                        return {"processed_text": [], "length": [], "duplicates_removed": 0}
                    
                    prompts = []
                    completions = []
                    
                    # Process each example in the batch
                    for i in range(batch_size):
                        # Extract a single example from the batch
                        example = {k: v[i] if isinstance(v, list) and i < len(v) else v for k, v in examples.items()}
                        
                        # Extract fields using our utility method
                        fields = self._extract_fields(example, field_mappings)
                        
                        # Handle combined entry_point and test case
                        if "prompt" not in fields or "completion" not in fields:
                            if "entry_point" in example and "test" in example:
                                entry_point = example.get("entry_point", "")
                                test_code = example.get("test", "")
                                
                                fields = {
                                    "prompt": f"Implement the function {entry_point}",
                                    "completion": test_code
                                }
                        
                        # Add to batch if we have valid fields
                        if "prompt" in fields and "completion" in fields:
                            prompts.append(fields["prompt"])
                            completions.append(fields["completion"])
                    
                    # Skip if we couldn't extract any valid examples
                    if not prompts or not completions:
                        return {"processed_text": [], "length": [], "duplicates_removed": 0}
                    
                    return self.common_preprocessing(
                        {"prompt": prompts, "completion": completions},
                        "prompt", "completion",
                        streaming=streaming
                    )
                
                try:
                    # Try to get column names from dataset if available
                    column_names = dataset.column_names if hasattr(dataset, "column_names") else None
                    
                    return dataset.map(
                        process_sample,
                        batched=True,
                        remove_columns=column_names if not streaming else None,
                        batch_size=100 if streaming else None
                    )
                except Exception as e:
                    logger.warning(f"Error mapping HumanEval dataset: {e}")
                    logger.error(traceback.format_exc())
                    # Try with smaller batch size
                    return dataset.map(
                        process_sample,
                        batched=True,
                        batch_size=20
                    )
            except Exception as e:
                logger.error(f"Error processing HumanEval dataset: {e}")
                logger.error(traceback.format_exc())
                # Return a simple fallback processing
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
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        
        # Check for HF_TOKEN environment variable
        if os.environ.get("HF_TOKEN") is None:
            logger.warning("HF_TOKEN environment variable not set. Some gated datasets might be inaccessible.")
            logger.warning("You may need to set it using: export HF_TOKEN=your_huggingface_token")
        else:
            logger.info("Using Hugging Face token from environment for authentication")
         
        # Check system resources before starting
        resources = self._check_resources(force=True)
        if resources:
            # Explicitly mention available VRAM if detected
            if "gpu_max_mb" in resources:
                logger.info(f"GPU VRAM available: {resources['gpu_max_mb']:.1f} MB")
                logger.info(f"Current GPU usage: {resources.get('gpu_allocated_mb', 0):.1f} MB " +
                           f"({resources.get('gpu_percent', 0):.1f}%)")
            
            # Warning if high memory usage
            if resources.get("memory_percent", 0) > 90:
                logger.warning(f"System memory is very low ({resources.get('memory_percent')}%). Processing may fail.")
            
        # Process datasets with minimal logging and optimized for low memory usage
        total_datasets = len([name for name, cfg in dataset_config.items() if cfg.get("enabled", True)])
        logger.info(f"Processing {total_datasets} datasets with streaming mode and low memory footprint")
        
        for i, (dataset_name, config) in enumerate(dataset_config.items()):
            # Skip datasets that are explicitly disabled
            if config.get("enabled") is False:
                continue
                
            logger.info(f"[{i+1}/{total_datasets}] Processing dataset: {dataset_name}")
            
            # Check resources before processing each dataset
            self._check_resources()
            
            try:
                # Get streaming and caching options - force streaming for minimal memory usage
                streaming = True  # Always use streaming to minimize memory
                use_cache = config.get("use_cache", False)  # Prefer no caching to save RAM
                max_samples = config.get("max_samples", 10000)
                
                # Set cache directory if needed
                if not use_cache:
                    os.environ["HF_DATASETS_CACHE"] = "no"
                
                # Load the dataset with proper error handling and minimal caching
                try:
                    hf_token = os.environ.get("HF_TOKEN")
                    
                    load_params = {
                        "path": config["path"],
                        "name": config.get("name"),
                        "split": config.get("split"),
                        "streaming": streaming,
                        "trust_remote_code": True,
                    }
                    
                    # Add token parameter with backwards compatibility
                    try:
                        # Try new token parameter
                        load_params["token"] = hf_token
                        dataset = load_dataset(**load_params)
                    except TypeError:
                        # Fall back to older use_auth_token parameter
                        del load_params["token"]
                        load_params["use_auth_token"] = hf_token
                        dataset = load_dataset(**load_params)
                    
                    # Check resources after loading
                    self._check_resources()
                    
                except (ImportError, ModuleNotFoundError) as e:
                    logger.error(f"Missing dependency for loading dataset {dataset_name}: {str(e)}")
                    continue
                except Exception as e:
                    error_msg = str(e)
                    if "is a gated dataset" in error_msg:
                        logger.error(f"Dataset {config['path']} requires authentication. Make sure your HF_TOKEN has proper access rights.")
                    elif "doesn't exist on the Hub" in error_msg:
                        logger.error(f"Dataset {config['path']} was not found.")
                    else:
                        logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
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
                    # Just log the number of languages for brevity
                    logger.info(f"Processing for {len(config['languages'])} languages")
                    # For processors that have built-in language support, pass language=None to process all languages
                    if config['processor'] in ["the_stack", "codesearchnet"]:
                        processor_args["language"] = None
                
                # Add support for natural language filtering (for comments/docstrings)
                if "natural_languages" in config and isinstance(config["natural_languages"], list):
                    if config['processor'] == "the_stack":
                        processor_args["natural_languages"] = config["natural_languages"]
                
                # Add sampling ratio if specified
                if "sampling_ratio" in config and 0.0 < config["sampling_ratio"] <= 1.0:
                    if config['processor'] == "the_stack":
                        processor_args["sampling_ratio"] = config["sampling_ratio"]
                
                # Add max samples if specified
                if "max_samples" in config and config["max_samples"] > 0:
                    if config['processor'] == "the_stack":
                        processor_args["max_samples"] = config["max_samples"]
                
                # Process the dataset
                try:
                    # Add special handling for The Stack which can hang indefinitely
                    is_stack_dataset = "stack" in config["path"].lower()
                    if is_stack_dataset:
                        logger.info(f"Using timeouts for The Stack dataset processing")
                        
                        # Set a maximum processing time for The Stack
                        import threading
                        import queue
                        
                        # Create a queue for the result
                        result_queue = queue.Queue()
                        
                        # Define a function to process the dataset and put the result in the queue
                        def process_with_timeout():
                            try:
                                result = processor_func(dataset, **processor_args)
                                result_queue.put(("success", result))
                            except Exception as e:
                                result_queue.put(("error", e))
                        
                        # Start a thread to process the dataset
                        processing_thread = threading.Thread(target=process_with_timeout)
                        processing_thread.daemon = True
                        processing_thread.start()
                        
                        # Wait for the result with a timeout
                        max_wait_time = config.get("max_processing_time", 600)  # Default 10 minutes
                        try:
                            logger.info(f"Waiting up to {max_wait_time} seconds for The Stack processing to complete")
                            result_type, result_value = result_queue.get(timeout=max_wait_time)
                            
                            if result_type == "success":
                                processed_dataset = result_value
                                logger.info(f"The Stack dataset processing completed successfully")
                            else:
                                raise result_value
                        except queue.Empty:
                            logger.error(f"The Stack dataset processing timed out after {max_wait_time} seconds")
                            logger.info(f"Skipping The Stack dataset due to timeout")
                            continue
                    else:
                        # Normal processing for other datasets
                        processed_dataset = processor_func(dataset, **processor_args)
                    
                    self._check_resources()
                except Exception as e:
                    logger.error(f"Failed to process dataset {dataset_name}: {str(e)}")
                    continue
                
                # Create save directory
                save_path_for_dataset = os.path.join(save_path, f"{dataset_name}_processed")
                
                # Track total duplicates removed
                total_duplicates_removed = 0
                
                # For streaming datasets we need to materialize and convert to non-streaming format
                if streaming:
                    logger.info(f"Materializing streaming dataset (taking {max_samples} samples)...")
                    
                    from datasets import Dataset as HFDataset
                    
                    # Collect examples incrementally to minimize memory usage
                    collected_data = {"processed_text": [], "length": []}
                    count = 0
                    errors = 0
                    
                    # Handle different return types from processor
                    if isinstance(processed_dataset, list):
                        for i, example in enumerate(processed_dataset):
                            if count >= max_samples:
                                break
                            
                            try:
                                if isinstance(example, dict) and "processed_text" in example and "length" in example:
                                    # Ensure processed_text is a string and not a list or other type
                                    if isinstance(example["processed_text"], str):
                                        collected_data["processed_text"].append(example["processed_text"])
                                        collected_data["length"].append(example["length"])
                                        # Track duplicate counts
                                        if "duplicates_removed" in example:
                                            total_duplicates_removed += example["duplicates_removed"]
                                        count += 1
                                    elif isinstance(example["processed_text"], list) and len(example["processed_text"]) > 0:
                                        # If it's a list (batch processing returned), take the first item
                                        collected_data["processed_text"].append(example["processed_text"][0])
                                        collected_data["length"].append(example["length"][0] if isinstance(example["length"], list) else example["length"])
                                        count += 1
                                        
                                # Update progress with simple counter to avoid tqdm overhead
                                if i % 1000 == 0:
                                    logger.info(f"Processed {i} examples...")
                                    
                                # Check resources periodically
                                if i % self._memory_check_interval == 0:
                                    self._check_resources()
                                    # Free memory by saving intermediate results if collection is getting large
                                    if len(collected_data["processed_text"]) > 5000:
                                        try:
                                            # Save current batch and reset collections
                                            interim_dataset = HFDataset.from_dict(collected_data)
                                            interim_path = f"{save_path_for_dataset}_interim_{i}"
                                            interim_dataset.save_to_disk(interim_path)
                                            logger.info(f"Saved interim results ({len(interim_dataset)} examples)")
                                            
                                            # Store path for later merging
                                            processed_datasets.setdefault(dataset_name + "_parts", []).append(interim_path)
                                            
                                            # Reset collections to free memory
                                            collected_data = {"processed_text": [], "length": []}
                                            gc.collect()
                                        except Exception as e:
                                            # Continue collecting if saving fails
                                            logger.warning(f"Failed to save interim results: {e}")
                            except Exception as e:
                                errors += 1
                                if errors <= 3 or errors % 100 == 0:  # Log only a few errors
                                    logger.warning(f"Error collecting example: {e}")
                                if errors > 100:
                                    logger.error("Too many errors, stopping collection")
                                    break
                    else:
                        # Use safe iterator for regular streaming datasets
                        for example in self._safe_dataset_iterator(processed_dataset, max_samples):
                            try:
                                # Verify the example has the expected structure
                                if isinstance(example, dict) and "processed_text" in example and "length" in example:
                                    # Ensure the processed_text field is actually a string
                                    if example["processed_text"] and isinstance(example["processed_text"], str):
                                        collected_data["processed_text"].append(example["processed_text"])
                                        collected_data["length"].append(example["length"])
                                        # Track duplicate counts
                                        if "duplicates_removed" in example:
                                            total_duplicates_removed += example["duplicates_removed"]
                                        count += 1
                                    elif isinstance(example["processed_text"], list) and len(example["processed_text"]) > 0:
                                        # Handle case where processed_text is a list
                                        for j, text in enumerate(example["processed_text"]):
                                            if count >= max_samples:
                                                break
                                            if text and isinstance(text, str):
                                                collected_data["processed_text"].append(text)
                                                # Get corresponding length if available
                                                if isinstance(example["length"], list) and j < len(example["length"]):
                                                    collected_data["length"].append(example["length"][j])
                                                else:
                                                    collected_data["length"].append(0)  # Default length
                                                count += 1
                                                
                                    # Free memory by saving intermediate results if collection is getting large
                                    if len(collected_data["processed_text"]) > 5000:
                                        try:
                                            # Save current batch and reset collections
                                            interim_dataset = HFDataset.from_dict(collected_data)
                                            interim_path = f"{save_path_for_dataset}_interim_{count}"
                                            interim_dataset.save_to_disk(interim_path)
                                            logger.info(f"Saved interim results ({len(interim_dataset)} examples)")
                                            
                                            # Store path for later merging
                                            processed_datasets.setdefault(dataset_name + "_parts", []).append(interim_path)
                                            
                                            # Reset collections to free memory
                                            collected_data = {"processed_text": [], "length": []}
                                            gc.collect()
                                        except Exception as e:
                                            # Continue collecting if saving fails
                                            logger.warning(f"Failed to save interim results: {e}")
                            except Exception as e:
                                errors += 1
                                if errors <= 3 or errors % 100 == 0:  # Log only a few errors
                                    logger.warning(f"Error processing example: {e}")
                                if errors > 100:
                                    logger.error("Too many errors, stopping collection")
                                    break
                    
                    # Check resources before creating the final dataset
                    self._check_resources(force=True)
                    
                    # Verify we have data to save
                    if count == 0 and not (dataset_name + "_parts" in processed_datasets):
                        logger.error(f"No examples could be collected for {dataset_name}")
                        continue
                        
                    # Ensure both lists have the same length
                    min_len = min(len(collected_data["processed_text"]), len(collected_data["length"]))
                    collected_data["processed_text"] = collected_data["processed_text"][:min_len]
                    collected_data["length"] = collected_data["length"][:min_len]
                    
                    # Convert to HF Dataset and save
                    try:
                        # If we have interim datasets, we'll either merge them or just use them as is
                        interim_parts = processed_datasets.get(dataset_name + "_parts", [])
                        
                        if len(collected_data["processed_text"]) > 0:
                            # Handle any remaining data in memory
                            final_dataset = HFDataset.from_dict(collected_data)
                            
                            if interim_parts:
                                # We need to save this last part to merge with others
                                last_part_path = f"{save_path_for_dataset}_interim_final"
                                final_dataset.save_to_disk(last_part_path)
                                interim_parts.append(last_part_path)
                                logger.info(f"Saved final part with {len(final_dataset)} examples")
                            else:
                                # This is the only part, save it directly
                                final_dataset.save_to_disk(save_path_for_dataset)
                                logger.info(f"Saved dataset with {len(final_dataset)} examples")
                                processed_datasets[dataset_name] = final_dataset
                        
                        # If we have interim parts, create a reference dataset
                        if interim_parts:
                            # Just store the paths, actual merging would be done when loading
                            with open(f"{save_path_for_dataset}_parts.json", "w") as f:
                                json.dump({"parts": interim_parts}, f)
                            logger.info(f"Dataset {dataset_name} saved in {len(interim_parts)} parts")
                            processed_datasets[dataset_name] = {"parts": interim_parts}
                    except Exception as e:
                        logger.error(f"Failed to create or save dataset: {e}")
                    
                else:
                    # For non-streaming datasets
                    try:
                        # Verify we have a valid dataset to save
                        if isinstance(processed_dataset, Dataset) and len(processed_dataset) > 0:
                            # Check resources before saving
                            self._check_resources(force=True)
                            
                            # Save dataset
                            processed_dataset.save_to_disk(save_path_for_dataset)
                            logger.info(f"Saved processed dataset with {len(processed_dataset)} examples")
                            processed_datasets[dataset_name] = processed_dataset
                        else:
                            logger.error(f"Processed dataset for {dataset_name} is empty or invalid")
                    except Exception as e:
                        logger.error(f"Failed to save dataset {dataset_name}: {e}")
                
                # Log the total number of duplicates removed
                if total_duplicates_removed > 0:
                    logger.info(f"Removed {total_duplicates_removed} duplicate examples")
                
                # Force cleanup after each dataset is processed
                gc.collect()
                self._check_resources(force=True)
                
                # Try to release VRAM if using PyTorch
                try:
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            
            except Exception as e:
                logger.error(f"Error processing {dataset_name}: {str(e)}")
                
                # Try to recover from error and continue with next dataset
                gc.collect()
        
        # Final resource check
        self._check_resources(force=True)
        
        return processed_datasets 