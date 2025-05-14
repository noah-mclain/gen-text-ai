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
import contextlib
# Import text processor functions
try:
    from src.data.processors.text_processors import (
        gpteacher_processor,
        pile_processor,
        persona_chat_processor,
        writingprompts_processor,
        openassistant_processor
    )
    TEXT_PROCESSORS_AVAILABLE = True
except ImportError:
    TEXT_PROCESSORS_AVAILABLE = False
    
# For GPU memory management
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define a timer context manager
@contextlib.contextmanager
def timer(description: str):
    """Context manager for timing code execution."""
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        logging.info(f"{description} completed in {elapsed:.2f} seconds")

class DataPreprocessor:
    # Define popular programming languages - include those in CodeSearchNet plus popular ones
    POPULAR_PROGRAMMING_LANGUAGES = {
        'python', 'javascript', 'java', 'go', 'php', 'ruby',  # CodeSearchNet languages
        'c', 'cpp', 'csharp', 'c++', 'c#',                    # C family
        'typescript', 'rust', 'kotlin', 'swift', 'scala',     # Other popular langs
        'html', 'css', 'sql', 'bash', 'shell',                # Web/scripting
    }
    
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
        
        # Check if we can use GPU
        if TORCH_AVAILABLE and torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024):.2f} GB")
        
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
                              max_samples: Optional[int] = None) -> Iterator[Any]:
        """Safely iterate through a dataset with proper error handling and progress tracking."""
        count = 0
        errors = 0
        max_errors = 100  # Maximum number of errors before stopping iteration
        
        # Set max_samples to a reasonable default if None
        if max_samples is None:
            # If we can determine dataset size, use that with a cap
            if hasattr(dataset, '__len__'):
                max_samples = min(len(dataset), 100000)  # Cap at 100K samples to prevent memory issues
                logger.info(f"Processing up to {max_samples} examples (dataset size: {len(dataset)})")
            else:
                max_samples = 50000  # A more conservative default for unknown size datasets
                logger.info(f"Processing up to {max_samples} examples (dataset size unknown)")
            
            # Inform user they can override
            logger.info(f"To process more or fewer examples, specify max_samples in your configuration")
        
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
                        if count % self._memory_check_interval == 0:
                            resources = self._check_resources(force=False)
                            
                            # Stop processing if memory usage is critically high (90%+)
                            if resources.get('memory_percent', 0) > 90 or resources.get('gpu_percent', 0) > 90:
                                logger.warning(f"Memory usage too high (RAM: {resources.get('memory_percent')}%, "
                                             f"GPU: {resources.get('gpu_percent')}%), stopping early at {count} examples")
                                break
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
            yield from []
            
        # Log final stats
        logger.info(f"Processed {count} examples with {errors} errors")
        
    def _extract_fields(self, example: Dict, field_mappings: Dict[str, List[str]]) -> Dict[str, Any]:
        """Extract fields from an example using various possible field names.
        
        Args:
            example: Dictionary containing example data
            field_mappings: Dictionary mapping field types to potential field names
            
        Returns:
            Dictionary with extracted fields
        """
        result = {}
        
        # Handle different possible field names
        for field_type, field_names in field_mappings.items():
            # Try each possible field name
            for name in field_names:
                if name in example and example[name]:
                    result[field_type] = example[name]
                    break
                    
        return result
    
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
                             streaming: bool = False, language: str = None, 
                             max_samples: Optional[int] = None) -> Union[Dataset, DatasetDict]:
        """
        Process CodeSearchNet dataset.
        
        Args:
            dataset: The CodeSearchNet dataset
            streaming: Whether to use streaming mode for memory efficiency
            language: Specific language to filter by, or None to process all languages
            max_samples: Maximum number of samples to process
            
        Returns:
            Processed dataset or list of processed examples
        """
        # Display clear logging information
        if language:
            logger.info(f"Processing CodeSearchNet dataset for {language}...")
            # Filter dataset by language if specified
            try:
                if hasattr(dataset, "column_names") and "language" in dataset.column_names:
                    dataset = dataset.filter(lambda x: x["language"].lower() == language.lower())
                    if isinstance(dataset, Dataset) and len(dataset) == 0:
                        logger.warning(f"No samples found for language {language}. Check if the language name is correct.")
                        return dataset
                    logger.info(f"Filtered to {len(dataset) if hasattr(dataset, '__len__') else 'unknown'} examples for language: {language}")
            except Exception as e:
                logger.warning(f"Error filtering by language {language}: {e}")
        else:
            logger.info("Processing CodeSearchNet dataset for all languages...")
            
        # For streaming mode, process examples one at a time to avoid memory issues
        if streaming:
            processed_examples = []
            error_counts = {lang: 0 for lang in self.POPULAR_PROGRAMMING_LANGUAGES}
            success_counts = {lang: 0 for lang in self.POPULAR_PROGRAMMING_LANGUAGES}
            
            # Process each example individually using our safe iterator
            for example in self._safe_dataset_iterator(dataset, max_samples=max_samples):
                try:
                    # Extract docstring and code
                    doc = example.get("func_documentation_string", "")
                    code = example.get("func_code_string", "")
                    lang = example.get("language", "").lower()
                    
                    # Skip if missing required fields
                    if not doc or not code:
                        continue
                    
                    # Skip if language doesn't match requested language (when specified)
                    if language and lang.lower() != language.lower():
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
                    
                    # Add language tag to track the source language
                    if "processed_text" in processed:
                        processed["language"] = lang
                        
                    processed_examples.append(processed)
                    
                    # Track successful examples by language
                    if lang in success_counts:
                        success_counts[lang] += 1
                    
                    # Periodically log progress
                    total_processed = sum(success_counts.values())
                    if total_processed % 1000 == 0:
                        logger.info(f"Processed {total_processed} CodeSearchNet examples so far")
                        
                except Exception as e:
                    # Track errors by language
                    if "language" in example and example["language"] in error_counts:
                        error_counts[example["language"]] += 1
                    logger.debug(f"Error processing example: {e}")
                    continue
            
            # Log statistics at the end
            logger.info(f"Completed processing {len(processed_examples)} CodeSearchNet examples")
            
            # Log language statistics
            processed_langs = {lang: count for lang, count in success_counts.items() if count > 0}
            if processed_langs:
                logger.info(f"Language distribution: {processed_langs}")
            
            return processed_examples
        
        # For non-streaming mode - more efficient batch processing
        def process_sample(examples):
            # Handle different field names depending on the dataset structure
            if not isinstance(examples, dict):
                logger.warning(f"Expected dictionary, got {type(examples)}")
                return {"processed_text": [], "length": [], "duplicates_removed": 0, "language": []}
                
            try:
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
                    return {"processed_text": [], "length": [], "duplicates_removed": 0, "language": []}
                
                # Truncate to matching length
                docstrings = docstrings[:min_len]
                code_snippets = code_snippets[:min_len]
                
                # Additional filtering if a specific language is requested
                if language and len(languages) >= min_len:
                    # Create masks for examples that match the requested language
                    valid_indices = [i for i, lang in enumerate(languages[:min_len]) 
                                   if lang.lower() == language.lower()]
                    
                    # Filter based on language
                    if valid_indices:
                        docstrings = [docstrings[i] for i in valid_indices]
                        code_snippets = [code_snippets[i] for i in valid_indices]
                        languages = [languages[i] for i in valid_indices]
                    else:
                        # No matching languages in this batch
                        return {"processed_text": [], "length": [], "duplicates_removed": 0, "language": []}
                else:
                    # No specific language requested, just truncate languages list
                    languages = languages[:min_len]
                
                # Create language-specific prompts if language information is available
                if len(languages) >= len(docstrings):
                    prompts = [f"'''{doc}'''\n# Write a {lang} function" 
                              for doc, lang in zip(docstrings, languages)]
                else:
                    # Default to generic prompt if no language info
                    prompts = [f"'''{doc}'''\n# Write a function" for doc in docstrings]
                    # Fill in languages with empty values if needed
                    languages = languages + [""] * (len(docstrings) - len(languages))
                
                # Process the examples
                result = self.common_preprocessing(
                    {"prompt": prompts, "completion": code_snippets},
                    "prompt", "completion",
                    streaming=streaming
                )
                
                # Add languages to the result
                if "processed_text" in result and len(result["processed_text"]) > 0:
                    # Ensure languages list matches processed text length
                    result_len = len(result["processed_text"])
                    result["language"] = languages[:result_len] if len(languages) >= result_len else languages + [""] * (result_len - len(languages))
                else:
                    result["language"] = []
                
                return result
            except Exception as e:
                logger.warning(f"Error in batch processing: {e}")
                return {"processed_text": [], "length": [], "duplicates_removed": 0, "language": []}
        
        # Main batch processing with fallbacks
        try:
            logger.info("Processing CodeSearchNet with batch processing")
            
            # First try: Standard batch processing
            processed = dataset.map(
                process_sample,
                batched=True, 
                remove_columns=dataset.column_names if not streaming else None,
                batch_size=100 if streaming else None,
                num_proc=1  # Safer processing option
            )
            
            # Log processing statistics
            if isinstance(processed, Dataset) and hasattr(processed, "__len__"):
                logger.info(f"Successfully processed {len(processed)} CodeSearchNet examples")
                # Count by language if available
                if "language" in processed.column_names:
                    # Sample to get language distribution (avoid scanning full dataset)
                    sample_size = min(1000, len(processed))
                    languages = processed.select(range(sample_size))["language"]
                    lang_counts = {}
                    for lang in languages:
                        if lang:  # Skip empty language values
                            lang_counts[lang] = lang_counts.get(lang, 0) + 1
                    if lang_counts:
                        logger.info(f"Language distribution (sample): {lang_counts}")
            
            return processed
            
        except Exception as e:
            logger.warning(f"Error in standard batch processing: {e}")
            
            try:
                logger.info("Trying with smaller batch size...")
                # Second try: Smaller batch size
                processed = dataset.map(
                    process_sample,
                    batched=True,
                    batch_size=20  # Smaller batch size
                )
                
                if isinstance(processed, Dataset) and hasattr(processed, "__len__"):
                    logger.info(f"Successfully processed {len(processed)} examples with smaller batch size")
                
                return processed
                
            except Exception as e2:
                logger.error(f"Failed second attempt to process CodeSearchNet: {e2}")
                logger.error(traceback.format_exc())
                
                # Third try: Item-by-item processing as absolute fallback
                try:
                    logger.info("Falling back to item-by-item processing...")
                    processed_examples = []
                    
                    # Process a limited number to avoid running out of memory
                    max_fallback = 5000
                    count = 0
                    
                    for example in dataset:
                        if count >= max_fallback:
                            break
                            
                        try:
                            # Extract docstring and code
                            doc = example.get("func_documentation_string", "")
                            code = example.get("func_code_string", "")
                            lang = example.get("language", "")
                            
                            # Skip if missing required fields
                            if not doc or not code:
                                continue
                            
                            # Skip if language doesn't match the requested language
                            if language and lang.lower() != language.lower():
                                continue
                            
                            # Create a prompt with language information if available
                            if lang:
                                prompt = f"'''{doc}'''\n# Write a {lang} function"
                            else:
                                prompt = f"'''{doc}'''\n# Write a function"
                            
                            # Direct formatting
                            processed_text = f"User: {prompt}\nAssistant: {code}"
                            processed_examples.append({
                                "processed_text": processed_text,
                                "length": len(processed_text),
                                "language": lang
                            })
                            
                            count += 1
                        except Exception:
                            continue
                    
                    logger.info(f"Fallback processing completed with {len(processed_examples)} examples")
                    # Create a Dataset from the processed examples
                    from datasets import Dataset as HFDataset
                    
                    # Extract features
                    features = {
                        "processed_text": [ex["processed_text"] for ex in processed_examples],
                        "length": [ex["length"] for ex in processed_examples],
                        "language": [ex["language"] for ex in processed_examples],
                    }
                    
                    return HFDataset.from_dict(features)
                    
                except Exception as e3:
                    logger.error(f"All processing attempts failed: {e3}")
                    
                return dataset  # Return original dataset if all processing attempts fail
    
    def process_code_alpaca(self, dataset: Union[Dataset, DatasetDict],
                           streaming: bool = False, max_samples: Optional[int] = None) -> Union[Dataset, DatasetDict]:
        """Process CodeAlpaca-20K dataset."""
        logger.info("Processing CodeAlpaca dataset...")
        
        # For streaming mode, process one example at a time to avoid issues
        if streaming:
            processed_examples = []
            
            # Process each example individually using our safe iterator
            for i, example in enumerate(self._safe_dataset_iterator(dataset, max_samples=max_samples)):
                if i >= 10000 and max_samples is None:  # Limit to prevent processing too many examples unless overridden
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
    
    def process_mbpp(self, dataset: Union[Dataset, DatasetDict],
                     streaming: bool = False, max_samples: Optional[int] = None) -> Union[Dataset, DatasetDict]:
        """Process MBPP dataset."""
        logger.info("Processing MBPP dataset...")
        
        # For streaming mode, process one example at a time to avoid issues
        if streaming:
            processed_examples = []
            total_examples = 0
            
            # Process each example individually using our safe iterator
            for example in self._safe_dataset_iterator(dataset, max_samples=max_samples):
                total_examples += 1
                try:
                    # Extract fields
                    prompt = example.get("text", "")
                    completion = example.get("code", "")
                    
                    # Skip if missing required fields
                    if not prompt or not completion:
                        continue
                        
                    processed = self.common_preprocessing(
                        {"prompt": prompt, "completion": completion},
                        "prompt", "completion",
                        streaming=streaming
                    )
                    
                    # Validate processed results
                    if "processed_text" in processed and processed["processed_text"]:
                        processed_examples.append(processed)
                        
                        # Log progress at regular intervals
                        if len(processed_examples) % 100 == 0:
                            logger.info(f"Processed {len(processed_examples)} valid MBPP examples (from {total_examples} total)")
                except Exception as e:
                    logger.warning(f"Error processing example: {str(e)}")
                    continue
                
            # Log final stats
            logger.info(f"Completed processing {len(processed_examples)} MBPP examples from {total_examples} total examples")
            return processed_examples
        else:
            # For non-streaming mode
            try:
                return dataset.map(
                    lambda examples: self.common_preprocessing(
                        {"prompt": examples["text"], "completion": examples["code"]},
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
                        {"prompt": examples.get("text", ""), "completion": examples.get("code", "")},
                        "prompt", "completion",
                        streaming=streaming
                    ),
                    batched=True,
                    batch_size=100 if streaming else None
                )
    
    def process_ds1000(self, dataset: Union[Dataset, DatasetDict],
                       streaming: bool = False, max_samples: Optional[int] = None) -> Union[Dataset, DatasetDict]:
        """Process DS-1000 dataset."""
        logger.info("Processing DS-1000 dataset...")
        
        # DS-1000 is a benchmark dataset, not a dataset to be loaded from HF Hub
        # It should be processed from local files
        try:
            # Look for DS-1000 in expected directories
            ds1000_dirs = [
                "./DS-1000", 
                "../DS-1000", 
                "./data/DS-1000", 
                "./data/raw/DS-1000"
            ]
            
            ds1000_path = None
            for directory in ds1000_dirs:
                if os.path.exists(directory) and os.path.isdir(directory):
                    ds1000_path = directory
                    break
                    
            if not ds1000_path:
                logger.error("DS-1000 benchmark directory not found. Please clone the repository from GitHub.")
                logger.error("Run: git clone https://github.com/microsoft/DS-1000.git")
                return []
                
            logger.info(f"Found DS-1000 benchmark at {ds1000_path}")
            
            # Process the benchmark by reading problem files
            processed_examples = []
            
            # Iterate through library directories
            for lib_dir in os.listdir(ds1000_path):
                lib_path = os.path.join(ds1000_path, lib_dir)
                
                # Skip if not a directory or is a hidden directory
                if not os.path.isdir(lib_path) or lib_dir.startswith('.'):
                    continue
                    
                logger.info(f"Processing library: {lib_dir}")
                
                # Iterate through problem directories
                problem_dirs = [d for d in os.listdir(lib_path) if os.path.isdir(os.path.join(lib_path, d))]
                
                for problem_dir in problem_dirs:
                    problem_path = os.path.join(lib_path, problem_dir)
                    
                    # Look for prompt and solution files
                    prompt_file = os.path.join(problem_path, "prompt.txt")
                    solution_file = os.path.join(problem_path, "solutions/reference.py")
                    
                    if not os.path.exists(prompt_file) or not os.path.exists(solution_file):
                        continue
                        
                    # Read prompt and solution
                    try:
                        with open(prompt_file, 'r') as f:
                            prompt = f.read()
                            
                        with open(solution_file, 'r') as f:
                            solution = f.read()
                            
                        # Create formatted prompt
                        formatted_prompt = f"# Library: {lib_dir}\n# Problem: {problem_dir}\n\n{prompt}"
                        
                        # Process the example
                        processed = self.common_preprocessing(
                            {"prompt": formatted_prompt, "completion": solution},
                            "prompt", "completion",
                            streaming=streaming
                        )
                        
                        if processed and "processed_text" in processed and processed["processed_text"]:
                            processed_examples.append(processed)
                    except Exception as e:
                        logger.warning(f"Error processing problem {lib_dir}/{problem_dir}: {e}")
                        continue
                        
                    # Respect max_samples if set
                    if max_samples and len(processed_examples) >= max_samples:
                        logger.info(f"Reached max_samples limit of {max_samples}")
                        break
                        
                if max_samples and len(processed_examples) >= max_samples:
                    break
                    
            logger.info(f"Completed processing {len(processed_examples)} DS-1000 examples")
            return processed_examples
            
        except Exception as e:
            logger.error(f"Error processing DS-1000 benchmark: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def process_humaneval(self, dataset: Union[Dataset, DatasetDict],
                         streaming: bool = False, max_samples: Optional[int] = None) -> Union[Dataset, DatasetDict]:
        """Process HumanEval dataset."""
        logger.info("Processing HumanEval dataset...")
        
        # For streaming mode, process one example at a time to avoid issues
        if streaming:
            processed_examples = []
            total_examples = 0
            
            # Process each example individually using our safe iterator
            for example in self._safe_dataset_iterator(dataset, max_samples=max_samples):
                total_examples += 1
                try:
                    # Extract fields
                    prompt = example.get("prompt", "")
                    completion = example.get("canonical_solution", "")
                    
                    # Skip if missing required fields
                    if not prompt or not completion:
                        continue
                        
                    processed = self.common_preprocessing(
                        {"prompt": prompt, "completion": completion},
                        "prompt", "completion",
                        streaming=streaming
                    )
                    
                    # Validate processed results
                    if "processed_text" in processed and processed["processed_text"]:
                        processed_examples.append(processed)
                        
                        # Log progress at regular intervals
                        if len(processed_examples) % 50 == 0:
                            logger.info(f"Processed {len(processed_examples)} valid HumanEval examples (from {total_examples} total)")
                except Exception as e:
                    logger.warning(f"Error processing example: {str(e)}")
                    continue
                
            # Log final stats
            logger.info(f"Completed processing {len(processed_examples)} HumanEval examples from {total_examples} total examples")
            return processed_examples
        else:
            # For non-streaming mode
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
                logger.warning(f"Error in batch processing mode: {e}")
                # Fallback to simple processing with expected field names
                return dataset.map(
                    lambda examples: self.common_preprocessing(
                        {"prompt": examples.get("prompt", ""), "completion": examples.get("canonical_solution", "")},
                        "prompt", "completion",
                        streaming=streaming
                    ),
                    batched=True,
                    batch_size=100 if streaming else None
                )
    
    def process_the_stack(self, dataset: Union[Dataset, DatasetDict],
                         streaming: bool = False, max_samples: Optional[int] = None) -> Union[Dataset, DatasetDict]:
        """Process The Stack dataset."""
        logger.info("Processing The Stack dataset...")
        
        # For streaming mode, process one example at a time to avoid issues
        if streaming:
            processed_examples = []
            
            # Process each example individually using our safe iterator
            for i, example in enumerate(self._safe_dataset_iterator(dataset, max_samples=max_samples)):
                try:
                    # Extract fields
                    prompt = example.get("rewritten_file", "")
                    completion = example.get("rewritten_file", "")
                    
                    # Skip if missing required fields
                    if not prompt or not completion:
                        continue
                        
                    processed = self.common_preprocessing(
                        {"prompt": prompt, "completion": completion},
                        "prompt", "completion",
                        streaming=streaming
                    )
                    
                    # Validate processed results
                    if "processed_text" in processed and processed["processed_text"]:
                        processed_examples.append(processed)
                        
                        # Log progress at regular intervals
                        if (i + 1) % 1000 == 0:
                            logger.info(f"Processed {i + 1} The Stack examples")
                except Exception as e:
                    logger.warning(f"Error processing example {i}: {e}")
                    continue
                
            # Log final stats
            logger.info(f"Completed processing {len(processed_examples)} The Stack examples")
            return processed_examples
        else:
            # For non-streaming mode
            try:
                return dataset.map(
                    lambda examples: self.common_preprocessing(
                        {"prompt": examples["rewritten_file"], "completion": examples["rewritten_file"]},
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
                        {"prompt": examples.get("rewritten_file", ""), "completion": examples.get("rewritten_file", "")},
                        "prompt", "completion",
                        streaming=streaming
                    ),
                    batched=True,
                    batch_size=100 if streaming else None
                )
    
    def process_codeparrot(self, dataset: Union[Dataset, DatasetDict],
                          streaming: bool = False, max_samples: Optional[int] = None) -> Union[Dataset, DatasetDict]:
        """Process CodeParrot dataset."""
        logger.info("Processing CodeParrot dataset...")
        
        # CodeParrot structure can vary - try to detect the correct fields
        try:
            # Sample the first example to determine field names
            sample = next(iter(dataset))
            
            # Detect available fields
            content_field = None
            for field_name in ["content", "code", "source", "text"]:
                if field_name in sample and sample[field_name]:
                    content_field = field_name
                    break
                    
            if not content_field:
                logger.warning("Could not identify content field in CodeParrot dataset")
                logger.info(f"Available fields: {list(sample.keys())}")
                # Try a fallback field if nothing found
                content_field = list(sample.keys())[0] if sample else "content"
                
            logger.info(f"Using '{content_field}' as the content field for CodeParrot")
        except Exception as e:
            logger.warning(f"Error determining fields: {e}")
            # Default to 'content' if we couldn't determine
            content_field = "content"
        
        # For streaming mode, process one example at a time to avoid issues
        if streaming:
            processed_examples = []
            total_examples = 0
            
            # Process each example individually using our safe iterator
            for example in self._safe_dataset_iterator(dataset, max_samples=max_samples):
                total_examples += 1
                try:
                    # Check if the example is a dictionary
                    if not isinstance(example, dict):
                        continue
                        
                    # Extract content - try multiple field names
                    content = None
                    for field in [content_field, "content", "code", "source", "text"]:
                        if field in example and example[field]:
                            content = example[field]
                            break
                    
                    # Skip if missing content
                    if not content:
                        continue
                        
                    # For CodeParrot, we use the content both as prompt and completion
                    # This is common for language modeling datasets
                    prompt = "# Complete the following code"
                    completion = content
                    
                    processed = self.common_preprocessing(
                        {"prompt": prompt, "completion": completion},
                        "prompt", "completion",
                        streaming=streaming
                    )
                    
                    # Validate processed results
                    if "processed_text" in processed and processed["processed_text"]:
                        processed_examples.append(processed)
                        
                        # Log progress at regular intervals
                        if len(processed_examples) % 1000 == 0:
                            logger.info(f"Processed {len(processed_examples)} valid CodeParrot examples (from {total_examples} total)")
                except Exception as e:
                    logger.warning(f"Error processing example: {str(e)}")
                    continue
                
            # Log final stats
            logger.info(f"Completed processing {len(processed_examples)} CodeParrot examples from {total_examples} total examples")
            return processed_examples
        else:
            # For non-streaming mode, batch process
            try:
                def process_batch(examples):
                    batch_size = len(next(iter(examples.values()))) if examples else 0
                    
                    if batch_size == 0:
                        return {"processed_text": [], "length": [], "duplicates_removed": 0}
                        
                    # Extract contents from the batch
                    contents = []
                    for i in range(batch_size):
                        content = None
                        for field in [content_field, "content", "code", "source", "text"]:
                            if field in examples and i < len(examples[field]) and examples[field][i]:
                                content = examples[field][i]
                                break
                                
                        if content:
                            contents.append(content)
                            
                    # Skip if no valid contents
                    if not contents:
                        return {"processed_text": [], "length": [], "duplicates_removed": 0}
                        
                    # Use content as completion with generic prompt
                    prompts = ["# Complete the following code"] * len(contents)
                    
                    return self.common_preprocessing(
                        {"prompt": prompts, "completion": contents},
                        "prompt", "completion",
                        streaming=streaming
                    )
                    
                return dataset.map(
                    process_batch,
                    batched=True,
                    remove_columns=dataset.column_names if not streaming else None,
                    batch_size=50 if streaming else None
                )
            except Exception as e:
                logger.warning(f"Error in batch processing mode: {e}")
                # Return empty result as fallback
                logger.error(f"Failed to process CodeParrot: {e}")
                return []
    
    def process_instruct_code(self, dataset: Union[Dataset, DatasetDict],
                              streaming: bool = False, max_samples: Optional[int] = None) -> Union[Dataset, DatasetDict]:
        """
        Process InstructCode dataset.
        
        This processor handles instruction datasets like:
        - Magicoder-OSS-Instruct
        - Other instruction-tuning datasets with code
        """
        logger.info("Processing InstructCode dataset...")
        
        # Define field mappings for different dataset structures
        field_mappings = {
            "prompt": ["problem", "instruction", "instructions", "input", "query", "question", "text"],
            "completion": ["solution", "output", "response", "answer", "code", "content"]
        }
        
        # For streaming mode, process examples one at a time to avoid issues
        if streaming:
            processed_examples = []
            
            # Process each example individually using our safe iterator
            for example in self._safe_dataset_iterator(dataset, max_samples=max_samples):
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
    
    def load_and_process_all_datasets(self, dataset_config: Dict, save_path: str) -> Dict[str, Dataset]:
        """Load and process all datasets according to configuration."""
        processed_datasets = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        total_datasets = len(dataset_config)
        logger.info(f"Processing {total_datasets} datasets")
        
        # Track datasets for progress reporting
        dataset_count = 0
        successful_count = 0
        failed_count = 0
        
        for dataset_name, config in dataset_config.items():
            dataset_count += 1
            logger.info(f"[{dataset_count}/{total_datasets}] Processing dataset: {dataset_name}")
            
            # Report memory usage before processing each dataset
            memory_info = self._check_resources(force=True)
            logger.info(f"Memory usage: {memory_info.get('memory_mb', 0):.1f} MB ({memory_info.get('memory_percent', 0):.1f}% of system memory)")
            if 'gpu_allocated_mb' in memory_info:
                logger.info(f"GPU memory: {memory_info.get('gpu_allocated_mb', 0):.1f} MB ({memory_info.get('gpu_percent', 0):.1f}% of GPU memory)")
            
            # Skip dataset if system resources are too constrained
            if memory_info.get('memory_percent', 0) > 85 or memory_info.get('gpu_percent', 0) > 85:
                logger.warning(f"System resources too low to process dataset {dataset_name}. Skipping.")
                failed_count += 1
                continue
                
            try:
                # Extract processor name and dataset path
                processor_name = config.get("processor", dataset_name)
                dataset_path = config.get("path", dataset_name)
                split = config.get("split", "train")
                
                # Force streaming mode for memory efficiency, unless explicitly disabled
                streaming = config.get("streaming", True)  # Default to True
                
                # Get max_samples from config, use default if not specified
                max_samples = config.get("max_samples", None)
                if max_samples is None:
                    # For streaming datasets, use a reasonable default unless specified
                    if streaming:
                        max_samples = 50000  # Default to 50K for streaming datasets
                        logger.info(f"Using default max_samples={max_samples} for streaming dataset {dataset_name}")
                        logger.info(f"To change this limit, specify max_samples in your dataset config")
                
                # Check if we should skip processing if it already exists
                output_path = os.path.join(save_path, f"{dataset_name}_processed")
                if os.path.exists(output_path) and not config.get("force_reprocess", False):
                    logger.info(f"Dataset '{dataset_name}' already processed at {output_path}. Skipping.")
                    try:
                        # Try to load the dataset to verify it's valid
                        from datasets import load_from_disk
                        processed_datasets[dataset_name] = load_from_disk(output_path)
                        successful_count += 1
                        continue
                    except Exception as e:
                        logger.warning(f"Error loading existing dataset: {e}. Will reprocess.")
                
                # Map some dataset names to their appropriate processors if needed
                # This ensures we handle datasets that might have different names than their processors
                processor_mapping = {
                    "magicoder": "instruct_code",   # Magicoder is an instruction-tuning dataset
                    "codealpaca": "code_alpaca",    # Normalize name
                    "mbpp": "mbpp",                 # Already matches
                    "humaneval": "humaneval",       # Already matches
                    "codesearchnet": "codesearchnet", # Already matches
                    "codeparrot": "codeparrot",     # Already matches
                }
                
                # Apply mapping if available
                if dataset_name.lower() in processor_mapping:
                    processor_name = processor_mapping[dataset_name.lower()]
                
                # For text datasets, use the specialized processors that handle streaming and saving
                is_text_dataset = processor_name in ['openassistant', 'gpteacher', 'pile', 'persona_chat', 'writingprompts']
                
                if is_text_dataset and TEXT_PROCESSORS_AVAILABLE:
                    # Get the appropriate processor function for text datasets
                    if processor_name == 'openassistant':
                        processor_func = openassistant_processor
                    elif processor_name == 'gpteacher':
                        processor_func = gpteacher_processor
                    elif processor_name == 'pile':
                        processor_func = pile_processor
                    elif processor_name == 'persona_chat':
                        processor_func = persona_chat_processor
                    elif processor_name == 'writingprompts':
                        processor_func = writingprompts_processor
                    else:
                        logger.warning(f"Unknown text processor: {processor_name}")
                        failed_count += 1
                        continue
                
                    # Process using the text processor
                    try:
                        # Check if we have the necessary dependencies before processing
                        if processor_name == 'pile':
                            try:
                                import zstandard
                                logger.info("zstandard package is available for Pile dataset")
                            except ImportError:
                                logger.error("Missing required dependency 'zstandard' for Pile dataset")
                                logger.error("Install with: pip install zstandard")
                                failed_count += 1
                                continue
                        
                        # Process the dataset with a large timeout for larger datasets
                        with tqdm(total=1, desc=f"Processing {dataset_name}", leave=True) as pbar:
                            # Process the dataset
                            processed_dataset = processor_func(
                                dataset_path=dataset_path,
                                output_dir=save_path,
                                split=split,
                                streaming=streaming,
                                use_cache=config.get("use_cache", True),
                                max_samples=max_samples
                            )
                            
                            # Store the processed dataset
                            processed_datasets[dataset_name] = processed_dataset
                            pbar.update(1)
                            successful_count += 1
                            logger.info(f"Successfully processed {dataset_name}")
                    except Exception as e:
                        logger.error(f"Error processing dataset {dataset_name}: {e}")
                        logger.error(traceback.format_exc())
                        failed_count += 1
                else:
                    # Use the appropriate processor method if available
                    processor_method = f"process_{processor_name}"
                    if hasattr(self, processor_method):
                        # Get the processor method
                        processor = getattr(self, processor_method)
                        
                        # Special case for DS-1000 benchmark which is not on HF Hub
                        if processor_name == 'ds1000':
                            try:
                                logger.info(f"Processing DS-1000 benchmark")
                                processed_dataset = processor(None, streaming=streaming)
                                
                                # Save the processed dataset
                                if processed_dataset:
                                    if streaming and config.get("defer_save", False):
                                        # For streaming datasets, we can defer saving to save memory
                                        logger.info(f"Deferring save of processed dataset {dataset_name} (streaming mode)")
                                        processed_datasets[dataset_name] = processed_dataset
                                        successful_count += 1
                                        logger.info(f"Successfully processed {dataset_name} (save deferred)")
                                    else:
                                        # Standard save behavior
                                        logger.info(f"Saving processed dataset to {output_path}")
                                        if not streaming and hasattr(processed_dataset, 'save_to_disk'):
                                            processed_dataset.save_to_disk(output_path)
                                        
                                        # Important: Store the processed dataset and increment success count
                                        processed_datasets[dataset_name] = processed_dataset
                                        successful_count += 1
                                        logger.info(f"Successfully processed {dataset_name}")
                                else:
                                    logger.error(f"No data returned when processing {dataset_name}")
                                    failed_count += 1
                            except Exception as e:
                                logger.error(f"Error processing DS-1000 benchmark: {e}")
                                logger.error(traceback.format_exc())
                                failed_count += 1
                                continue
                        
                        # Handle Magicoder dataset which is an instruction-tuning dataset
                        if processor_name == 'instruct_code' and 'magicoder' in dataset_path.lower():
                            logger.info(f"Processing Magicoder dataset using instruct_code processor")
                        
                        # Load the dataset from HF Hub for other datasets
                        try:
                            # Some datasets require special configurations
                            logger.info(f"Loading dataset: {dataset_path}")
                            
                            # Check if HuggingFace token is available
                            token_available = "HF_TOKEN" in os.environ
                            
                            # Additional parameters for specific datasets or formats
                            extra_params = {}
                            
                            # For MBPP, Magicoder and other datasets with Python examples, trust remote code
                            if any(name in dataset_path.lower() for name in ['mbpp', 'code', 'magicoder']):
                                extra_params["trust_remote_code"] = True
                            
                            with timer(f"Loading dataset {dataset_name}"):
                                try:
                                    dataset = load_dataset(
                                        dataset_path,
                                        split=split,
                                        streaming=streaming,
                                        token=os.environ.get("HF_TOKEN") if token_available else None,
                                        **extra_params
                                    )
                                except Exception as e:
                                    # Try with a different authentication approach
                                    logger.warning(f"Error loading dataset with token: {e}, trying alternative")
                                    dataset = load_dataset(
                                        dataset_path,
                                        split=split,
                                        streaming=streaming,
                                        use_auth_token=os.environ.get("HF_TOKEN") if token_available else None,
                                        **extra_params
                                    )
                                
                                # Ensure the dataset is not empty
                                if not dataset:
                                    logger.error(f"Empty dataset loaded for {dataset_name}")
                                    failed_count += 1
                                    continue
                        
                                # Process the dataset
                                logger.info(f"Processing dataset {dataset_name} with {processor_method}")
                                processed_dataset = processor(dataset, streaming=streaming, max_samples=max_samples)
                                
                                # Save the processed dataset if we got anything back
                                if processed_dataset:
                                    if streaming and config.get("defer_save", False):
                                        # For streaming datasets, we can defer saving to save memory
                                        logger.info(f"Deferring save of processed dataset {dataset_name} (streaming mode)")
                                        processed_datasets[dataset_name] = processed_dataset
                                        successful_count += 1
                                        logger.info(f"Successfully processed {dataset_name} (save deferred)")
                                    else:
                                        # Standard save behavior
                                        logger.info(f"Saving processed dataset to {output_path}")
                                        if not streaming and hasattr(processed_dataset, 'save_to_disk'):
                                            processed_dataset.save_to_disk(output_path)
                                        
                                        # Important: Store the processed dataset and increment success count
                                        processed_datasets[dataset_name] = processed_dataset
                                        successful_count += 1
                                        logger.info(f"Successfully processed {dataset_name}")
                                else:
                                    logger.error(f"No data returned when processing {dataset_name}")
                                    failed_count += 1
                        except Exception as e:
                            logger.error(f"Error loading or processing dataset {dataset_name}: {e}")
                            logger.error(traceback.format_exc())
                            failed_count += 1
                    else:
                        logger.error(f"No processor found for {dataset_name} (looking for {processor_method})")
                        failed_count += 1
            except Exception as e:
                logger.error(f"Unexpected error processing dataset {dataset_name}: {e}")
                logger.error(traceback.format_exc())
                failed_count += 1
        
        # Final memory report
        memory_info = self._check_resources(force=True)
        logger.info(f"Memory usage: {memory_info.get('memory_mb', 0):.1f} MB ({memory_info.get('memory_percent', 0):.1f}% of system memory)")
        if 'gpu_allocated_mb' in memory_info:
            logger.info(f"GPU memory: {memory_info.get('gpu_allocated_mb', 0):.1f} MB ({memory_info.get('gpu_percent', 0):.1f}% of GPU memory)")
        
        logger.info(f"Processed {successful_count} datasets successfully, {failed_count} failed")
        
        if successful_count == 0:
            logger.warning("No datasets were processed successfully")
        
        # Save to Google Drive
        for dataset_name, processed_dataset in processed_datasets.items():
            if hasattr(processed_dataset, 'save_to_disk'):
                gdrive_path = f"/content/drive/MyDrive/datasets/{dataset_name}_processed"
                processed_dataset.save_to_disk(gdrive_path)
                print(f"Saved {dataset_name} to Google Drive")
        
        return processed_datasets