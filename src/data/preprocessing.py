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
        
    def detect_language(self, text):
        """
        Detect the language of a text.
        
        Args:
            text: Text to analyze
        
        Returns:
            ISO 639-1 language code (e.g., 'en', 'ar')
        """
        try:
            # Try using langid for detection
            import langid
            lang_code, _ = langid.classify(text)
            return lang_code
        except (ImportError, Exception) as e:
            logger.debug(f"langid detection failed: {str(e)}")
            
            try:
                # Fallback to langdetect
                from langdetect import detect
                return detect(text)
            except (ImportError, Exception) as e:
                logger.debug(f"langdetect detection failed: {str(e)}")
                
                # Default to English if detection fails
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
            # Use our own detect_language method instead of undefined 'detect' function
            lang_code = self.detect_language(text[:1000])  # Only use first 1000 chars for speed
            
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
        """
        Process CodeSearchNet dataset.
        
        Args:
            dataset: The CodeSearchNet dataset
            streaming: Whether to use streaming mode for memory efficiency
            language: Specific language to filter by, or None to process all languages
            
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
            for example in self._safe_dataset_iterator(dataset):
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
        
        # Handle non-dictionary examples (like strings or lists)
        if not isinstance(example, dict):
            # For string examples, return a default formatting
            if isinstance(example, str):
                return {
                    "prompt": "Write a Python function",
                    "completion": example
                }
            # For other non-dict types, return empty result
            return result
        
        # Normal processing for dictionary examples
        for target_field, possible_names in field_names_map.items():
            for name in possible_names:
                if name in example and example[name]:
                    result[target_field] = example[name]
                    break
                    
        return result
        
    def process_instruct_code(self, dataset: Union[Dataset, DatasetDict],
                             streaming: bool = False) -> Union[Dataset, DatasetDict]:
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
    
    def process_ds1000(self, dataset: Union[Dataset, DatasetDict, str] = None,
                      streaming: bool = False) -> Union[Dataset, DatasetDict]:
        """
        Process DS-1000 benchmark dataset.
        
        Args:
            dataset: Optional dataset object or path to the dataset
            streaming: Whether to use streaming mode for memory efficiency
            
        Returns:
            Processed dataset or list of processed examples
        """
        logger.info("Attempting to process DS-1000 benchmark...")
        
        try:
            # DS-1000 is a benchmark for code generation, not a standard dataset on HF Hub
            # Check if the dataset was provided as a local path or object
            if dataset is None or isinstance(dataset, str):
                # Try to locate DS-1000 manually if not provided
                possible_paths = [
                    "data/raw/DS-1000",
                    "DS-1000",
                    "../DS-1000",
                    "data/DS-1000"
                ]
                
                # Check if any path exists
                ds_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        ds_path = path
                        logger.info(f"Found DS-1000 at {path}")
                        break
                
                if ds_path is None:
                    logger.warning("DS-1000 benchmark not found. Please clone it from GitHub: "
                                 "https://github.com/salesforce/DS-1000")
                    logger.warning("You can clone it with: git clone https://github.com/salesforce/DS-1000.git")
                    return {"processed_text": [], "length": [], "message": "DS-1000 benchmark not found"}
                
                # Create a simpler dataset format from the benchmark
                processed_examples = []
                
                # Process each problem domain directory
                domains = [d for d in os.listdir(ds_path) if os.path.isdir(os.path.join(ds_path, d)) 
                          and not d.startswith('.')]
                
                logger.info(f"Found {len(domains)} domains in DS-1000: {domains}")
                
                for domain in domains:
                    domain_path = os.path.join(ds_path, domain)
                    problems = [d for d in os.listdir(domain_path) if os.path.isdir(os.path.join(domain_path, d))]
                    
                    for problem_id in problems:
                        problem_path = os.path.join(domain_path, problem_id)
                        
                        # Look for prompt and canonical solution files
                        prompt_file = os.path.join(problem_path, "prompt.txt")
                        solution_file = os.path.join(problem_path, "solutions/reference.py")
                        
                        if not os.path.exists(prompt_file) or not os.path.exists(solution_file):
                            continue
                        
                        try:
                            # Read prompt and solution
                            with open(prompt_file, 'r', encoding='utf-8') as f:
                                prompt = f.read().strip()
                            
                            with open(solution_file, 'r', encoding='utf-8') as f:
                                solution = f.read().strip()
                            
                            # Skip if either is empty
                            if not prompt or not solution:
                                continue
                            
                            # Add domain information to prompt
                            formatted_prompt = f"Given a problem from {domain}:\n\n{prompt}"
                            
                            # Process using common preprocessing
                            processed = self.common_preprocessing(
                                {"prompt": formatted_prompt, "completion": solution},
                                "prompt", "completion",
                                streaming=True  # Use streaming processing for individual examples
                            )
                            
                            # Add metadata
                            if "processed_text" in processed and processed["processed_text"]:
                                processed["domain"] = domain
                                processed["problem_id"] = problem_id
                                processed_examples.append(processed)
                            
                        except Exception as e:
                            logger.warning(f"Error processing DS-1000 problem {domain}/{problem_id}: {e}")
                
                # Create proper dataset object if not in streaming mode
                if not streaming and processed_examples:
                    from datasets import Dataset as HFDataset
                    
                    # Extract features
                    features = {
                        "processed_text": [ex["processed_text"][0] if isinstance(ex["processed_text"], list) else ex["processed_text"] 
                                         for ex in processed_examples if "processed_text" in ex and ex["processed_text"]],
                        "length": [ex["length"][0] if isinstance(ex["length"], list) else ex["length"] 
                                 for ex in processed_examples if "length" in ex and ex["length"]],
                        "domain": [ex.get("domain", "") for ex in processed_examples if "processed_text" in ex and ex["processed_text"]],
                        "problem_id": [ex.get("problem_id", "") for ex in processed_examples if "processed_text" in ex and ex["processed_text"]]
                    }
                    
                    logger.info(f"Created dataset with {len(features['processed_text'])} examples from DS-1000 benchmark")
                    return HFDataset.from_dict(features)
                else:
                    logger.info(f"Processed {len(processed_examples)} examples from DS-1000 benchmark")
                    return processed_examples
            else:
                # If a dataset object was provided directly, process it normally
                logger.info("Processing provided DS-1000 dataset object")
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
                    logger.warning(f"Error processing provided DS-1000 dataset: {e}")
                    return dataset.map(
                        lambda examples: self.common_preprocessing(
                            {"prompt": examples["prompt"], "completion": examples["canonical_solution"]},
                            "prompt", "completion",
                            streaming=streaming
                        ),
                        batched=True,
                        batch_size=100 if streaming else None
                    )
        except Exception as e:
            logger.error(f"Error processing DS-1000 benchmark: {e}")
            logger.error(traceback.format_exc())
            return {"processed_text": [], "length": [], "error": str(e)}
    
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
                    except Exception:
                        continue

                return processed_examples
    
    def process_the_stack(self, dataset: Union[Dataset, DatasetDict], language: str = None, 
                         streaming: bool = True, callback: Optional[Callable] = None, 
                         intermediate_save: bool = True, save_every: int = 10000,
                         natural_languages: List[str] = None, sampling_ratio: float = 1.0,
                         max_samples: int = None) -> Iterator[Dict]:
        """
        Process The Stack dataset (DISABLED - kept for compatibility).
        This dataset has been disabled in the configuration.
        
        Returns an empty iterator to maintain compatibility with existing code.
        """
        logger.warning("The Stack dataset processing is disabled")
        logger.warning("The dataset has been removed from active configuration")
        logger.warning("To process The Stack, please fix the dataset configuration and enable it")
        
        # Return empty iterator to maintain compatibility
        return iter([])
    
    def process_humaneval(self, dataset: Union[Dataset, DatasetDict],
                         streaming: bool = False) -> Union[Dataset, DatasetDict]:
        """Process HumanEval dataset by loading it directly from HuggingFace."""
        logger.info("Processing HumanEval dataset via direct loading...")
        
        # Always use direct loading from Hugging Face for HumanEval
        try:
            from datasets import load_dataset, Dataset as HFDataset
            
            # Load the dataset directly from Hugging Face
            hf_token = os.environ.get("HF_TOKEN")
            logger.info("Loading HumanEval directly from Hugging Face")
            
            try:
                # Try with token authentication first
                raw_dataset = load_dataset("openai/openai_humaneval", token=hf_token)
                logger.info(f"Successfully loaded HumanEval dataset with authentication")
            except Exception as e:
                # Fallback to loading without token
                logger.warning(f"Failed to load with token, trying without: {e}")
                raw_dataset = load_dataset("openai/openai_humaneval")
                logger.info(f"Successfully loaded HumanEval dataset without authentication")
            
            # Verify the dataset structure and size
            if "test" in raw_dataset and hasattr(raw_dataset["test"], "__len__"):
                logger.info(f"HumanEval dataset has {len(raw_dataset['test'])} examples")
            else:
                logger.warning(f"Unexpected dataset structure: {list(raw_dataset.keys())}")
            
            # Process the examples
            processed_examples = []
            
            for example in raw_dataset["test"]:
                try:
                    # Extract the key fields
                    prompt = example.get("prompt", "")
                    solution = example.get("canonical_solution", "")
                    task_id = example.get("task_id", "")
                    
                    # Skip empty examples
                    if not prompt or not solution:
                        continue
                    
                    # Create a properly formatted example
                    processed_text = f"User: {prompt.strip()}\nAssistant: {solution.strip()}"
                    
                    # Add to processed examples
                    processed_examples.append({
                        "processed_text": processed_text,
                        "length": len(processed_text),
                        "task_id": task_id
                    })
                except Exception as e:
                    logger.warning(f"Error processing HumanEval example: {e}")
            
            logger.info(f"Successfully processed {len(processed_examples)} examples from HumanEval")
            
            # Return results based on whether streaming is requested
            if not streaming:
                # Create a new Dataset with the processed examples
                features = {
                    "processed_text": [ex["processed_text"] for ex in processed_examples],
                    "length": [ex["length"] for ex in processed_examples],
                    "task_id": [ex["task_id"] for ex in processed_examples]
                }
                return HFDataset.from_dict(features)
            else:
                return processed_examples
            
        except Exception as e:
            logger.error(f"Failed to process HumanEval dataset: {e}")
            logger.error(traceback.format_exc())
            
            # Return empty results as fallback
            if not streaming:
                from datasets import Dataset as HFDataset
                return HFDataset.from_dict({
                    "processed_text": [],
                    "length": [],
                    "task_id": []
                })
            else:
                return []
    
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
            
            try:
                # Extract processor name and dataset path
                processor_name = config.get("processor", dataset_name)
                dataset_path = config.get("path", dataset_name)
                split = config.get("split", "train")
                streaming = config.get("streaming", False)
                
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
                                max_samples=config.get("max_samples", None)
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
                                    logger.info(f"Saving processed DS-1000 to {output_path}")
                                    if not streaming and hasattr(processed_dataset, 'save_to_disk'):
                                        processed_dataset.save_to_disk(output_path)
                                    processed_datasets[dataset_name] = processed_dataset
                                    successful_count += 1
                                    logger.info(f"Successfully processed {dataset_name}")
                                else:
                                    logger.error(f"No data returned when processing {dataset_name}")
                                    failed_count += 1
                                continue
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
                                processed_dataset = processor(dataset, streaming=streaming)
                                
                                # Save the processed dataset if we got anything back
                                if processed_dataset:
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
        
        return processed_datasets