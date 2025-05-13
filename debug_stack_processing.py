#!/usr/bin/env python3
"""
Debug script for The Stack dataset processing

This script directly loads and processes The Stack dataset with verbose logging
to identify any issues in the processing pipeline.
"""

import os
import sys
import json
import logging
import time
from datasets import load_dataset
from src.data.preprocessing import DataPreprocessor

# Configure debug-level logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("debug_stack")

def debug_stack_processing():
    """Debug the processing of The Stack dataset"""
    
    logger.info("Starting debug of The Stack dataset processing")
    
    # Set environment variables
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN not set in environment. Authentication might fail.")
        
    # Load configuration
    try:
        config_path = "config/dataset_config.json"
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        stack_config = config.get("the_stack_filtered")
        if not stack_config:
            logger.error("the_stack_filtered configuration not found!")
            return False
            
        logger.info(f"Stack config: {json.dumps(stack_config, indent=2)}")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return False
    
    # Create preprocessor
    logger.info("Creating DataPreprocessor")
    preprocessor = DataPreprocessor(max_length=1024)
    
    # Test language detection
    logger.info("Testing language detection...")
    test_text = "# This is a Python comment in English"
    lang = preprocessor.detect_language(test_text)
    logger.info(f"Language detection test: '{test_text}' => {lang}")
    should_include = preprocessor.should_include_by_language(test_text, ["en", "ar"])
    logger.info(f"Language filtering test: '{test_text}' => should include: {should_include}")
    
    # Load The Stack dataset
    try:
        logger.info("Loading The Stack dataset...")
        
        # Get parameters from configuration
        path = stack_config.get("path", "bigcode/the-stack")
        data_dir = stack_config.get("data_dir", "data")
        split = stack_config.get("split", "train")
        
        # Try loading each language separately for debugging
        languages = stack_config.get("languages", ["python"])
        
        for language in languages:
            logger.info(f"Testing loading for language: {language}")
            try:
                # Form the specific language directory path
                language_dir = os.path.join(data_dir, language)
                logger.info(f"Loading from {path} with data_dir={language_dir}")
                
                # Try loading with the specific language directory
                dataset = load_dataset(
                    path,
                    data_dir=language_dir,
                    split=split,
                    streaming=True,
                    trust_remote_code=True,
                    use_auth_token=hf_token
                )
                
                # Try to access a few samples
                logger.info(f"Attempting to fetch samples for {language}...")
                
                # Count available examples
                example_count = 0
                for example in dataset.take(3):
                    example_count += 1
                    logger.info(f"Example for {language}: {str(example)[:200]}...")
                
                logger.info(f"Successfully loaded {example_count} examples for {language}")
                
                # Process a small sample directly with the processor
                logger.info(f"Testing direct processing for {language}...")
                try:
                    logger.info("Starting direct processing test")
                    
                    # Create a small test dataset with just one example
                    test_example = next(iter(dataset.take(1)))
                    
                    # Process using the appropriate method
                    logger.info(f"Processing example: {str(test_example)[:200]}...")
                    
                    # Process with language filtering
                    result_generator = preprocessor.process_the_stack(
                        [test_example],  # Pass as a list for compatibility
                        language=language,
                        streaming=True,
                        natural_languages=stack_config.get("natural_languages", ["en"]),
                        sampling_ratio=1.0,  # Don't skip any examples in this test
                        max_samples=1  # Just need one example
                    )
                    
                    # Check if we got any results
                    results = list(result_generator)
                    if results:
                        logger.info(f"Successfully processed example for {language}")
                        logger.info(f"Processed result: {str(results[0])[:200]}...")
                    else:
                        logger.warning(f"No results returned for {language} after processing!")
                        # Check for language and comment matching issues
                        if "content" in test_example:
                            content = test_example["content"]
                            lang_name = test_example.get("lang", "").lower()
                            logger.info(f"Language in example: {lang_name}")
                            
                            # Extract comments for language detection test
                            import re
                            comment_pattern = r'(?:\/\/.*?$|\/\*[\s\S]*?\*\/|#.*?$|\'\'\'[\s\S]*?\'\'\'|"""[\s\S]*?""")'
                            comments = re.findall(comment_pattern, content, re.MULTILINE)
                            
                            if comments:
                                logger.info(f"Found {len(comments)} comments")
                                for i, comment in enumerate(comments[:3]):  # Show first 3 comments
                                    logger.info(f"Comment {i}: {comment[:100]}...")
                                    comment_lang = preprocessor.detect_language(comment)
                                    logger.info(f"Detected language: {comment_lang}")
                                    should_include = preprocessor.should_include_by_language(
                                        comment, stack_config.get("natural_languages", ["en"])
                                    )
                                    logger.info(f"Should include based on language: {should_include}")
                            else:
                                logger.warning("No comments found in the example")
                    
                except Exception as e:
                    logger.error(f"Error in direct processing for {language}: {e}")
                
            except Exception as e:
                logger.error(f"Error loading dataset for {language}: {e}")
                continue
        
        logger.info("Completed testing The Stack dataset loading and processing")
        return True
        
    except Exception as e:
        logger.error(f"Error in The Stack dataset processing: {e}")
        return False

if __name__ == "__main__":
    success = debug_stack_processing()
    sys.exit(0 if success else 1) 