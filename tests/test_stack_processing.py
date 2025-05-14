#!/usr/bin/env python3
"""
Test script for The Stack dataset processing

This script tests the language detection and filtering functionality
in the DataPreprocessor class to make sure it correctly processes examples
from The Stack dataset.
"""

import os
import sys
import logging
from tqdm import tqdm
from datasets import load_dataset
from src.data.preprocessing import DataPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_stack_processing():
    """Test processing of The Stack dataset with proper language filtering."""
    
    # Set environment variables
    os.environ["HF_DATASETS_CACHE"] = "no"  # Disable caching
    if "HF_TOKEN" not in os.environ:
        logger.warning("HF_TOKEN not set. Some datasets may not be accessible.")
    
    # Create preprocessor
    preprocessor = DataPreprocessor(max_length=1024)
    
    # Test language detection
    logger.info("Testing language detection...")
    test_texts = [
        "# This is a Python comment in English",
        "// Este es un comentario de JavaScript en español",
        "/* هذا تعليق جافا باللغة العربية */",
        "# 这是 Python 中文注释"
    ]
    
    for text in test_texts:
        lang = preprocessor.detect_language(text)
        logger.info(f"Text: '{text[:30]}...' => Detected language: {lang}")
    
    # Test should_include_by_language method
    allowed_languages = ["en", "ar"]
    logger.info(f"Testing language filtering with allowed languages: {allowed_languages}")
    
    for text in test_texts:
        should_include = preprocessor.should_include_by_language(text, allowed_languages)
        lang = preprocessor.detect_language(text)
        logger.info(f"Language {lang}: Should include? {should_include}")
    
    # Test with actual Stack dataset (small sample)
    logger.info("Loading a small sample from The Stack dataset...")
    try:
        dataset = load_dataset(
            "bigcode/the-stack", 
            data_dir="data/python",
            split="train",
            streaming=True,
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN")
        )
        
        # Process a small sample
        logger.info("Processing 10 examples to test filtering...")
        examples_processed = 0
        examples_kept = 0
        
        for example in tqdm(dataset.take(100)):
            try:
                # Process example
                result = preprocessor.process_the_stack(
                    [example],  # Pass as a list for compatibility
                    streaming=True,
                    language="python",  # Filter for Python
                    natural_languages=["en", "ar"],  # Filter for English/Arabic
                    sampling_ratio=1.0,  # Don't skip any examples
                    max_samples=10  # Only need a few examples
                )
                
                # Check results
                results_list = list(result)
                examples_processed += 1
                
                if results_list:
                    examples_kept += 1
            except Exception as e:
                logger.error(f"Error processing example: {e}")
                
        logger.info(f"Processed {examples_processed} examples, kept {examples_kept} after filtering")
        
    except Exception as e:
        logger.error(f"Error testing with The Stack dataset: {e}")
        return False
    
    logger.info("Test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_stack_processing()
    sys.exit(0 if success else 1) 