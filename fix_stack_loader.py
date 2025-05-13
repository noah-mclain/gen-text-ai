#!/usr/bin/env python3
"""
Fix for The Stack dataset loading

This script directly loads The Stack dataset from Hugging Face for specific languages
and saves small samples locally for training. This bypasses potential issues with
the complex filtering logic in the DataPreprocessor.
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset, Dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fix_stack_loader")

# Define sample size per language
MAX_SAMPLES = 10000

def save_dataset_sample(dataset, output_path, max_samples=MAX_SAMPLES):
    """Save a sample of the dataset to disk"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Get samples from streaming dataset
    samples = []
    
    for i, example in enumerate(tqdm(dataset.take(max_samples), desc=f"Saving samples to {output_path}", total=max_samples)):
        if i >= max_samples:
            break
        
        # Keep only specific fields to reduce storage
        processed_example = {
            "processed_text": example.get("content", ""),
            "length": len(example.get("content", "")),
            "language": example.get("lang", "")
        }
        
        samples.append(processed_example)
    
    # Create Dataset from the collected samples
    if samples:
        dataset_obj = Dataset.from_dict({
            "processed_text": [s["processed_text"] for s in samples],
            "length": [s["length"] for s in samples],
            "language": [s["language"] for s in samples]
        })
        
        # Save to disk
        dataset_obj.save_to_disk(output_path)
        logger.info(f"Saved {len(samples)} examples to {output_path}")
        return True
    else:
        logger.error(f"No samples could be collected for dataset")
        return False

def main():
    """Main function to fix The Stack dataset loading"""
    
    # Check for HF_TOKEN
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("No HF_TOKEN found in environment. Set it for authenticated access.")
        logger.warning("export HF_TOKEN=your_token")
    
    # Languages to process
    languages = ["python", "java", "javascript"]
    
    # Create base output directory
    output_base = "data/processed"
    os.makedirs(output_base, exist_ok=True)
    
    all_samples = []
    
    # Process each language
    for language in languages:
        logger.info(f"Processing {language} examples...")
        
        try:
            # Try loading the dataset directly
            language_dir = f"data/{language}"
            
            # Load the dataset
            logger.info(f"Loading dataset bigcode/the-stack with data_dir={language_dir}")
            dataset = load_dataset(
                "bigcode/the-stack",
                data_dir=language_dir,
                split="train",
                streaming=True,
                trust_remote_code=True,
                use_auth_token=hf_token
            )
            
            # Save language-specific samples
            lang_output_path = os.path.join(output_base, f"the_stack_{language}_processed")
            success = save_dataset_sample(dataset, lang_output_path, max_samples=MAX_SAMPLES // len(languages))
            
            if success:
                logger.info(f"Successfully saved {language} samples")
            else:
                logger.error(f"Failed to save {language} samples")
                
        except Exception as e:
            logger.error(f"Error processing {language}: {e}")
            continue
    
    # Try to save combined dataset
    try:
        # Combine all saved datasets
        combined_samples = []
        
        for language in languages:
            lang_path = os.path.join(output_base, f"the_stack_{language}_processed")
            
            if os.path.exists(lang_path):
                try:
                    # Load saved dataset
                    lang_dataset = Dataset.load_from_disk(lang_path)
                    logger.info(f"Loaded {len(lang_dataset)} samples from {lang_path}")
                    
                    # Extract examples
                    for example in lang_dataset:
                        combined_samples.append(example)
                except Exception as e:
                    logger.error(f"Error loading dataset from {lang_path}: {e}")
        
        # Save combined dataset
        if combined_samples:
            # Create dataset object
            combined_dataset = Dataset.from_list(combined_samples)
            
            # Save to disk
            combined_output_path = os.path.join(output_base, "the_stack_filtered_processed")
            combined_dataset.save_to_disk(combined_output_path)
            
            logger.info(f"Saved combined dataset with {len(combined_dataset)} examples to {combined_output_path}")
        else:
            logger.error("No samples were collected for the combined dataset")
    except Exception as e:
        logger.error(f"Error creating combined dataset: {e}")
    
    # Check if we were successful
    combined_path = os.path.join(output_base, "the_stack_filtered_processed")
    if os.path.exists(combined_path):
        logger.info(f"SUCCESS: The Stack dataset is now available at {combined_path}")
        return 0
    else:
        logger.error("FAILED: Could not create The Stack dataset")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 