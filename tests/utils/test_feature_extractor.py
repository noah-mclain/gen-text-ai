#!/usr/bin/env python3
"""
Test Feature Extractor

This script tests the functionality of the feature extractor
without requiring actual datasets from Google Drive.
"""

import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path
import logging
import numpy as np

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from datasets import Dataset
    from src.data.processors.feature_extractor import FeatureExtractor
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure all required dependencies are installed")
    sys.exit(1)

class TestFeatureExtractor(unittest.TestCase):
    """Test cases for the feature extractor."""
    
    def setUp(self):
        """Set up for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "features")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create a simple test dataset
        self.test_dataset = Dataset.from_dict({
            "text": [
                "This is a test sentence.",
                "Another example for testing tokenization.",
                "Feature extraction should work on this text."
            ]
        })
        
        # Save the test dataset to disk
        self.dataset_path = os.path.join(self.temp_dir, "test_dataset")
        self.test_dataset.save_to_disk(self.dataset_path)
        
        # Initialize feature extractor with a small model
        self.feature_extractor = FeatureExtractor(
            model_name="google/flan-t5-small",  # Small model for faster tests
            max_length=32,  # Small max length for faster tests
            padding="max_length",
            truncation=True
        )
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
    
    def test_tokenize_examples(self):
        """Test tokenization of examples."""
        examples = {"text": ["This is a test."]}
        
        # Test for causal language model
        result = self.feature_extractor.tokenize_examples(examples, is_encoder_decoder=False)
        
        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)
        self.assertIn("labels", result)
        
        # Test for encoder-decoder model
        result_ed = self.feature_extractor.tokenize_examples(examples, is_encoder_decoder=True)
        
        self.assertIn("input_ids", result_ed)
        self.assertIn("attention_mask", result_ed)
        self.assertIn("labels", result_ed)
        
        # Verify shape with max_length
        self.assertEqual(result["input_ids"].shape[1], self.feature_extractor.max_length)
        
        # Verify shifting for causal LM
        # Labels should be shifted (input without first token, -100 at the end)
        self.assertEqual(result["labels"][0, -1].item(), -100)
    
    def test_prepare_dataset_for_training(self):
        """Test preparation of a dataset for training."""
        processed = self.feature_extractor.prepare_dataset_for_training(
            self.test_dataset,
            text_column="text",
            is_encoder_decoder=True,
            batch_size=2,
            num_proc=1
        )
        
        # Check that the processing was successful
        self.assertEqual(len(processed), len(self.test_dataset))
        self.assertIn("input_ids", processed.column_names)
        self.assertIn("attention_mask", processed.column_names)
        self.assertIn("labels", processed.column_names)
        
        # Check dimensions
        self.assertEqual(processed[0]["input_ids"].shape[0], self.feature_extractor.max_length)
    
    def test_load_dataset_from_disk(self):
        """Test loading a dataset from disk."""
        loaded = self.feature_extractor.load_dataset_from_drive(
            self.dataset_path,
            from_google_drive=False,
            drive_manager=None
        )
        
        # Check that loading was successful
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded), len(self.test_dataset))
        self.assertIn("text", loaded.column_names)
    
    def test_extract_features_from_disk(self):
        """Test extracting features from a dataset on disk."""
        dataset_paths = {
            "test": self.dataset_path
        }
        
        processed = self.feature_extractor.extract_features_from_drive_datasets(
            dataset_paths=dataset_paths,
            output_dir=self.output_dir,
            from_google_drive=False,
            text_column="text",
            is_encoder_decoder=True,
            save_to_disk=True,
            drive_manager=None
        )
        
        # Check that processing was successful
        self.assertIsNotNone(processed)
        self.assertEqual(len(processed), len(self.test_dataset))
        
        # Check that files were saved
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "test_features")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "combined_features")))

    def test_alternate_text_column(self):
        """Test using an alternate text column when the specified one doesn't exist."""
        # Create a dataset with 'content' instead of 'text'
        alt_dataset = Dataset.from_dict({
            "content": [
                "This is in the content field.",
                "Another example in content field."
            ]
        })
        
        # Save the dataset
        alt_path = os.path.join(self.temp_dir, "alt_dataset")
        alt_dataset.save_to_disk(alt_path)
        
        # Try to process with 'text' column (which doesn't exist)
        processed = self.feature_extractor.prepare_dataset_for_training(
            alt_dataset,
            text_column="text",  # This doesn't exist, should use 'content' instead
            is_encoder_decoder=True,
            batch_size=2,
            num_proc=1
        )
        
        # Check that it successfully used the alternative column
        self.assertEqual(len(processed), len(alt_dataset))
        self.assertIn("input_ids", processed.column_names)

if __name__ == "__main__":
    unittest.main() 