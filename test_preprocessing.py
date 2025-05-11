import os
import argparse
import logging
from datasets import load_dataset
from src.data.preprocessing import DataPreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_preprocessing(dataset_name, processor_name, max_samples=10):
    """Test preprocessing on a small sample of data."""
    logger.info(f"Testing preprocessing for {dataset_name} using {processor_name} processor")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    try:
        # Load a small sample of the dataset
        logger.info(f"Loading dataset: {dataset_name}")
        
        # Find appropriate dataset path and split based on common repositories
        split = "train"  # Default split
        trust_remote_code = False  # Default to not trust remote code
        
        if dataset_name == "humaneval":
            dataset_path = "openai_humaneval"
            split = "test"  # HumanEval only has a test split
        elif dataset_name == "mbpp":
            dataset_path = "mbpp"
        elif dataset_name == "codesearchnet":
            dataset_path = "code_search_net"
            trust_remote_code = True  # CodeSearchNet needs remote code
        elif dataset_name == "the_stack":
            dataset_path = "bigcode/the-stack-smol"  # Use smaller test version
            trust_remote_code = True  # The Stack needs remote code
        elif dataset_name == "instruct_code":
            dataset_path = "codeparrot/apps"  # Alternative dataset with similar structure
            trust_remote_code = True  # APPS dataset needs remote code
        else:
            dataset_path = dataset_name
            
        # Always use streaming mode to avoid batch processing errors
        logger.info(f"Loading dataset with split: {split} in streaming mode (trust_remote_code={trust_remote_code})")
        dataset = load_dataset(
            dataset_path, 
            streaming=True, 
            split=split,
            trust_remote_code=trust_remote_code
        )
            
        logger.info(f"Successfully loaded dataset with streaming mode")
        
        # Find appropriate processor method
        processor_func = getattr(preprocessor, f"process_{processor_name}")
        
        # Process the dataset with streaming
        logger.info(f"Processing dataset...")
        processed_dataset = processor_func(dataset, streaming=True)
        
        # For streaming mode, process a few examples
        count = 0
        for example in processed_dataset:
            if count >= max_samples:
                break
            if isinstance(example, dict) and "processed_text" in example:
                if count < 3:  # Show only first 3 examples to keep output readable
                    logger.info(f"Example {count}: {example['processed_text'][:100]}...")
            count += 1
            
        logger.info(f"Successfully processed {count} examples in streaming mode")
        logger.info("Test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during test: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test preprocessing on a dataset")
    parser.add_argument("--dataset", type=str, default="humaneval", help="Dataset to test")
    parser.add_argument("--processor", type=str, default="humaneval", help="Processor type to use")
    parser.add_argument("--max_samples", type=int, default=10, help="Maximum samples to process")
    
    args = parser.parse_args()
    
    success = test_preprocessing(
        args.dataset, 
        args.processor,
        max_samples=args.max_samples
    )
    
    if success:
        print("Test completed successfully!")
    else:
        print("Test failed!")
        exit(1) 