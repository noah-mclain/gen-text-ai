import os
import argparse
import logging
from datasets import load_dataset
from src.data.preprocessing import DataPreprocessor

# Configure logging to be more concise
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'  # Shorter time format
)
logger = logging.getLogger(__name__)

# Reduce verbose output from Hugging Face
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

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
            
        # Always use streaming mode to avoid batch processing errors and minimize memory usage
        logger.info(f"Loading dataset with streaming mode (trust_remote_code={trust_remote_code})")
        dataset = load_dataset(
            dataset_path, 
            streaming=True, 
            split=split,
            trust_remote_code=trust_remote_code,
            # Minimize caching to avoid RAM/VRAM usage
            use_auth_token=os.environ.get("HF_TOKEN")
        )
            
        logger.info(f"Successfully loaded dataset")
        
        # Find appropriate processor method
        processor_func = getattr(preprocessor, f"process_{processor_name}")
        
        # Process the dataset with streaming
        logger.info(f"Processing dataset...")
        processed_dataset = processor_func(dataset, streaming=True)
        
        # For streaming mode, process a few examples
        count = 0
        successful = 0
        
        # Only show a minimal number of examples to avoid cluttering the terminal
        for example in processed_dataset:
            if count >= max_samples:
                break
                
            count += 1
            
            if isinstance(example, dict) and "processed_text" in example:
                successful += 1
                # Only show the first example to minimize output
                if successful == 1:
                    # Show just a brief preview
                    preview = example["processed_text"][:80] + "..." if len(example["processed_text"]) > 80 else example["processed_text"]
                    logger.info(f"Example: {preview}")
            
        logger.info(f"Successfully processed {successful}/{count} examples")
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
    parser.add_argument("--quiet", action="store_true", help="Reduce logging output")
    
    args = parser.parse_args()
    
    # Set even more minimal logging if quiet mode is enabled
    if args.quiet:
        logging.basicConfig(level=logging.WARNING)
    
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