#!/usr/bin/env python3
import os
import argparse
import json
import logging

# Fix the evaluator import to use a relative path
try:
    from .evaluator import ModelEvaluator
except ImportError:
    try:
        # Try alternate import path
        from src.evaluation.evaluator import ModelEvaluator
    except ImportError:
        # Last resort: add the directory to sys.path
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from evaluator import ModelEvaluator

# Import drive utils
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils.google_drive_manager import mount_google_drive, setup_drive_directories, get_drive_path
except ImportError:
    try:
        from src.utils.google_drive_manager import mount_google_drive, setup_drive_directories, get_drive_path
    except ImportError:
        # Fallback - define stub functions
        def mount_google_drive(*args, **kwargs):
            logger.warning("Google Drive mounting not available")
            return False
            
        def setup_drive_directories(*args, **kwargs):
            logger.warning("Google Drive setup not available")
            return {}
            
        def get_drive_path(local_path, drive_path=None, fallback_path=None):
            return fallback_path or local_path

# Import DS-1000 benchmark
try:
    from .ds1000_benchmark import evaluate_model_on_ds1000
except ImportError:
    try:
        from src.evaluation.ds1000_benchmark import evaluate_model_on_ds1000
    except ImportError:
        # Last resort: add the directory to sys.path if not already added
        if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from ds1000_benchmark import evaluate_model_on_ds1000

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_custom_prompts(prompts_file):
    """Load custom prompts from a JSON file."""
    with open(prompts_file, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned deepseek-coder models")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the fine-tuned model or adapter")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Path to the base model (required if model_path is a PEFT adapter)")
    parser.add_argument("--use_unsloth", action="store_true",
                        help="Use Unsloth for faster inference")
    parser.add_argument("--output_dir", type=str, default="../../results",
                        help="Directory to save evaluation results")
    
    # Evaluation options
    parser.add_argument("--eval_humaneval", action="store_true",
                        help="Evaluate on HumanEval benchmark")
    parser.add_argument("--eval_mbpp", action="store_true",
                        help="Evaluate on MBPP benchmark")
    parser.add_argument("--eval_ds1000", action="store_true",
                        help="Evaluate on DS-1000 data science benchmark")
    parser.add_argument("--ds1000_libraries", nargs="+", default=None,
                        help="Specific libraries to evaluate for DS-1000 (e.g., numpy pandas sklearn)")
    parser.add_argument("--eval_code_quality", action="store_true",
                        help="Evaluate code quality metrics (complexity, linting)")
    parser.add_argument("--eval_runtime", action="store_true",
                        help="Evaluate runtime metrics (execution time, memory usage)")
    parser.add_argument("--eval_semantic", action="store_true",
                        help="Evaluate semantic similarity metrics")
    parser.add_argument("--eval_all", action="store_true",
                        help="Evaluate on all available metrics and benchmarks")
    
    # Custom prompts
    parser.add_argument("--custom_prompts", type=str, default=None,
                        help="Path to a JSON file with custom prompts to evaluate")
    
    # Google Drive integration
    parser.add_argument("--use_drive", action="store_true", 
                        help="Use Google Drive for storage")
    parser.add_argument("--drive_base_dir", type=str, default="DeepseekCoder",
                        help="Base directory on Google Drive (if using Drive)")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature for evaluation")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Maximum number of new tokens to generate")
    
    args = parser.parse_args()
    
    # If eval_all is specified, enable all evaluation types
    if args.eval_all:
        args.eval_humaneval = True
        args.eval_mbpp = True
        args.eval_ds1000 = True
        args.eval_code_quality = True
        args.eval_runtime = True
        args.eval_semantic = True
    
    # Setup Google Drive if requested
    drive_paths = None
    if args.use_drive:
        logger.info("Attempting to mount Google Drive...")
        if mount_google_drive():
            logger.info(f"Setting up directories in Google Drive under {args.drive_base_dir}")
            drive_base = os.path.join("/content/drive/MyDrive", args.drive_base_dir)
            drive_paths = setup_drive_directories(drive_base)
            
            # Update output directory to use Google Drive
            args.output_dir = get_drive_path(args.output_dir, drive_paths["results"], args.output_dir)
            logger.info(f"Results will be saved to {args.output_dir}")
        else:
            logger.warning("Failed to mount Google Drive. Using local storage instead.")
            args.use_drive = False
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    logger.info(f"Initializing evaluator with model {args.model_path}")
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        base_model_path=args.base_model,
        use_unsloth=args.use_unsloth
    )
    
    results = {}
    
    # Evaluate on HumanEval
    if args.eval_humaneval:
        logger.info("Evaluating on HumanEval benchmark")
        humaneval_output = os.path.join(args.output_dir, "humaneval_results.json")
        humaneval_results = evaluator.evaluate_humaneval(output_path=humaneval_output)
        results["humaneval"] = humaneval_results
    
    # Evaluate on MBPP
    if args.eval_mbpp:
        logger.info("Evaluating on MBPP benchmark")
        mbpp_output = os.path.join(args.output_dir, "mbpp_results.json")
        mbpp_results = evaluator.evaluate_mbpp(output_path=mbpp_output)
        results["mbpp"] = mbpp_results
    
    # Evaluate on DS-1000
    if args.eval_ds1000:
        logger.info("Evaluating on DS-1000 benchmark")
        ds1000_output_dir = os.path.join(args.output_dir, "ds1000")
        try:
            ds1000_results = evaluate_model_on_ds1000(
                model=evaluator.model,
                tokenizer=evaluator.tokenizer,
                output_dir=ds1000_output_dir,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                libraries=args.ds1000_libraries
            )
            results["ds1000"] = ds1000_results
        except Exception as e:
            logger.error(f"Error evaluating on DS-1000: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Evaluate on custom prompts
    if args.custom_prompts:
        logger.info(f"Evaluating on custom prompts from {args.custom_prompts}")
        try:
            prompts = load_custom_prompts(args.custom_prompts)
            custom_output = os.path.join(args.output_dir, "custom_results.json")
            custom_results = evaluator.evaluate_code_generation(prompts, output_path=custom_output)
            results["custom"] = custom_results
        except Exception as e:
            logger.error(f"Error evaluating custom prompts: {str(e)}")
    
    # Create a metadata file with the evaluation settings
    metadata = {
        "model_path": args.model_path,
        "base_model": args.base_model,
        "evaluation_types": {
            "humaneval": args.eval_humaneval,
            "mbpp": args.eval_mbpp,
            "ds1000": args.eval_ds1000,
            "ds1000_libraries": args.ds1000_libraries,
            "code_quality": args.eval_code_quality,
            "runtime": args.eval_runtime,
            "semantic": args.eval_semantic,
            "custom_prompts": args.custom_prompts is not None
        },
        "generation_params": {
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens
        },
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }
    
    # Save overall results
    overall_output = os.path.join(args.output_dir, "overall_results.json")
    with open(overall_output, "w") as f:
        json.dump(results, f, indent=2)
        
    # Save metadata
    metadata_output = os.path.join(args.output_dir, "evaluation_metadata.json")
    with open(metadata_output, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Evaluation completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 