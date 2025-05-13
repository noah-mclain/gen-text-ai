#!/usr/bin/env python3
import os
import argparse
import logging
import subprocess
import json
import datetime
import tempfile
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up CUDA environment variables to prevent common issues
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6"
os.environ["TORCH_NVCC_FLAGS"] = "-Xfatbin -compress-all"

# Set CUBLAS workspace config to avoid CUDA OOM errors
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Enable safe PyTorch fallback mode
os.environ["PYTORCH_ALLOW_NET_FALLBACK"] = "1"

# Check for GPU and PyTorch compatibility
try:
    import torch
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"CUDNN version: {torch.backends.cudnn.version()}")
        logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA is not available. Using CPU mode.")
except ImportError as e:
    logger.error(f"Failed to import PyTorch: {e}")
    logger.warning("Continuing without PyTorch pre-check. This may lead to issues later.")
except Exception as e:
    logger.error(f"PyTorch/CUDA initialization error: {e}")
    logger.warning("Setting environment variables to attempt fallback to CPU mode")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
# Import Drive API utils after PyTorch to avoid import order issues
try:
    from src.utils.google_drive_manager import (
        sync_to_drive, 
        sync_from_drive, 
        configure_sync_method,
        setup_drive_directories
    )
except ImportError as e:
    logger.error(f"Failed to import Drive utils: {e}")

def ensure_directories():
    """Ensure all necessary directories exist."""
    dirs = [
        "data/raw",
        "data/processed",
        "models",
        "results",
        "visualizations",
        "logs"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        logger.info(f"Ensured directory exists: {d}")

def run_command(command, description=None):
    """Run a shell command and log its output."""
    if description:
        logger.info(f"Running: {description}")
    
    logger.info(f"Command: {command}")
    
    try:
        # Modify environment for the subprocess
        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=env
        )
        
        # Stream and log output
        for line in process.stdout:
            logger.info(line.strip())
        
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"Command failed with return code {process.returncode}")
            return False
        return True
    except Exception as e:
        logger.error(f"Error running command: {str(e)}")
        return False

def process_datasets(config_path, datasets=None, streaming=False, no_cache=False, 
                    use_drive_api=False, use_drive=False, credentials_path=None, 
                    drive_base_dir=None, headless=False, skip_local_storage=False):
    """Process datasets for fine-tuning."""
    cmd = f"python -m src.data.process_datasets --config {config_path}"
    if datasets:
        cmd += f" --datasets {' '.join(datasets)}"
    
    if streaming:
        cmd += " --streaming"
    
    if no_cache:
        cmd += " --no_cache"
    
    if use_drive_api:
        cmd += f" --use_drive_api --credentials_path {credentials_path} --drive_base_dir {drive_base_dir}"
        if headless:
            cmd += " --headless"
    elif use_drive:
        cmd += f" --use_drive --drive_base_dir {drive_base_dir}"
        if headless:
            cmd += " --headless"
    
    # Add the skip_local_storage parameter
    if skip_local_storage:
        cmd += " --skip_local_storage"
    
    return run_command(cmd, "Processing datasets")

def train_model(config_path, data_dir, use_drive_api=False, use_drive=False, 
              credentials_path=None, drive_base_dir=None, headless=False):
    """Train the model."""
    # Check if PYTHONPATH includes the project root, if not, add it to the command
    pythonpath_cmd = ""
    if os.getcwd() not in os.environ.get("PYTHONPATH", ""):
        pythonpath_cmd = f"export PYTHONPATH={os.getcwd()}:$PYTHONPATH && "

    cmd = f"{pythonpath_cmd}python -m src.training.train --config {config_path} --data_dir {data_dir}"
    
    if use_drive_api:
        cmd += f" --use_drive_api --credentials_path {credentials_path} --drive_base_dir {drive_base_dir}"
        if headless:
            cmd += " --headless"
    elif use_drive:
        cmd += f" --use_drive --drive_base_dir {drive_base_dir}"
        if headless:
            cmd += " --headless"
    
    return run_command(cmd, "Training model")

def evaluate_model(model_path, base_model, output_dir="results", 
                 use_drive_api=False, use_drive=False, credentials_path=None, 
                 drive_base_dir=None, headless=False):
    """Evaluate the model."""
    cmd = f"python -m src.evaluation.evaluate --model_path {model_path} --base_model {base_model} --output_dir {output_dir} --eval_humaneval --eval_mbpp --eval_ds1000"
    
    if use_drive_api:
        cmd += f" --use_drive_api --credentials_path {credentials_path} --drive_base_dir {drive_base_dir}"
        if headless:
            cmd += " --headless"
    elif use_drive:
        cmd += f" --use_drive --drive_base_dir {drive_base_dir}"
        if headless:
            cmd += " --headless"
    
    return run_command(cmd, "Evaluating model")

def visualize_results(training_log, results_dir, output_dir="visualizations", 
                     use_drive_api=False, use_drive=False, credentials_path=None, 
                     drive_base_dir=None, headless=False):
    """Visualize results."""
    cmd = f"python -m src.utils.visualize --training_log {training_log} --results_dir {results_dir} --output_dir {output_dir}"
    
    if use_drive_api:
        cmd += f" --use_drive_api --credentials_path {credentials_path} --drive_base_dir {drive_base_dir}"
        if headless:
            cmd += " --headless"
    elif use_drive:
        cmd += f" --use_drive --drive_base_dir {drive_base_dir}"
        if headless:
            cmd += " --headless"
    
    return run_command(cmd, "Visualizing results")

def calculate_hours_until_midnight():
    """Calculate the number of hours until midnight."""
    now = datetime.datetime.now()
    midnight = (now + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    time_until_midnight = midnight - now
    hours_until_midnight = time_until_midnight.total_seconds() / 3600
    
    # Round down to the nearest half hour and ensure at least 1 hour
    hours_until_midnight = max(1, int(hours_until_midnight * 2) / 2)
    
    return hours_until_midnight

def optimize_training_config(config_path, max_hours):
    """Optimize training configuration for time constraints."""
    logger.info(f"Optimizing training configuration for {max_hours} hour(s) time constraint")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create a temporary file for the updated config
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
        tmp_path = tmp_file.name
        
        # Update for faster training
        if "training" in config:
            # Limit epochs
            config["training"]["num_train_epochs"] = 1
            
            # Increase batch size if possible
            if config["training"].get("per_device_train_batch_size", 8) < 12:
                config["training"]["per_device_train_batch_size"] = 12
            
            # Increase gradient accumulation
            config["training"]["gradient_accumulation_steps"] = 8
            
            # Set evaluation and saving steps
            config["training"]["eval_steps"] = 100
            config["training"]["save_steps"] = 500
            
            # Enable mixed precision
            config["training"]["fp16"] = True
            
            # Set max training time
            config["training"]["max_train_time_hours"] = max_hours

        # Update dataset config for smaller sequence length
        if "dataset" in config:
            # Reduce sequence length for faster training
            if config["dataset"].get("max_length", 2048) > 1024:
                config["dataset"]["max_length"] = 1024
            
            # Adjust sampling to account for the limited time
            if "max_samples" in config["dataset"]:
                # Reduce samples for all datasets except the_stack_filtered
                for key in config["dataset"]["max_samples"]:
                    config["dataset"]["max_samples"][key] = min(5000, config["dataset"]["max_samples"].get(key, 5000))
            
            # Add our special stack dataset
            if "dataset_weights" in config["dataset"]:
                config["dataset"]["dataset_weights"]["the_stack_filtered"] = 3.0
                
            # Ensure streaming is enabled
            config["dataset"]["streaming"] = True
        
        # Save the updated config
        json.dump(config, tmp_file, indent=2)
    
    # Copy the temp file to the original location
    with open(tmp_path, 'r') as tmp_file:
        with open(config_path, 'w') as original_file:
            original_file.write(tmp_file.read())
    
    # Clean up the temp file
    os.unlink(tmp_path)
    
    logger.info(f"Updated training configuration saved to {config_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="DeepSeek-Coder Fine-Tuning Pipeline")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["all", "process", "train", "evaluate", "visualize", "quick-stack"],
                        help="Pipeline mode to run")
    parser.add_argument("--dataset_config", type=str, default="config/dataset_config.json",
                        help="Path to dataset configuration file")
    parser.add_argument("--training_config", type=str, default="config/training_config.json",
                        help="Path to training configuration file")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Optional list of dataset names to process")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the model or adapter (required for evaluate mode)")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Path to the base model (required for evaluate mode if model_path is an adapter)")
    
    # Memory efficiency options
    parser.add_argument("--streaming", action="store_true",
                        help="Load datasets in streaming mode to save memory")
    parser.add_argument("--no_cache", action="store_true",
                        help="Disable caching for datasets to save disk space")
    
    # Google Drive API options (for Paperspace)
    parser.add_argument("--use_drive_api", action="store_true",
                        help="Use Google Drive API instead of mounting (for Paperspace)")
    parser.add_argument("--use_drive", action="store_true",
                        help="Use Google Drive integration with rclone (preferred)")
    parser.add_argument("--credentials_path", type=str, default="credentials.json",
                        help="Path to Google Drive API credentials JSON file")
    parser.add_argument("--drive_base_dir", type=str, default="DeepseekCoder",
                        help="Base directory on Google Drive")
    parser.add_argument("--headless", action="store_true",
                        help="Use headless authentication for environments without a browser")
    parser.add_argument("--skip_local_storage", action="store_true",
                        help="Skip storing datasets locally, sync directly to Drive")
    
    # Quick Stack mode options
    parser.add_argument("--max-hours", type=float, default=None,
                        help="Maximum training time in hours (for quick-stack mode)")
    parser.add_argument("--auto-time", action="store_true",
                        help="Automatically calculate time until midnight (for quick-stack mode)")
    parser.add_argument("--skip-preprocessing", action="store_true",
                        help="Skip preprocessing step (for quick-stack mode)")
    
    args = parser.parse_args()
    
    # Ensure directories exist before doing anything else
    ensure_directories()
    
    # Check for HF_TOKEN environment variable
    if os.environ.get("HF_TOKEN"):
        logger.info("Using Hugging Face token from environment variables")
    else:
        logger.warning("HF_TOKEN environment variable not set. Some datasets might be inaccessible.")
        logger.warning("Set your token using: export HF_TOKEN=your_huggingface_token")
    
    # Special handling for quick-stack mode
    if args.mode == "quick-stack":
        # Set up for the Quick Stack pipeline
        logger.info("Starting Quick Stack pipeline")
        
        # Determine optimal training time
        if args.auto_time:
            max_hours = calculate_hours_until_midnight()
            logger.info(f"Auto-calculated training time: {max_hours} hours")
        else:
            max_hours = args.max_hours or 10.0
            logger.info(f"Using specified training time: {max_hours} hours")
        
        # Optimize training config for the available time
        optimize_training_config(args.training_config, max_hours)
        
        # Run the optimized pipeline for The Stack
        if not args.skip_preprocessing:
            # Special dataset list that includes the Stack
            stack_datasets = ["the_stack_filtered"]
            
            # Add other datasets if specified
            if args.datasets:
                stack_datasets.extend(args.datasets)
            else:
                # Default set of additional datasets
                stack_datasets.extend(["codesearchnet_all", "code_alpaca", "mbpp", "humaneval", "codeparrot"])
            
            logger.info(f"Processing datasets for Quick Stack pipeline: {', '.join(stack_datasets)}")
            
            if not process_datasets(
                args.dataset_config,
                datasets=stack_datasets,
                streaming=args.streaming,
                no_cache=args.no_cache,
                use_drive_api=args.use_drive_api,
                use_drive=args.use_drive,
                credentials_path=args.credentials_path, 
                drive_base_dir=args.drive_base_dir,
                headless=args.headless,
                skip_local_storage=args.skip_local_storage
            ):
                logger.error("Dataset processing failed")
                return 1
        
        # Train with the optimized config
        if not train_model(
            args.training_config,
            "data/processed",
            use_drive_api=args.use_drive_api,
            use_drive=args.use_drive,
            credentials_path=args.credentials_path, 
            drive_base_dir=args.drive_base_dir,
            headless=args.headless
        ):
            logger.error("Training failed")
            return 1
        logger.info(f"Quick Stack training completed successfully")
        return
    
    # Setup Google Drive API if requested
    drive_api = None
    directory_ids = None
    
    if args.use_drive_api:
        logger.info("Initializing Google Drive API...")
        
        # Ensure drive_base_dir is a proper string
        if args.drive_base_dir is True or args.drive_base_dir is None:
            args.drive_base_dir = "DeepseekCoder"
            logger.warning(f"Invalid drive_base_dir, using default: {args.drive_base_dir}")
            
        drive_api = initialize_drive_api(args.credentials_path, headless=args.headless)
        
        if drive_api and drive_api.authenticated:
            logger.info(f"Setting up directories in Google Drive under {args.drive_base_dir}")
            try:
                directory_ids = setup_drive_directories(drive_api, args.drive_base_dir)
                
                if not directory_ids:
                    logger.error("Failed to set up Google Drive directories")
                    args.use_drive_api = False
            except Exception as e:
                logger.error(f"Error setting up Google Drive directories: {e}")
                args.use_drive_api = False
        else:
            logger.error("Failed to authenticate with Google Drive API")
            args.use_drive_api = False
    
    # Get training configuration
    training_config = {}
    if os.path.exists(args.training_config):
        with open(args.training_config, 'r') as f:
            training_config = json.load(f)
    
    # Run the pipeline based on the mode
    if args.mode == "all" or args.mode == "process":
        if not process_datasets(
            args.dataset_config,
            datasets=args.datasets,
            streaming=args.streaming,
            no_cache=args.no_cache,
            use_drive_api=args.use_drive_api,
            use_drive=args.use_drive,
            credentials_path=args.credentials_path, 
            drive_base_dir=args.drive_base_dir,
            headless=args.headless,
            skip_local_storage=args.skip_local_storage
        ):
            logger.error("Dataset processing failed")
            return 1
    
    if args.mode == "all" or args.mode == "train":
        if not train_model(
            args.training_config,
            "data/processed",
            use_drive_api=args.use_drive_api,
            use_drive=args.use_drive,
            credentials_path=args.credentials_path, 
            drive_base_dir=args.drive_base_dir,
            headless=args.headless
        ):
            logger.error("Training failed")
            return 1
    
    if args.mode == "all" or args.mode == "evaluate":
        # Make sure required args are provided
        if not args.model_path:
            logger.error("--model_path is required for evaluate mode")
            return 1
        
        # If we're evaluating an adapter, we need the base model too
        is_adapter = "adapter" in args.model_path or "lora" in args.model_path.lower()
        if is_adapter and not args.base_model:
            logger.error("--base_model is required when evaluating an adapter")
            return 1
            
        base_model = args.base_model or "deepseek-ai/deepseek-coder-6.7b-base"
        
        if not evaluate_model(
            args.model_path,
            base_model,
            use_drive_api=args.use_drive_api,
            use_drive=args.use_drive,
            credentials_path=args.credentials_path, 
            drive_base_dir=args.drive_base_dir,
            headless=args.headless
        ):
            logger.error("Evaluation failed")
            return 1
    
    if args.mode == "all" or args.mode == "visualize":
        # Find the most recent training log
        training_logs = Path("logs").glob("*.log")
        try:
            training_log = max(training_logs, key=os.path.getmtime)
            training_log = str(training_log)
        except (ValueError, FileNotFoundError):
            training_log = None
            logger.warning("No training logs found, skipping visualization")
        
        if training_log and not visualize_results(
            training_log,
            "results", 
            use_drive_api=args.use_drive_api,
            use_drive=args.use_drive,
            credentials_path=args.credentials_path, 
            drive_base_dir=args.drive_base_dir,
            headless=args.headless
        ):
            logger.error("Visualization failed")
            return 1
    
    # If using Google Drive API, upload results
    if args.use_drive_api and drive_api and directory_ids:
        if args.mode in ["all", "train", "quick-stack"]:
            # Upload model files
            model_dir = training_config.get("training", {}).get("output_dir", "models/deepseek-coder-finetune")
            if os.path.exists(model_dir):
                logger.info(f"Uploading model files to Google Drive from {model_dir}")
                if "models" in directory_ids:
                    save_to_drive(drive_api, model_dir, directory_ids["models"])
        
        if args.mode in ["all", "evaluate"]:
            # Upload results
            if os.path.exists("results"):
                logger.info("Uploading evaluation results to Google Drive")
                if "results" in directory_ids:
                    save_to_drive(drive_api, "results", directory_ids["results"])
        
        if args.mode in ["all", "visualize"]:
            # Upload visualizations
            if os.path.exists("visualizations"):
                logger.info("Uploading visualizations to Google Drive")
                if "visualizations" in directory_ids:
                    save_to_drive(drive_api, "visualizations", directory_ids["visualizations"])
    
    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    main() 