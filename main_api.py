#!/usr/bin/env python3
import os
import argparse
import logging
import subprocess
import json
import datetime
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Drive API utils
from src.utils.drive_api_utils import initialize_drive_api, setup_drive_directories, save_to_drive, load_from_drive

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

def run_command(command, description=None):
    """Run a shell command and log its output."""
    if description:
        logger.info(f"Running: {description}")
    
    logger.info(f"Command: {command}")
    
    try:
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True
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
                    use_drive_api=False, credentials_path=None, drive_base_dir=None, headless=False):
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
    
    return run_command(cmd, "Processing datasets")

def train_model(config_path, data_dir, use_drive_api=False, credentials_path=None, drive_base_dir=None, headless=False):
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
    
    return run_command(cmd, "Training model")

def evaluate_model(model_path, base_model, output_dir="results", 
                 use_drive_api=False, credentials_path=None, drive_base_dir=None, headless=False):
    """Evaluate the model."""
    cmd = f"python -m src.evaluation.evaluate --model_path {model_path} --base_model {base_model} --output_dir {output_dir} --eval_humaneval --eval_mbpp --eval_ds1000"
    
    if use_drive_api:
        cmd += f" --use_drive_api --credentials_path {credentials_path} --drive_base_dir {drive_base_dir}"
        if headless:
            cmd += " --headless"
    
    return run_command(cmd, "Evaluating model")

def visualize_results(training_log, results_dir, output_dir="visualizations", 
                     use_drive_api=False, credentials_path=None, drive_base_dir=None, headless=False):
    """Visualize results."""
    cmd = f"python -m src.utils.visualize --training_log {training_log} --results_dir {results_dir} --output_dir {output_dir}"
    
    if use_drive_api:
        cmd += f" --use_drive_api --credentials_path {credentials_path} --drive_base_dir {drive_base_dir}"
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
    parser.add_argument("--credentials_path", type=str, default="credentials.json",
                        help="Path to Google Drive API credentials JSON file")
    parser.add_argument("--drive_base_dir", type=str, default="DeepseekCoder",
                        help="Base directory on Google Drive")
    parser.add_argument("--headless", action="store_true",
                        help="Use headless authentication for environments without a browser")
    
    # Quick Stack mode options
    parser.add_argument("--max-hours", type=float, default=None,
                        help="Maximum training time in hours (for quick-stack mode)")
    parser.add_argument("--auto-time", action="store_true",
                        help="Automatically calculate time until midnight (for quick-stack mode)")
    parser.add_argument("--skip-preprocessing", action="store_true",
                        help="Skip preprocessing step (for quick-stack mode)")
    
    args = parser.parse_args()
    
    # Check for HF_TOKEN environment variable
    if os.environ.get("HF_TOKEN"):
        logger.info("Using Hugging Face token from environment variables")
    else:
        logger.warning("HF_TOKEN environment variable not set. Some datasets might be inaccessible.")
        logger.warning("Set your token using: export HF_TOKEN=your_huggingface_token")
    
    # Ensure directories exist
    ensure_directories()
    
    # Special handling for quick-stack mode
    if args.mode == "quick-stack":
        logger.info("Running in quick Stack training mode")
        
        # Set max training time (auto-calculate or use provided value)
        max_hours = args.max_hours
        if args.auto_time or max_hours is None:
            max_hours = calculate_hours_until_midnight()
            logger.info(f"Auto-calculated {max_hours} hours until midnight")
            
        # Always set streaming and no_cache for quick stack mode
        args.streaming = True
        args.no_cache = True
        
        # Default to the_stack_filtered if no datasets specified
        if args.datasets is None:
            args.datasets = ["the_stack_filtered"]
            logger.info("Using the_stack_filtered dataset by default")
        
        # Optimize training config for time constraint
        optimize_training_config(args.training_config, max_hours)
        
        # Replace mode with "all" to run the full pipeline
        logger.info(f"Running optimized Stack training pipeline to complete in {max_hours} hours")
        
        # Decide whether to run process step
        if not args.skip_preprocessing:
            logger.info("Running preprocessing step for Stack dataset")
            if not process_datasets(
                args.dataset_config, 
                args.datasets, 
                streaming=args.streaming, 
                no_cache=args.no_cache,
                use_drive_api=args.use_drive_api, 
                credentials_path=args.credentials_path, 
                drive_base_dir=args.drive_base_dir,
                headless=args.headless
            ):
                logger.error("Dataset processing failed")
                return
        else:
            logger.info("Skipping preprocessing step, using direct loading")
        
        # Run training with optimized config
        logger.info("Starting training with time-optimized configuration")
        
        # Add the current directory to PYTHONPATH if not already there
        project_root = os.getcwd()
        if project_root not in os.environ.get("PYTHONPATH", ""):
            os.environ["PYTHONPATH"] = f"{project_root}:{os.environ.get('PYTHONPATH', '')}"
            logger.info(f"Added {project_root} to PYTHONPATH for module imports")
            
        if not train_model(
            args.training_config, 
            "data/processed",
            use_drive_api=args.use_drive_api, 
            credentials_path=args.credentials_path, 
            drive_base_dir=args.drive_base_dir,
            headless=args.headless
        ):
            logger.error("Training failed")
            return
        
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
    if args.mode in ["all", "process"]:
        if not process_datasets(
            args.dataset_config, 
            args.datasets, 
            streaming=args.streaming, 
            no_cache=args.no_cache,
            use_drive_api=args.use_drive_api, 
            credentials_path=args.credentials_path, 
            drive_base_dir=args.drive_base_dir,
            headless=args.headless
        ):
            logger.error("Dataset processing failed")
            if args.mode != "all":
                return
    
    if args.mode in ["all", "train"]:
        if not train_model(
            args.training_config, 
            "data/processed",
            use_drive_api=args.use_drive_api, 
            credentials_path=args.credentials_path, 
            drive_base_dir=args.drive_base_dir,
            headless=args.headless
        ):
            logger.error("Training failed")
            if args.mode != "all":
                return
    
    if args.mode in ["all", "evaluate"]:
        # For "all" mode, use the trained model path from the config
        model_path = args.model_path
        base_model = args.base_model
        
        if args.mode == "all":
            output_dir = training_config.get("training", {}).get("output_dir", "models/deepseek-coder-finetune")
            model_path = output_dir
            base_model = training_config.get("model", {}).get("base_model")
        
        if not model_path:
            logger.error("Model path is required for evaluation")
            return
        
        if not evaluate_model(
            model_path, 
            base_model,
            use_drive_api=args.use_drive_api, 
            credentials_path=args.credentials_path, 
            drive_base_dir=args.drive_base_dir,
            headless=args.headless
        ):
            logger.error("Evaluation failed")
            if args.mode != "all":
                return
    
    if args.mode in ["all", "visualize"]:
        # For "all" mode, use the trained model's log
        training_log = os.path.join(
            training_config.get("training", {}).get("output_dir", "models/deepseek-coder-finetune"),
            "trainer_state.json"
        )
        
        if not visualize_results(
            training_log, 
            "results", 
            use_drive_api=args.use_drive_api,
            credentials_path=args.credentials_path, 
            drive_base_dir=args.drive_base_dir,
            headless=args.headless
        ):
            logger.error("Visualization failed")
            return
    
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