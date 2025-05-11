#!/usr/bin/env python3
import os
import argparse
import logging
import subprocess
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def process_datasets(config_path, datasets=None):
    """Process datasets for fine-tuning."""
    cmd = f"python -m src.data.process_datasets --config {config_path}"
    if datasets:
        cmd += f" --datasets {' '.join(datasets)}"
    
    return run_command(cmd, "Processing datasets")

def train_model(config_path, data_dir):
    """Train the model."""
    cmd = f"python -m src.training.train --config {config_path} --data_dir {data_dir}"
    return run_command(cmd, "Training model")

def evaluate_model(model_path, base_model, output_dir="results"):
    """Evaluate the model."""
    cmd = f"python -m src.evaluation.evaluate --model_path {model_path} --base_model {base_model} --output_dir {output_dir} --eval_humaneval --eval_mbpp"
    return run_command(cmd, "Evaluating model")

def visualize_results(training_log, results_dir, output_dir="visualizations"):
    """Visualize results."""
    cmd = f"python -m src.utils.visualize --training_log {training_log} --results_dir {results_dir} --output_dir {output_dir}"
    return run_command(cmd, "Visualizing results")

def main():
    parser = argparse.ArgumentParser(description="DeepSeek-Coder Fine-Tuning Pipeline")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["all", "process", "train", "evaluate", "visualize"],
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
    
    args = parser.parse_args()
    
    # Ensure directories exist
    ensure_directories()
    
    # Get training configuration
    training_config = {}
    if os.path.exists(args.training_config):
        with open(args.training_config, 'r') as f:
            training_config = json.load(f)
    
    # Run the pipeline based on the mode
    if args.mode in ["all", "process"]:
        if not process_datasets(args.dataset_config, args.datasets):
            logger.error("Dataset processing failed")
            if args.mode != "all":
                return
    
    if args.mode in ["all", "train"]:
        if not train_model(args.training_config, "data/processed"):
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
        
        if not evaluate_model(model_path, base_model):
            logger.error("Evaluation failed")
            if args.mode != "all":
                return
    
    if args.mode in ["all", "visualize"]:
        # For "all" mode, use the trained model's log
        training_log = os.path.join(
            training_config.get("training", {}).get("output_dir", "models/deepseek-coder-finetune"),
            "trainer_state.json"
        )
        
        if not visualize_results(training_log, "results"):
            logger.error("Visualization failed")
            return
    
    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    main() 