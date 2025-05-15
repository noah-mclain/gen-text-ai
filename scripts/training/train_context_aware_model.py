#!/usr/bin/env python3
"""
Train Context-Aware FLAN-UT Model

This script trains a context-aware FLAN-UT model with intent analysis
and conversation memory capabilities for enhanced text generation.

Usage:
    python scripts/training/train_context_aware_model.py \
        --config config/training/context_aware_config.json \
        --data_dir data/processed \
        --use_drive
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train context-aware FLAN-UT model")
    
    parser.add_argument("--config", type=str, default="config/training/context_aware_config.json",
                        help="Path to the training configuration file")
    
    parser.add_argument("--data_dir", type=str, default="data/processed",
                        help="Directory containing the processed datasets")
    
    parser.add_argument("--use_drive", action="store_true",
                        help="Use Google Drive for storage")
    
    parser.add_argument("--drive_base_dir", type=str, default=None,
                        help="Base directory on Google Drive")
    
    parser.add_argument("--prepare_datasets", action="store_true",
                        help="Prepare conversation datasets with context and intent before training")
    
    parser.add_argument("--dataset_paths", type=str, nargs="+",
                        help="Paths to conversation datasets to process (if prepare_datasets is True)")
    
    parser.add_argument("--pretrained_model_path", type=str, default=None,
                        help="Path to a previously fine-tuned model to use as base")
    
    return parser.parse_args()

def prepare_datasets(data_paths, output_dir):
    """
    Prepare conversation datasets with context and intent analysis.
    
    Args:
        data_paths: List of paths to conversation datasets
        output_dir: Output directory for processed datasets
    """
    try:
        from src.data.processors.context_aware_processor import context_aware_conversation_processor
    except ImportError:
        try:
            from data.processors.context_aware_processor import context_aware_conversation_processor
        except ImportError:
            logger.error("Could not import context_aware_conversation_processor. Make sure it's installed.")
            sys.exit(1)
    
    # Process each dataset
    for path in data_paths:
        logger.info(f"Processing dataset from {path} with context and intent analysis")
        
        dataset_name = Path(path).stem
        
        try:
            processed_dataset = context_aware_conversation_processor(
                dataset_path=path,
                output_dir=output_dir,
                streaming=True,
                max_samples=50000  # Adjust as needed
            )
            
            logger.info(f"Successfully processed dataset {dataset_name}")
            
        except Exception as e:
            logger.error(f"Error processing dataset {path}: {str(e)}")

def main():
    """Main function to train the context-aware model."""
    args = parse_arguments()
    
    # Prepare datasets if requested
    if args.prepare_datasets:
        if not args.dataset_paths:
            logger.error("Dataset paths must be provided when prepare_datasets is True")
            sys.exit(1)
        
        prepare_datasets(args.dataset_paths, args.data_dir)
    
    # Import the context-aware trainer
    try:
        from src.training.context_aware_trainer import ContextAwareTrainer
    except ImportError:
        try:
            from training.context_aware_trainer import ContextAwareTrainer
        except ImportError:
            logger.error("Could not import ContextAwareTrainer. Make sure it's installed.")
            sys.exit(1)
    
    # Set up the trainer
    logger.info(f"Initializing context-aware trainer with config: {args.config}")
    trainer = ContextAwareTrainer(
        config_path=args.config,
        use_drive=args.use_drive,
        drive_base_dir=args.drive_base_dir,
        pretrained_model_path=args.pretrained_model_path
    )
    
    # Train the model
    logger.info(f"Starting training with data from: {args.data_dir}")
    metrics = trainer.train(args.data_dir)
    
    # Log the final metrics
    logger.info(f"Training complete. Final metrics: {metrics}")

if __name__ == "__main__":
    main() 