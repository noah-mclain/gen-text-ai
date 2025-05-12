#!/usr/bin/env python3
"""
Checkpoint Manager

Handles saving, loading, and syncing checkpoints with Google Drive.
"""

import os
import re
import json
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Union, Callable

import torch
from transformers import PreTrainedModel, Trainer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CheckpointManager:
    """
    Manages model checkpoints, including saving, loading, and syncing with Google Drive.
    """
    
    def __init__(
        self, 
        output_dir: str, 
        model_name: str,
        use_drive: bool = False,
        is_text_model: bool = False,
        max_checkpoints: int = 3,
        checkpoint_criteria: str = "loss"
    ):
        """
        Initialize the checkpoint manager.
        
        Args:
            output_dir: Directory where checkpoints are saved
            model_name: Name of the model
            use_drive: Whether to sync checkpoints with Google Drive
            is_text_model: Whether this is a text generation model (vs code model)
            max_checkpoints: Maximum number of checkpoints to keep
            checkpoint_criteria: Criteria for keeping checkpoints ('loss', 'step', etc.)
        """
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.use_drive = use_drive
        self.is_text_model = is_text_model
        self.max_checkpoints = max_checkpoints
        self.checkpoint_criteria = checkpoint_criteria
        self.checkpoints = []
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Log checkpoint manager configuration
        logger.info(f"Initialized CheckpointManager with output_dir={output_dir}, model_name={model_name}")
        logger.info(f"Drive sync: {use_drive}, Max checkpoints: {max_checkpoints}, Criteria: {checkpoint_criteria}")
        
        # If using drive, import drive_sync and set up
        if self.use_drive:
            try:
                import sys
                sys.path.append(str(Path(__file__).parent.parent))
                from utils.drive_sync import sync_to_drive, sync_from_drive
                self.sync_to_drive = sync_to_drive
                self.sync_from_drive = sync_from_drive
                logger.info("Google Drive sync enabled for checkpoints")
            except ImportError:
                logger.warning("Failed to import drive_sync module. Drive sync will be disabled.")
                self.use_drive = False
        
        # If use_drive is True, try to load checkpoint info from Drive
        if self.use_drive:
            self._sync_checkpoint_info_from_drive()
        else:
            # Load checkpoint info from local storage
            self._load_local_checkpoint_info()
    
    def _get_drive_folder_key(self) -> str:
        """Get the appropriate Drive folder key based on model type."""
        return "text_checkpoints" if self.is_text_model else "checkpoints"
    
    def _sync_checkpoint_info_from_drive(self):
        """Try to load checkpoint info from Google Drive."""
        try:
            # Create a temporary file to download checkpoint info
            temp_info_path = os.path.join(self.output_dir, f"{self.model_name}_checkpoints_info_temp.json")
            
            # Try to download the checkpoint info file
            success = self.sync_from_drive(
                f"{self._get_drive_folder_key()}/{self.model_name}_checkpoints_info.json", 
                temp_info_path
            )
            
            if success and os.path.exists(temp_info_path):
                # Load the info
                with open(temp_info_path, 'r') as f:
                    self.checkpoints = json.load(f)
                logger.info(f"Loaded checkpoint info from Drive with {len(self.checkpoints)} checkpoints")
                
                # Clean up temp file
                os.remove(temp_info_path)
            else:
                logger.info("No checkpoint info found on Drive, starting fresh")
                self.checkpoints = []
                
                # Initialize with local checkpoints if they exist
                self._load_local_checkpoint_info()
                
                # Save info to Drive
                self._save_checkpoint_info()
        except Exception as e:
            logger.warning(f"Error syncing checkpoint info from Drive: {e}")
            # Fall back to local info
            self._load_local_checkpoint_info()
    
    def _load_local_checkpoint_info(self):
        """Load checkpoint info from local storage."""
        info_path = os.path.join(self.output_dir, f"{self.model_name}_checkpoints_info.json")
        
        if os.path.exists(info_path):
            try:
                with open(info_path, 'r') as f:
                    self.checkpoints = json.load(f)
                logger.info(f"Loaded local checkpoint info with {len(self.checkpoints)} checkpoints")
            except Exception as e:
                logger.warning(f"Error loading local checkpoint info: {e}")
                self.checkpoints = []
                
                # Try to discover checkpoints in the output directory
                self._discover_checkpoints()
        else:
            logger.info("No local checkpoint info found, discovering checkpoints")
            self.checkpoints = []
            self._discover_checkpoints()
    
    def _discover_checkpoints(self):
        """Discover existing checkpoints in the output directory."""
        try:
            # Look for checkpoint directories with the pattern checkpoint-{step}
            checkpoint_dirs = list(self.output_dir.glob("checkpoint-*"))
            
            for checkpoint_dir in checkpoint_dirs:
                if checkpoint_dir.is_dir():
                    # Extract step number
                    match = re.search(r'checkpoint-(\d+)', str(checkpoint_dir))
                    if match:
                        step = int(match.group(1))
                        
                        # Try to find the checkpoint's metadata
                        trainer_state_path = checkpoint_dir / "trainer_state.json"
                        if trainer_state_path.exists():
                            try:
                                with open(trainer_state_path, 'r') as f:
                                    trainer_state = json.load(f)
                                    
                                # Extract relevant info
                                checkpoint_info = {
                                    "step": step,
                                    "path": str(checkpoint_dir),
                                    "timestamp": datetime.now().isoformat(),
                                    "metrics": {}
                                }
                                
                                # Extract metrics from trainer state
                                if "best_metric" in trainer_state:
                                    checkpoint_info["metrics"]["best_metric"] = trainer_state["best_metric"]
                                
                                if "log_history" in trainer_state and trainer_state["log_history"]:
                                    last_log = trainer_state["log_history"][-1]
                                    for key, value in last_log.items():
                                        if key not in ["epoch", "step"]:
                                            checkpoint_info["metrics"][key] = value
                                
                                self.checkpoints.append(checkpoint_info)
                                logger.info(f"Discovered checkpoint: step {step}")
                                
                            except Exception as e:
                                logger.warning(f"Error loading trainer state for {checkpoint_dir}: {e}")
                                
                                # Add basic info without metrics
                                self.checkpoints.append({
                                    "step": step,
                                    "path": str(checkpoint_dir),
                                    "timestamp": datetime.now().isoformat(),
                                    "metrics": {}
                                })
            
            # Sort checkpoints by step
            self.checkpoints.sort(key=lambda x: x["step"])
            logger.info(f"Discovered {len(self.checkpoints)} checkpoints")
            
            # Save the checkpoint info
            self._save_checkpoint_info()
                                
        except Exception as e:
            logger.warning(f"Error discovering checkpoints: {e}")
    
    def _save_checkpoint_info(self):
        """Save checkpoint info to disk and optionally to Drive."""
        try:
            # Save locally
            info_path = os.path.join(self.output_dir, f"{self.model_name}_checkpoints_info.json")
            with open(info_path, 'w') as f:
                json.dump(self.checkpoints, f, indent=2)
            
            # Sync to Drive if enabled
            if self.use_drive:
                self.sync_to_drive(
                    info_path, 
                    f"{self._get_drive_folder_key()}/{self.model_name}_checkpoints_info.json", 
                    update_only=False
                )
        except Exception as e:
            logger.warning(f"Error saving checkpoint info: {e}")
    
    def save_checkpoint(self, trainer: Trainer, metrics: Optional[Dict[str, float]] = None) -> str:
        """
        Save a checkpoint using the Trainer's save_model method and sync to Drive if enabled.
        
        Args:
            trainer: Transformers Trainer instance
            metrics: Optional metrics to store with the checkpoint
            
        Returns:
            Path to the saved checkpoint
        """
        # Get the current global step
        current_step = trainer.state.global_step
        
        # Create checkpoint directory
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{current_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save the model
        trainer.save_model(checkpoint_dir)
        
        # Save optimizer and scheduler states
        trainer.save_state()
        
        # Create checkpoint info
        checkpoint_info = {
            "step": current_step,
            "path": checkpoint_dir,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics or {}
        }
        
        # Add to list of checkpoints
        self.checkpoints.append(checkpoint_info)
        
        # Prune checkpoints if needed
        self._prune_checkpoints()
        
        # Save checkpoint info
        self._save_checkpoint_info()
        
        # Sync to Drive if enabled
        if self.use_drive:
            logger.info(f"Syncing checkpoint {current_step} to Drive")
            self.sync_to_drive(
                checkpoint_dir, 
                f"{self._get_drive_folder_key()}/{self.model_name}/checkpoint-{current_step}", 
                update_only=False
            )
        
        return checkpoint_dir
    
    def _prune_checkpoints(self):
        """Remove old checkpoints to keep only the best ones."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # Sort checkpoints based on criteria
        if self.checkpoint_criteria == "loss":
            # Lower is better for loss
            key_func = lambda x: x["metrics"].get("loss", float("inf"))
            reverse = False
        elif self.checkpoint_criteria == "eval_loss":
            key_func = lambda x: x["metrics"].get("eval_loss", float("inf"))
            reverse = False
        elif self.checkpoint_criteria == "best_metric":
            # Higher is better for most metrics
            key_func = lambda x: x["metrics"].get("best_metric", 0)
            reverse = True
        else:
            # Default to keeping the most recent checkpoints
            key_func = lambda x: x["step"]
            reverse = True
        
        # Sort checkpoints by criteria
        sorted_checkpoints = sorted(self.checkpoints, key=key_func, reverse=reverse)
        
        # Decide which to keep and which to remove
        to_keep = sorted_checkpoints[:self.max_checkpoints]
        to_remove = sorted_checkpoints[self.max_checkpoints:]
        
        # Extract steps to keep for easier lookup
        keep_steps = set(checkpoint["step"] for checkpoint in to_keep)
        
        # Update checkpoints list to only include ones we're keeping
        self.checkpoints = [c for c in self.checkpoints if c["step"] in keep_steps]
        
        # Remove directories for checkpoints we're not keeping
        for checkpoint in to_remove:
            try:
                checkpoint_path = checkpoint["path"]
                if os.path.exists(checkpoint_path):
                    shutil.rmtree(checkpoint_path)
                    logger.info(f"Removed checkpoint: {checkpoint_path}")
                    
                # If syncing with Drive, also remove from Drive
                if self.use_drive:
                    # Note: We're not actually deleting from Drive to avoid API complexity
                    # Just log that we're not syncing it anymore
                    step = checkpoint["step"]
                    logger.info(f"Checkpoint {step} will no longer be synced to Drive")
            except Exception as e:
                logger.warning(f"Error removing checkpoint {checkpoint['path']}: {e}")
    
    def load_best_checkpoint(self, model_class: Optional[type] = None) -> Union[PreTrainedModel, str]:
        """
        Load the best checkpoint based on checkpoint_criteria.
        
        Args:
            model_class: Optional class to use for loading the model
            
        Returns:
            Either the loaded model or the path to the best checkpoint
        """
        if not self.checkpoints:
            logger.warning("No checkpoints available to load")
            return None
        
        # Sort checkpoints based on criteria
        if self.checkpoint_criteria == "loss":
            # Lower is better for loss
            key_func = lambda x: x["metrics"].get("loss", float("inf"))
            reverse = False
        elif self.checkpoint_criteria == "eval_loss":
            key_func = lambda x: x["metrics"].get("eval_loss", float("inf"))
            reverse = False
        elif self.checkpoint_criteria == "best_metric":
            # Higher is better for most metrics
            key_func = lambda x: x["metrics"].get("best_metric", 0)
            reverse = True
        else:
            # Default to loading the most recent checkpoint
            key_func = lambda x: x["step"]
            reverse = True
        
        # Find the best checkpoint
        sorted_checkpoints = sorted(self.checkpoints, key=key_func, reverse=reverse)
        best_checkpoint = sorted_checkpoints[0]
        logger.info(f"Selected best checkpoint: step {best_checkpoint['step']}")
        
        # If the checkpoint is not local but we're using Drive, download it
        best_path = best_checkpoint["path"]
        if not os.path.exists(best_path) and self.use_drive:
            step = best_checkpoint["step"]
            logger.info(f"Best checkpoint not found locally, downloading from Drive: step {step}")
            
            # Make sure local directory exists
            os.makedirs(best_path, exist_ok=True)
            
            # Download from Drive
            success = self.sync_from_drive(
                f"{self._get_drive_folder_key()}/{self.model_name}/checkpoint-{step}", 
                best_path
            )
            
            if not success:
                logger.error(f"Failed to download checkpoint from Drive: step {step}")
                # Try to find a local checkpoint instead
                for checkpoint in sorted_checkpoints[1:]:
                    if os.path.exists(checkpoint["path"]):
                        best_checkpoint = checkpoint
                        best_path = best_checkpoint["path"]
                        logger.info(f"Falling back to local checkpoint: step {best_checkpoint['step']}")
                        break
        
        # If model_class is provided, load and return the model
        if model_class and os.path.exists(best_path):
            try:
                model = model_class.from_pretrained(best_path)
                logger.info(f"Loaded model from checkpoint: {best_path}")
                return model
            except Exception as e:
                logger.error(f"Error loading model from checkpoint: {e}")
                return best_path
        
        # Otherwise just return the path
        return best_path
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the path to the latest checkpoint."""
        if not self.checkpoints:
            return None
        
        # Sort by step (descending)
        sorted_checkpoints = sorted(self.checkpoints, key=lambda x: x["step"], reverse=True)
        return sorted_checkpoints[0]["path"]
    
    def sync_all_checkpoints_to_drive(self):
        """Sync all checkpoints to Google Drive."""
        if not self.use_drive:
            logger.warning("Drive sync not enabled")
            return
        
        logger.info("Syncing all checkpoints to Drive")
        
        for checkpoint in self.checkpoints:
            try:
                checkpoint_path = checkpoint["path"]
                step = checkpoint["step"]
                
                if os.path.exists(checkpoint_path):
                    logger.info(f"Syncing checkpoint {step} to Drive")
                    self.sync_to_drive(
                        checkpoint_path,
                        f"{self._get_drive_folder_key()}/{self.model_name}/checkpoint-{step}",
                        update_only=False
                    )
                else:
                    logger.warning(f"Checkpoint {step} not found locally, skipping sync")
            except Exception as e:
                logger.warning(f"Error syncing checkpoint {checkpoint['step']} to Drive: {e}")
        
        # Also sync the checkpoint info
        self._save_checkpoint_info()
    
    def download_all_checkpoints_from_drive(self):
        """Download all checkpoints from Google Drive."""
        if not self.use_drive:
            logger.warning("Drive sync not enabled")
            return
        
        logger.info("Downloading all checkpoints from Drive")
        
        # First, sync the checkpoint info
        self._sync_checkpoint_info_from_drive()
        
        # Download each checkpoint
        for checkpoint in self.checkpoints:
            try:
                step = checkpoint["step"]
                checkpoint_path = checkpoint["path"]
                
                # Create directory if it doesn't exist
                os.makedirs(checkpoint_path, exist_ok=True)
                
                logger.info(f"Downloading checkpoint {step} from Drive")
                self.sync_from_drive(
                    f"{self._get_drive_folder_key()}/{self.model_name}/checkpoint-{step}",
                    checkpoint_path
                )
            except Exception as e:
                logger.warning(f"Error downloading checkpoint {checkpoint['step']} from Drive: {e}")

def create_checkpoint_callback(
    checkpoint_manager: CheckpointManager, 
    save_steps: int = 1000,
    eval_steps: Optional[int] = None,
    use_eval_metrics: bool = True
) -> Callable:
    """
    Create a callback function for Trainer to handle checkpoints.
    
    Args:
        checkpoint_manager: CheckpointManager instance
        save_steps: Save a checkpoint every N steps
        eval_steps: How often evaluation is performed (if different from save_steps)
        use_eval_metrics: Whether to use evaluation metrics for checkpointing
        
    Returns:
        Callback function for the Trainer
    """
    
    def checkpoint_callback(args, state, control, model=None, metrics=None, **kwargs):
        """Callback to handle checkpoint saving and syncing."""
        if state.is_local_process_zero:
            # Check if we should save a checkpoint (on save_steps or after evaluation)
            save_on_steps = control.should_save and (state.global_step % save_steps == 0)
            save_after_eval = use_eval_metrics and metrics is not None
            
            if save_on_steps or save_after_eval:
                logger.info(f"Saving checkpoint at step {state.global_step}")
                trainer = kwargs.get("trainer", None)
                if trainer:
                    checkpoint_manager.save_checkpoint(trainer, metrics)
                    # Skip the default checkpoint saving if we saved one
                    control.should_save = False
                else:
                    logger.warning("Trainer not available in callback, using built-in checkpointing")
        
        return control
    
    return checkpoint_callback 