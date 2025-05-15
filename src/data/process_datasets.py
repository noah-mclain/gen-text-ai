#!/usr/bin/env python3
import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
import time
import datetime
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to Python path to handle both local and Paperspace environments
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    logger.info(f"Added project root to Python path: {project_root}")

# Try to import preprocessing module
try:
    from .preprocessing import DataPreprocessor
    logger.info("Successfully imported DataPreprocessor using relative import")
except (ImportError, ValueError):
    try:
        # If relative import fails, try absolute import
        from src.data.preprocessing import DataPreprocessor
        logger.info("Successfully imported DataPreprocessor using absolute import")
    except ImportError:
        try:
            # Try with direct import as last resort
            sys.path.append(os.path.dirname(current_file))
            from preprocessing import DataPreprocessor
            logger.info("Successfully imported DataPreprocessor using direct import")
        except ImportError as e:
            logger.error(f"Failed to import DataPreprocessor: {e}")
            raise

# Try multiple import paths to handle reorganized structure
drive_utils_imported = False

# Try direct import from src.utils first (main implementation)
try:
    from src.utils.google_drive_manager import sync_to_drive, configure_sync_method
    logger.info("Successfully imported Drive manager from src.utils.google_drive_manager")
    drive_utils_imported = True
except ImportError:
    pass

# If not imported yet, try scripts.google_drive path 
if not drive_utils_imported:
    try:
        from scripts.google_drive.google_drive_manager import sync_to_drive, configure_sync_method
        logger.info("Successfully imported Drive manager from scripts.google_drive.google_drive_manager")
        drive_utils_imported = True
    except ImportError:
        pass

# If still not imported, try direct import from utils dir
if not drive_utils_imported:
    try:
        utils_path = os.path.join(project_root, 'src', 'utils')
        if utils_path not in sys.path:
            sys.path.append(utils_path)
        from google_drive_manager import sync_to_drive, configure_sync_method
        logger.info("Successfully imported Drive manager from utils path")
        drive_utils_imported = True
    except ImportError:
        pass

# If all imports failed, provide dummy functions
if not drive_utils_imported:
    logger.error("CRITICAL: Could not import Google Drive functions. Syncing will not work.")
    # Provide dummy functions to prevent crashes
    def sync_to_drive(*args, **kwargs):
        logger.error("Drive sync function not available - no files will be synced")
        return False
            
    def configure_sync_method(*args, **kwargs):
        logger.error("Drive sync configuration not available")
        return None

# Try to set HF token from credentials
try:
    # Try different import paths for set_hf_token
    hf_token_set = False
    
    # First try src.utils path (main implementation)
    try:
        from src.utils.set_hf_token import set_hf_token
        set_hf_token()
        logger.info("Set HF_TOKEN from credentials using src.utils path")
        hf_token_set = True
    except ImportError:
        pass
    
    # If not set yet, try scripts.utilities path
    if not hf_token_set:
        try:
            from scripts.utilities.set_hf_token import set_hf_token
            set_hf_token()
            logger.info("Set HF_TOKEN from credentials using scripts.utilities path")
            hf_token_set = True
        except ImportError:
            pass
    
    # If still not set, try direct import
    if not hf_token_set:
        try:
            utils_path = os.path.join(project_root, 'src', 'utils')
            if utils_path not in sys.path:
                sys.path.append(utils_path)
            from set_hf_token import set_hf_token
            set_hf_token()
            logger.info("Set HF_TOKEN from credentials using direct import")
        except ImportError:
            logger.warning("Could not import set_hf_token script. HF_TOKEN may not be set.")
except Exception as e:
    logger.warning(f"Error setting HF_TOKEN: {e}")

def process_datasets(config_path: str, datasets: Optional[List[str]] = None, 
                    streaming: bool = False, no_cache: bool = False, 
                    use_drive_api: bool = False, credentials_path: Optional[str] = None,
                    drive_base_dir: Optional[str] = None, headless: bool = False,
                    skip_local_storage: bool = False, verbose: bool = False,
                    use_preprocessed_folder: bool = False, temp_dir: Optional[str] = None):
    """Process datasets according to the configuration."""
    
    # Set logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check for existing HF_TOKEN environment variable first
    if os.environ.get("HF_TOKEN"):
        logger.info("Found existing HF_TOKEN in environment variables")
    # Only look in credentials if the token isn't already set
    elif credentials_path and os.path.exists(credentials_path):
        try:
            with open(credentials_path, 'r') as f:
                credentials = json.load(f)

                # Look for Hugging Face token in credentials
                if "huggingface" in credentials:
                    os.environ["HF_TOKEN"] = credentials["huggingface"].get("token", "")
                    logger.info("Set HF_TOKEN environment variable from credentials")
                elif "hf_token" in credentials:
                    os.environ["HF_TOKEN"] = credentials["hf_token"]
                    logger.info("Set HF_TOKEN environment variable from credentials")

                # Also look in other possible locations in the JSON structure
                if "api_keys" in credentials and "huggingface" in credentials["api_keys"]:
                    os.environ["HF_TOKEN"] = credentials["api_keys"]["huggingface"]
                    logger.info("Set HF_TOKEN environment variable from api_keys in credentials")
        except Exception as e:
            logger.error(f"Error loading credentials: {str(e)}")

    # Try to load dataset configuration from multiple possible locations
    dataset_config = None
    config_paths_to_try = [
        config_path,  # Try the provided path first
        os.path.join(project_root, config_path),  # Try with project root
        os.path.join(os.getcwd(), config_path),  # Try with current working directory
        os.path.abspath(config_path),  # Try absolute path
        os.path.join('/notebooks', config_path)  # Try Paperspace notebooks directory
    ]
    
    for path in config_paths_to_try:
        try:
            if os.path.exists(path):
                logger.info(f"Found configuration file at: {path}")
                with open(path, 'r') as f:
                    dataset_config = json.load(f)
                logger.info(f"Loaded dataset configuration with {len(dataset_config)} datasets")
                break
        except Exception as e:
            logger.debug(f"Failed to load configuration from {path}: {e}")
    
    if dataset_config is None:
        # If still not found, try to search for the file
        logger.error(f"Could not find configuration file at any of these locations: {config_paths_to_try}")
        # Try to find any dataset_config.json files in the project
        import glob
        possible_configs = glob.glob(os.path.join(project_root, "**", "dataset_config.json"), recursive=True)
        if possible_configs:
            logger.info(f"Found possible configuration files: {possible_configs}")
            try:
                with open(possible_configs[0], 'r') as f:
                    dataset_config = json.load(f)
                logger.info(f"Loaded dataset configuration from {possible_configs[0]} with {len(dataset_config)} datasets")
            except Exception as e:
                logger.error(f"Error loading dataset configuration: {str(e)}")
                return
        else:
            logger.error(f"Error loading dataset configuration: No configuration file found")
            return

    # Filter out explicitly disabled datasets
    filtered_config = {}
    for dataset_name, config in dataset_config.items():
        # Skip disabled datasets
        if isinstance(config, dict) and config.get("enabled") is False:
            logger.info(f"Skipping disabled dataset: {dataset_name}")
            continue
        filtered_config[dataset_name] = config
    dataset_config = filtered_config
    logger.info(f"Found {len(dataset_config)} enabled datasets")

    # Filter datasets if specified
    if datasets:
        filtered_config = {}
        for dataset_name in datasets:
            if dataset_name in dataset_config:
                filtered_config[dataset_name] = dataset_config[dataset_name]
                logger.info(f"Including specified dataset: {dataset_name}")
            else:
                logger.warning(f"Dataset {dataset_name} not found in configuration")
        dataset_config = filtered_config
        logger.info(f"Processing {len(dataset_config)} datasets specified by user")

    # Set streaming option for each dataset
    for dataset_name in dataset_config:
        dataset_config[dataset_name]["streaming"] = streaming
        dataset_config[dataset_name]["use_cache"] = not no_cache
        logger.debug(f"Dataset {dataset_name} configuration: streaming={streaming}, use_cache={not no_cache}")

    # Initialize preprocessor
    preprocessor = DataPreprocessor()

    # Process datasets
    # Try different paths for output directory to handle different environments
    if os.path.exists(os.path.join(project_root, "data", "processed")):
        output_dir = os.path.join(project_root, "data", "processed")
    elif os.path.exists(os.path.join(os.path.dirname(project_root), "data", "processed")):
        output_dir = os.path.join(os.path.dirname(project_root), "data", "processed")
    else:
        # Create a path if it doesn't exist yet
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/processed"))
        os.makedirs(output_dir, exist_ok=True)
        
    logger.info(f"Using output directory: {output_dir}")
    
    # Make sure the output directory has write permissions
    test_file_path = os.path.join(output_dir, ".write_test")
    try:
        with open(test_file_path, "w") as f:
            f.write("test")
        os.remove(test_file_path)
        logger.info(f"Confirmed write access to output directory: {output_dir}")
    except Exception as e:
        logger.error(f"Cannot write to output directory {output_dir}: {e}")
        logger.error("Please check directory permissions")
        return

    # Configure Google Drive syncing
    if use_drive_api:
        # Configure whether to use rclone or the Google Drive API
        # By default, use rclone if available since it's more reliable
        try:
            # Check if rclone is available
            import subprocess
            result = subprocess.run(
                ["rclone", "version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            use_rclone = result.returncode == 0
        except Exception:
            use_rclone = False

        # Ensure drive_base_dir is a string and not a boolean or None
        if drive_base_dir is None or drive_base_dir is True:
            # Set a default value if None or True
            drive_base_dir = "DeepseekCoder"
            logger.warning(f"Invalid drive_base_dir provided, using default: {drive_base_dir}")

        # Configure the drive sync method
        configure_sync_method(use_rclone=use_rclone, base_dir=drive_base_dir)
        logger.info(f"Configured Drive sync using {'rclone' if use_rclone else 'Google Drive API'}")

    # If skipping local storage, create a temporary directory for processing
    if skip_local_storage and (use_drive_api or drive_utils_imported):
        if temp_dir:
            # Use the provided visible temporary directory
            logger.info(f"Using provided temporary directory: {temp_dir}")
            os.makedirs(temp_dir, exist_ok=True)
            output_dir = os.path.join(temp_dir, "processed")
            os.makedirs(output_dir, exist_ok=True)
        else:
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix="dataset_processing_")
            logger.info(f"Using temporary directory for processing: {temp_dir}")
            output_dir = os.path.join(temp_dir, "processed")
            os.makedirs(output_dir, exist_ok=True)
    
    # Process the datasets - measure time for logging
    start_time = time.time()
    logger.info(f"Starting dataset processing at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    processed_datasets = preprocessor.load_and_process_all_datasets(dataset_config, output_dir)
    end_time = time.time()
    processing_time = end_time - start_time
    logger.info(f"Dataset processing completed in {processing_time:.2f} seconds")

    # Check if any datasets were processed
    if not processed_datasets:
        logger.warning("No datasets were processed successfully")
    else:
        logger.info(f"Successfully processed {len(processed_datasets)} datasets")
        for name in processed_datasets.keys():
            logger.info(f"  - {name}")

    # Upload processed datasets if using Drive API
    if use_drive_api or drive_utils_imported:
        logger.info("Syncing processed datasets to Google Drive")
        try:
            # Sync the processed datasets to Drive
            # Determine which folder to use based on the use_preprocessed_folder flag
            drive_folder = "preprocessed" if use_preprocessed_folder else "datasets"
            if drive_base_dir:
                if drive_base_dir != "datasets" and drive_base_dir != "preprocessed":
                    drive_folder = drive_base_dir
            
            logger.info(f"Using Google Drive folder: {drive_folder}")
            
            # Track sync successes to know when it's safe to delete temp directory
            sync_successes = 0
            sync_failures = 0
            
            for dataset_name in processed_datasets:
                # Check both possible naming patterns for dataset directories
                dataset_paths = [
                    os.path.join(output_dir, f"{dataset_name}_processed"),
                    os.path.join(output_dir, dataset_name),
                    os.path.join(output_dir, f"{dataset_name}")
                ]
                
                # Find the actual path that exists
                dataset_path = None
                for path in dataset_paths:
                    if os.path.exists(path):
                        dataset_path = path
                        break
                
                if dataset_path:
                    # Log directory details
                    try:
                        num_files = sum([len(files) for _, _, files in os.walk(dataset_path)])
                        size_mb = sum(os.path.getsize(os.path.join(root, file)) for root, _, files in os.walk(dataset_path) for file in files) / (1024 * 1024)
                        logger.info(f"Syncing dataset {dataset_name} ({num_files} files, {size_mb:.2f} MB) to Drive folder {drive_folder}")
                    except Exception as e:
                        logger.warning(f"Error calculating directory stats: {e}")
                        logger.info(f"Syncing dataset {dataset_name} to Drive folder {drive_folder}")
                    
                    # Do not delete source yet - we'll delete everything at once at the end if skip_local_storage is True
                    success = sync_to_drive(
                        dataset_path, 
                        drive_folder, 
                        delete_source=False,  # We'll handle deletion manually
                        update_only=False     # We want to ensure all files are synced
                    )

                    if success:
                        logger.info(f"Successfully synced dataset {dataset_name} to Drive")
                        sync_successes += 1
                    else:
                        logger.error(f"Failed to sync dataset {dataset_name} to Drive")
                        sync_failures += 1
                else:
                    logger.warning(f"Could not find directory for dataset {dataset_name}")
                    # Detailed debug info to help locate the issue
                    logger.debug(f"Tried paths: {dataset_paths}")
                    logger.debug(f"Current output_dir: {output_dir}")
                    logger.debug(f"Directory contents: {os.listdir(output_dir) if os.path.exists(output_dir) else 'directory does not exist'}")
            
            # Now it's safe to delete the temp directory if all syncs were successful
            if skip_local_storage and 'temp_dir' in locals() and temp_dir:
                if sync_failures == 0 and sync_successes > 0:
                    import shutil
                    try:
                        shutil.rmtree(temp_dir)
                        logger.info(f"Cleaned up temporary directory: {temp_dir}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up temporary directory: {e}")
                else:
                    logger.warning(f"Not cleaning up temporary directory due to {sync_failures} sync failures")
                    logger.info(f"Temporary directory location: {temp_dir}")
        except Exception as e:
            logger.error(f"Error syncing datasets to Google Drive: {str(e)}")
            if 'temp_dir' in locals() and temp_dir:
                logger.info(f"Not cleaning temporary directory due to error: {temp_dir}")

    logger.info(f"Processed {len(processed_datasets)} datasets")
    return processed_datasets

def main():
    parser = argparse.ArgumentParser(description="Process datasets for fine-tuning deepseek-coder")
    parser.add_argument("--config", type=str, default="../../config/dataset_config.json", 
                        help="Path to dataset configuration file")
    parser.add_argument("--save_dir", type=str, default="../../data/processed",
                        help="Directory to save processed datasets")
    parser.add_argument("--tokenizer", type=str, default="deepseek-ai/deepseek-coder-6.7b-base",
                        help="Path or name of the tokenizer to use")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum token length for examples")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Optional list of dataset names to process (if not specified, all will be processed)")
    
    # Google Drive API options
    parser.add_argument("--use_drive_api", action="store_true", 
                        help="Use Google Drive API for storage (for Paperspace)")
    parser.add_argument("--use_drive", action="store_true", 
                        help="Use Google Drive with rclone (preferred)")
    parser.add_argument("--credentials_path", type=str, default="credentials.json",
                        help="Path to Google Drive API credentials JSON file")
    parser.add_argument("--drive_base_dir", type=str, default="DeepseekCoder",
                        help="Base directory on Google Drive")
    parser.add_argument("--headless", action="store_true",
                        help="Use headless authentication for environments without a browser")
    parser.add_argument("--use_preprocessed_folder", action="store_true",
                        help="Use 'preprocessed' folder instead of 'processed_data' for compatibility with training scripts")
    
    # Memory efficiency options
    parser.add_argument("--streaming", action="store_true",
                        help="Load datasets in streaming mode to save memory")
    parser.add_argument("--no_cache", action="store_true",
                        help="Disable caching for datasets to save disk space")
    parser.add_argument("--skip_local_storage", action="store_true",
                        help="Skip storing datasets locally, sync directly to Drive")
    parser.add_argument("--temp_dir", type=str, default=None,
                        help="Custom visible temporary directory for processing")
    
    args = parser.parse_args()
    
    # Auto-enable Drive API if running on Paperspace
    is_paperspace = os.path.exists("/notebooks")
    if is_paperspace and not (args.use_drive_api or args.use_drive):
        logger.info("Detected Paperspace environment. Auto-enabling Google Drive integration.")
        args.use_drive_api = True
    
    process_datasets(
        args.config, 
        args.datasets, 
        args.streaming, 
        args.no_cache,
        args.use_drive_api or args.use_drive, 
        args.credentials_path,
        args.drive_base_dir, 
        args.headless,
        args.skip_local_storage,
        verbose=False,
        use_preprocessed_folder=args.use_preprocessed_folder,
        temp_dir=args.temp_dir
    )

if __name__ == "__main__":
    main() 