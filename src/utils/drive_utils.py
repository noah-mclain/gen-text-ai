import os
import logging
import subprocess
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def mount_google_drive(mount_point: str = "/content/drive") -> bool:
    """
    Mount Google Drive using Google Colab's authentication flow.
    
    Args:
        mount_point: Path where Google Drive will be mounted
        
    Returns:
        True if mounting was successful, False otherwise
    """
    try:
        from google.colab import drive
        drive.mount(mount_point)
        logger.info(f"Google Drive mounted at {mount_point}")
        return True
    except ImportError:
        logger.warning("google.colab module not found. Are you running in Google Colab?")
        return False
    except Exception as e:
        logger.error(f"Error mounting Google Drive: {str(e)}")
        return False

def setup_drive_directories(base_dir: str) -> dict:
    """
    Create necessary directories on Google Drive.
    
    Args:
        base_dir: Base directory on Google Drive for the project
        
    Returns:
        Dictionary with paths to all created directories
    """
    directories = {
        "data": os.path.join(base_dir, "data"),
        "preprocessed": os.path.join(base_dir, "data/processed"),
        "raw": os.path.join(base_dir, "data/raw"),
        "models": os.path.join(base_dir, "models"),
        "logs": os.path.join(base_dir, "logs"),
        "results": os.path.join(base_dir, "results"),
        "visualizations": os.path.join(base_dir, "visualizations"),
        "cache": os.path.join(base_dir, "cache")
    }
    
    for name, path in directories.items():
        os.makedirs(path, exist_ok=True)
        logger.info(f"Created directory: {path}")
    
    return directories

def is_drive_mounted(mount_point: str = "/content/drive") -> bool:
    """Check if Google Drive is mounted."""
    return os.path.exists(mount_point) and os.path.ismount(mount_point)

def get_drive_path(local_path: str, drive_base: str, default_path: Optional[str] = None) -> str:
    """
    Convert a local path to a Google Drive path.
    
    Args:
        local_path: Original local path
        drive_base: Base directory on Google Drive
        default_path: Default path to use if Drive is not mounted
        
    Returns:
        Path on Google Drive if mounted, otherwise local path or default path
    """
    if is_drive_mounted():
        # Extract the relative path from the local path
        rel_path = os.path.basename(local_path)
        return os.path.join(drive_base, rel_path)
    else:
        return default_path if default_path else local_path 