# Google Drive Dataset Checking

This document explains the new Google Drive dataset checking functionality that has been implemented to avoid redundant dataset processing.

## Overview

The Google Drive Dataset Checker is a utility that:

1. Checks if datasets exist locally in your data directory
2. If not found locally, checks if they exist on Google Drive
3. Downloads available datasets from Drive rather than reprocessing
4. Only processes datasets that aren't available locally or on Drive

This provides significant time savings by avoiding redundant dataset processing, especially for large datasets like CodeSearchNet with multiple languages or The Stack.

## Implementation Details

The functionality is implemented in `src/utils/drive_dataset_checker.py` and provides:

- Functions to check dataset availability on Google Drive
- Efficient downloading of datasets when available
- Integration with training scripts for automatic optimization
- Proper tracking and progress reporting during downloads

## Usage in Training Scripts

The dataset checking functionality is now integrated into training scripts:

### A6000 Optimized Training Script

The `train_a6000_optimized.sh` script now includes dataset checking:

```bash
# Check if datasets are already available on Google Drive
echo "Checking for pre-processed datasets on Google Drive..."
# Save the list of datasets that need processing
DATASETS_TO_PROCESS=$(python -c "
from src.utils.drive_dataset_checker import prepare_datasets
import json

config_path = 'config/dataset_config.json'
available, needed, _ = prepare_datasets(config_path, skip_drive=${SKIP_DRIVE:-False})

print(','.join(needed))
")

# Only process datasets if there are any that need processing
if [ -n "$DATASETS_TO_PROCESS" ]; then
    echo "Processing datasets: $DATASETS_TO_PROCESS"
    python main_api.py \
        --mode process \
        --datasets $DATASETS_TO_PROCESS \
        --streaming \
        --no_cache \
        --dataset_config config/dataset_config.json \
        2>&1 | tee logs/dataset_processing_$(date +%Y%m%d_%H%M%S).log
else
    echo "All datasets already available. Skipping dataset processing step."
fi
```

### Direct Python API Usage

You can also use the functionality directly in your Python scripts:

```python
from src.utils.drive_dataset_checker import prepare_datasets

# Get datasets from config file
config_path = 'config/dataset_config.json'
available_datasets, needed_datasets, download_time = prepare_datasets(
    config_path,
    output_dir='data/processed',
    drive_folder='preprocessed',
    skip_drive=False
)

print(f"Datasets already available: {available_datasets}")
print(f"Datasets that need processing: {needed_datasets}")
print(f"Download time: {download_time:.2f} seconds")

# Process only the needed datasets
if needed_datasets:
    # Your dataset processing code here
    pass
```

## Benefits

This functionality provides several benefits:

1. **Time Savings**: Avoids redundant processing of large datasets
2. **Resource Efficiency**: Reduces computational load and energy usage
3. **Continuity**: Enables seamless continuation of training across sessions
4. **Collaboration**: Makes it easier to share preprocessed datasets with team members

## Error Handling

The functionality includes robust error handling:

- Graceful degradation when Google Drive is not available
- Proper authentication checking before attempting operations
- Detailed logging for troubleshooting
- Timeout handling for large dataset downloads

## Configuration

The dataset checking can be configured with the following parameters:

- `skip_drive`: Set to `True` to skip checking Drive and only use local datasets
- `drive_folder`: The folder name in Google Drive where datasets are stored (default: "preprocessed")
- `output_dir`: The local directory to save downloaded datasets (default: "data/processed")

## Integration with Other Scripts

The dataset checking functionality can be integrated into any script that processes datasets. See `train_a6000_optimized.sh` for a complete example.
