#!/bin/bash
# Script to recover datasets by finding and copying JSONL files directly

echo "🔍 Deep searching for dataset files in temporary directories..."
echo "This might find datasets that other recovery methods missed."

# Make recovery script executable
chmod +x scripts/google_drive/recover_processed_datasets.py

# Run the recovery script
python scripts/google_drive/recover_processed_datasets.py \
  --recovery_dir "recovered_datasets" \
  --drive_folder "datasets"

echo ""
echo "If no datasets were found, try running these commands manually to find JSONL files:"
echo "  find /tmp -name \"*.jsonl\" -type f"
echo "  find /var/tmp -name \"*.jsonl\" -type f" 