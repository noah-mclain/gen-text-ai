#!/bin/bash
# Script to recover and sync previously processed datasets from temporary directories

echo "🔍 Looking for datasets in temporary directories..."

# Run the fix_dataset_sync.py script to find and sync datasets
python scripts/google_drive/fix_dataset_sync.py \
  --scan_temp \
  --drive_folder "datasets"

echo ""
echo "✅ Recovery process complete!"
echo ""
echo "To run future dataset processing with visible temp directories:"
echo "./process_and_sync.sh" 