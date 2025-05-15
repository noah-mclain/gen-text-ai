#!/bin/bash
# Script to recover and sync previously processed datasets from temporary directories

echo "🔍 Looking for datasets in temporary directories..."

# Run the fix_dataset_sync.py script with verbose logging to find and sync datasets
python scripts/google_drive/fix_dataset_sync.py \
  --scan_temp \
  --scan_paths "/tmp" "/tmp/**/processed" "/var/tmp" "/notebooks/tmp" "/notebooks/data/processed" "temp_datasets*" "data/processed" \
  --drive_folder "datasets" \
  --verbose

# Check if dataset_processing folders exist directly in /tmp
TMP_PROCESSED_DIRS=$(find /tmp -maxdepth 3 -type d -name "processed" -o -name "*_processed" 2>/dev/null)

if [ -n "$TMP_PROCESSED_DIRS" ]; then
  echo ""
  echo "📁 Found potential dataset directories in /tmp that might need manual review:"
  echo "$TMP_PROCESSED_DIRS"
  echo ""
  echo "To manually sync these directories, run:"
  echo 'python scripts/google_drive/fix_dataset_sync.py --scan_paths "/path/to/directory" --drive_folder "datasets"'
fi

echo ""
echo "✅ Recovery process complete!"
echo ""
echo "To run future dataset processing with visible temp directories:"
echo "./process_and_sync.sh" 