#!/bin/bash
# Fix dataset mapping issues before training

# Set up Python path to find modules
export PYTHONPATH=$PYTHONPATH:.

echo "===== FIXING DATASET MAPPING ISSUES ====="
# First, let's list the available datasets to see what we have
python fix_dataset_mapping.py --list

# Now let's create symbolic links to make the datasets easier to find
echo "Creating symbolic links for dataset directories..."
python fix_dataset_mapping.py

# Check if we're running on Paperspace (has /notebooks directory)
if [ -d "/notebooks" ]; then
    echo "===== DETECTED PAPERSPACE ENVIRONMENT ====="
    echo "Creating dataset mapping fix in /notebooks directory"
    
    # Copy the fix script to the notebooks directory if needed
    if [ ! -f "/notebooks/fix_dataset_mapping.py" ]; then
        cp fix_dataset_mapping.py /notebooks/
    fi
    
    # Run the fix in the notebooks directory
    cd /notebooks
    python fix_dataset_mapping.py
    
    # Return to original directory
    cd -
    echo "✅ Dataset mapping fixed in Paperspace environment"
fi

echo "===== DATASET MAPPING FIX COMPLETE ====="
echo "You can now run the training script with confidence that it will find your datasets." 