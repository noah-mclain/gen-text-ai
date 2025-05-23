#!/bin/bash

# Quick launcher for The Stack training with time constraint
# This script will start training with The Stack dataset to finish by midnight

# Calculate remaining hours until midnight
current_hour=$(date +%H)
current_minute=$(date +%M)
hours_to_midnight=$((23 - current_hour))
if [ $current_minute -gt 30 ]; then
    hours_to_midnight=$((hours_to_midnight - 1))
fi

# Set a minimum of 1 hour
if [ $hours_to_midnight -lt 1 ]; then
    hours_to_midnight=1
fi

echo "Starting training with approximately $hours_to_midnight hours until midnight"
echo "Using The Stack dataset with language filters (Python, Java, JavaScript, C, C++, C#, TypeScript, HTML, SQL, TeX, Dockerfile)"
echo "Filtering for English and Arabic natural languages in comments"

# Set up Google Drive authentication first
echo "Setting up Google Drive authentication..."
python scripts/setup_google_drive.py
if [ $? -ne 0 ]; then
    # Launch without Drive integration
./scripts/process_stack_direct.sh --max-hours $hours_to_midnight
else
    # Launch with Drive integration
    ./scripts/process_stack_direct.sh --max-hours $hours_to_midnight --drive
fi

# Done
echo "Training is complete."
echo "Your model is saved in the models/ directory." 