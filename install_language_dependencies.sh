#!/bin/bash

# Install language detection dependencies
echo "Installing language detection dependencies..."

# Install langid for language detection
pip install langid

# Install other potential dependencies for processing
pip install tqdm==4.67.1 datasets transformers

echo "Language detection dependencies installed successfully!" 