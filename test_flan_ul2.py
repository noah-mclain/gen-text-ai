#!/usr/bin/env python3
"""
Test script to verify FLAN-UL2 model loading with T5 configuration
"""

import os
import torch
from transformers import AutoConfig, AutoTokenizer, T5ForConditionalGeneration

# Define model type constant
UL2_MODEL_TYPE = "t5"

def test_flan_ul2_loading():
    """Test FLAN-UL2 model loading with T5 configuration"""
    print("Testing FLAN-UL2 model loading...")
    
    model_path = "google/flan-ul2"
    
    print(f"Creating T5 config for {model_path}...")
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        model_type=UL2_MODEL_TYPE,  # Explicitly set model type to t5
    )
    
    print(f"Config created successfully: {config.model_type}")
    
    print(f"Loading tokenizer with config...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
    )
    
    print(f"Tokenizer loaded successfully")
    
    print(f"Loading a small version of the model for testing...")
    # For testing purposes, load a very small version with just 2 layers to test the architecture
    config.num_layers = 2
    config.num_decoder_layers = 2
    
    # Load just the structure, not the weights, to verify it works
    print(f"Creating model structure...")
    model = T5ForConditionalGeneration(config)
    
    print(f"Model structure created successfully!")
    print(f"Model type: {model.config.model_type}")
    print(f"Model structure: {model.__class__.__name__}")
    
    print("Test complete - if you see this message, the model structure loading works!")
    return True

if __name__ == "__main__":
    test_flan_ul2_loading() 