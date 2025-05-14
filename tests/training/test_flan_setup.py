#!/usr/bin/env python3
"""
Test FLAN-UL2 model loading and verify it works with sequence-to-sequence configuration.
"""

import os
import sys
import json
import logging
import torch
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to pythonpath
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_flan_model_loading():
    """Test loading the FLAN-UL2 model as a sequence-to-sequence model."""
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
        
        logger.info("=== Testing FLAN-UL2 Model Loading ===")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "google/flan-ul2",
            trust_remote_code=True
        )
        logger.info("✅ Tokenizer loaded successfully")
        
        # Set up quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            use_nested_quantization=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Load model
        logger.info("Loading FLAN-UL2 model (this may take a while)...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-ul2",
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        logger.info("✅ Model loaded successfully as AutoModelForSeq2SeqLM")
        
        # Test model architecture
        logger.info(f"Model architecture: {model.__class__.__name__}")
        logger.info(f"Model has encoder: {hasattr(model, 'encoder')}")
        logger.info(f"Model has decoder: {hasattr(model, 'decoder')}")
        
        # Test basic tokenization and generation
        test_input = "Translate to French: Hello, how are you?"
        logger.info(f"Testing with input: '{test_input}'")
        
        input_ids = tokenizer(test_input, return_tensors="pt").input_ids.to(model.device)
        
        if torch.cuda.is_available():
            logger.info("CUDA is available, generating with GPU")
        else:
            logger.info("CUDA not available, generating with CPU")
        
        try:
            with torch.no_grad():
                outputs = model.generate(input_ids, max_length=50)
                output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                logger.info(f"Output: '{output_text}'")
                logger.info("✅ Generation successful")
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            
        # Test LoRA application for fine-tuning
        try:
            # Prepare model for k-bit training if using quantization
            model = prepare_model_for_kbit_training(model)
            
            # Configure LoRA
            lora_config = LoraConfig(
                r=32,
                lora_alpha=16,
                target_modules=["q", "k", "v", "o", "wi", "wo"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.SEQ_TO_SEQ_LM
            )
            
            # Apply LoRA
            peft_model = get_peft_model(model, lora_config)
            logger.info("✅ LoRA application successful")
            logger.info(f"Trainable parameters: {peft_model.print_trainable_parameters()}")
            
            return True
        except Exception as e:
            logger.error(f"❌ LoRA application failed: {e}")
            return False
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" FLAN-UL2 MODEL TEST ")
    print("=" * 60 + "\n")
    
    success = test_flan_model_loading()
    
    print("\n" + "=" * 60)
    print(f" TEST RESULT: {'✅ PASSED' if success else '❌ FAILED'} ")
    print("=" * 60 + "\n")
    
    sys.exit(0 if success else 1) 