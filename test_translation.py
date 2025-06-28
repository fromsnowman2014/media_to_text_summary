#!/usr/bin/env python3
"""
Minimal script to test translation functionality in isolation.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Set environment variables to prevent crashes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TRANSFORMERS_IGNORE_TORCH_VERSION'] = '1'

def test_translation():
    try:
        logging.info("Testing translation with a minimal example...")
        
        # Try importing the necessary modules
        import torch
        logging.info(f"PyTorch version: {torch.__version__}")
        
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        logging.info("Successfully imported transformers modules")
        
        # Instead of using the NLLB model which causes crashes, try a different model
        model_name = "Helsinki-NLP/opus-mt-en-ko"  # Different model for English to Korean
        
        logging.info(f"Loading tokenizer for model '{model_name}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logging.info("Tokenizer loaded successfully")
        
        logging.info(f"Loading model '{model_name}'...")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
        logging.info("Model loaded successfully")
        
        # Sample text to translate
        text = "Hello, this is a test translation."
        
        logging.info(f"Translating: '{text}'")
        inputs = tokenizer(text, return_tensors="pt")
        
        # Generate translation
        with torch.no_grad():
            output_ids = model.generate(**inputs)
            
        # Decode the output
        translated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        logging.info(f"Translation result: '{translated_text}'")
        
        return True, translated_text
    
    except Exception as e:
        logging.error(f"Error during translation test: {e}", exc_info=True)
        return False, str(e)

if __name__ == "__main__":
    success, result = test_translation()
    
    if success:
        logging.info("Translation test completed successfully.")
        sys.exit(0)
    else:
        logging.error(f"Translation test failed: {result}")
        sys.exit(1)
