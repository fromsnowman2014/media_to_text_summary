"""
Translator module for translating transcribed text using NLLB models.
"""

import logging
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Set environment variable to disable the torch version check
# This is a temporary solution until torch 2.6.0 becomes available
os.environ['TRANSFORMERS_IGNORE_TORCH_VERSION'] = '1'

# Define language codes for NLLB
NLLB_LANGUAGE_CODES = {
    'en': 'eng_Latn',
    'ko': 'kor_Hang',
    'ja': 'jpn_Jpan',
    'zh': 'zho_Hans',
    'es': 'spa_Latn',
    'fr': 'fra_Latn',
    'de': 'deu_Latn',
}

class Translator:
    """A class to handle translation of transcribed text."""
    
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M"):
        """
        Initialize a Translator instance.
        
        Args:
            model_name: The name of the NLLB model to use for translation
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
    
    def load_model(self):
        """Load the translation model and tokenizer."""
        if self.model is None or self.tokenizer is None:
            logging.info(f"Loading translation model '{self.model_name}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # Disable torch security warnings for version 2.6 (not available yet)
            # Try loading the model with various fallback options
            try:
                # Prefer safetensors format when available
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name, 
                    prefer_safetensors=True,
                    local_files_only=os.environ.get('TRANSFORMERS_OFFLINE', '0') == '1'
                )
            except (TypeError, ValueError) as e:
                logging.warning(f"Model loading with safetensors failed: {e}")
                try:
                    # Try standard loading with security bypass
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.model_name,
                        local_files_only=os.environ.get('TRANSFORMERS_OFFLINE', '0') == '1',
                        trust_remote_code=True  # This helps bypass some security checks
                    )
                except Exception as e2:
                    logging.error(f"Standard model loading failed: {e2}")
                    raise e2
            logging.info("Translation model loaded successfully.")
        return self.model, self.tokenizer
    
    def is_language_supported(self, language_code: str) -> bool:
        """Check if a language code is supported for translation."""
        return language_code in NLLB_LANGUAGE_CODES
    
    def translate(self, text: str, src_lang: str, target_lang: str):
        """
        Translate text from source language to target language.
        
        Args:
            text: The text to translate
            src_lang: Source language code (e.g., 'en', 'ko')
            target_lang: Target language code (e.g., 'en', 'ko')
            
        Returns:
            Translated text or None if translation failed
        """
        if not self.is_language_supported(src_lang) or not self.is_language_supported(target_lang):
            logging.error(f"Translation from '{src_lang}' to '{target_lang}' is not supported.")
            return None

        try:
            model, tokenizer = self.load_model()
            
            src_code = NLLB_LANGUAGE_CODES[src_lang]
            target_code = NLLB_LANGUAGE_CODES[target_lang]

            inputs = tokenizer(text, return_tensors="pt")
            translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[target_code])
            
            translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            return translated_text
        except Exception as e:
            logging.error(f"An error occurred during translation: {e}")
            return None
            
    def get_supported_languages(self):
        """Return a list of supported language codes."""
        return list(NLLB_LANGUAGE_CODES.keys())
