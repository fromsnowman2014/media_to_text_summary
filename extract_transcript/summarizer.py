"""
Summarizer module for generating summaries from transcribed text.
"""

import logging
import os
from transformers import pipeline

# Set environment variable to disable the torch version check
# This is a temporary solution until torch 2.6.0 becomes available
os.environ['TRANSFORMERS_IGNORE_TORCH_VERSION'] = '1'

class Summarizer:
    """A class to handle text summarization."""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        Initialize a Summarizer instance.
        
        Args:
            model_name: The name of the summarization model to use
        """
        self.model_name = model_name
        self.summarizer = None
    
    def load_model(self):
        """Load the summarization model."""
        if self.summarizer is None:
            logging.info(f"Loading summarization model '{self.model_name}'...")
            try:
                # Attempt to load with security bypass
                self.summarizer = pipeline("summarization", model=self.model_name, trust_remote_code=True)
                logging.info("Summarization model loaded successfully.")
            except Exception as e:
                logging.warning(f"First attempt to load summarization model failed: {e}")
                try:
                    # Fallback to offline loading
                    self.summarizer = pipeline(
                        "summarization", 
                        model=self.model_name,
                        trust_remote_code=True,
                        local_files_only=True
                    )
                    logging.info("Summarization model loaded successfully (offline mode).")
                except Exception as e2:
                    logging.error(f"Failed to load summarization model: {e2}")
                    raise e2
        return self.summarizer
    
    def summarize(self, text: str, max_length: int = 150, min_length: int = 30):
        """
        Generate a summary of the given text.
        
        Args:
            text: The text to summarize
            max_length: Maximum length of the summary in tokens
            min_length: Minimum length of the summary in tokens
            
        Returns:
            Summary text or None if summarization failed
        """
        try:
            summarizer = self.load_model()
            result = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
            if result and len(result) > 0:
                return result[0]['summary_text']
            return None
        except Exception as e:
            logging.error(f"An error occurred during summarization: {e}")
            return None
