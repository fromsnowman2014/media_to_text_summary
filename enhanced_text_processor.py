#!/usr/bin/env python3
"""
Enhanced text processor that extracts text from files and performs basic translation and summarization.
This script implements simplified versions of these features to avoid segmentation faults.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from extract_transcript.text_extractor import TextExtractor
from extract_transcript.output_writer import OutputWriter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Set environment variables to prevent crashes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TRANSFORMERS_IGNORE_TORCH_VERSION'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # Try to use cached models only

class SimpleTranslator:
    """A simplified translator that handles basic translations."""
    
    def __init__(self):
        self.initialized = False
        self.model = None
        self.tokenizer = None
    
    def translate(self, text, src_lang="en", target_lang="ko"):
        """
        Translate text using a simplified approach.
        
        Args:
            text: Text to translate
            src_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text or placeholder if translation fails
        """
        try:
            logging.info(f"Attempting translation from {src_lang} to {target_lang}...")
            
            # Try to import necessary modules
            import torch
            from transformers import MarianMTModel, MarianTokenizer
            
            # Use a simpler model for translation
            model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{target_lang}"
            
            try:
                # Try to load the model - this might fail if model isn't already downloaded
                if not self.initialized:
                    logging.info(f"Loading translation model {model_name}...")
                    try:
                        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
                        self.model = MarianMTModel.from_pretrained(model_name)
                        self.initialized = True
                    except Exception as model_error:
                        logging.warning(f"Failed to load translation model: {model_error}")
                        return self._get_placeholder_translation(text, target_lang)
                
                # If we successfully loaded the model, perform translation
                if self.initialized:
                    inputs = self.tokenizer(text, return_tensors="pt")
                    with torch.no_grad():
                        translated_ids = self.model.generate(**inputs)
                    translated_text = self.tokenizer.decode(translated_ids[0], skip_special_tokens=True)
                    return translated_text
                
                return self._get_placeholder_translation(text, target_lang)
                
            except Exception as e:
                logging.warning(f"Translation failed: {e}")
                return self._get_placeholder_translation(text, target_lang)
                
        except ImportError as e:
            logging.warning(f"Required modules not available for translation: {e}")
            return self._get_placeholder_translation(text, target_lang)
    
    def _get_placeholder_translation(self, text, target_lang):
        """Generate a placeholder when translation is unavailable."""
        return f"[Translation to {target_lang} unavailable - please use an online translation service for '{text[:50]}...']"


class SimpleSummarizer:
    """A simplified summarizer that handles basic text summarization."""
    
    def __init__(self):
        self.initialized = False
        self.model = None
        self.tokenizer = None
    
    def summarize(self, text, max_length=500, min_length=100):
        """
        Summarize text using a simplified approach.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of the summary
            min_length: Minimum length of the summary
            
        Returns:
            Summarized text or placeholder if summarization fails
        """
        try:
            logging.info("Attempting summarization...")
            
            # Try to import necessary modules
            import torch
            from transformers import pipeline
            
            try:
                # Try to load the summarization pipeline
                if not self.initialized:
                    logging.info("Loading summarization model...")
                    try:
                        # Try using a smaller summarization model
                        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                        self.initialized = True
                    except Exception as model_error:
                        logging.warning(f"Failed to load summarization model: {model_error}")
                        return self._get_placeholder_summary(text, max_length)
                
                # If we successfully loaded the model, perform summarization
                if self.initialized:
                    result = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
                    if result and len(result) > 0:
                        return result[0]['summary_text']
                
                return self._get_placeholder_summary(text, max_length)
                
            except Exception as e:
                logging.warning(f"Summarization failed: {e}")
                return self._get_placeholder_summary(text, max_length)
                
        except ImportError as e:
            logging.warning(f"Required modules not available for summarization: {e}")
            return self._get_placeholder_summary(text, max_length)
    
    def _get_placeholder_summary(self, text, max_length):
        """Generate a basic summary when model-based summarization is unavailable."""
        # Implement a very basic summarization by taking the first few sentences
        sentences = text.split('.')
        simple_summary = '. '.join(sentences[:5]) + '.'
        
        # Trim to max_length if needed
        if len(simple_summary) > max_length:
            simple_summary = simple_summary[:max_length] + "..."
            
        return f"[AI Summary unavailable - Basic extract: {simple_summary}]"


def process_text_file(input_file, output_dir, args):
    """
    Process a text file with translation and summarization.
    
    Args:
        input_file: Path to the input text file
        output_dir: Directory where output files will be saved
        args: Command line arguments
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Extract text from file
        extractor = TextExtractor()
        text_content = extractor.extract(input_file)
        
        # Create output writer
        output_writer = OutputWriter(output_dir)
        
        # Save the extracted text
        transcription_path = output_writer.save(input_file.name, text_content, "_transcription.txt")
        logging.info(f"Extracted text saved to {transcription_path}")
        
        # Handle translation if requested
        if args.translate_to:
            logging.info(f"Translating from {args.language} to {args.translate_to}...")
            translator = SimpleTranslator()
            translated_text = translator.translate(text_content, args.language, args.translate_to)
            translation_path = output_writer.save(input_file.name, translated_text, f"_translation_{args.translate_to}.txt")
            logging.info(f"Translation saved to {translation_path}")
        
        # Handle summarization if requested
        if args.summarize:
            logging.info("Generating summary...")
            summarizer = SimpleSummarizer()
            summary_text = summarizer.summarize(text_content, args.summary_length, args.summary_length // 3)
            summary_path = output_writer.save(input_file.name, summary_text, "_summary.txt")
            logging.info(f"Summary saved to {summary_path}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error processing {input_file}: {str(e)}")
        return False

def main():
    """Main entry point of the script."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Enhanced text processor with simplified translation and summarization.")
    parser.add_argument("input_file", type=str, help="Path to the input text file.")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory.")
    parser.add_argument("--language", type=str, default="en", help="Source language code.")
    parser.add_argument("--translate_to", type=str, help="Target language code for translation.")
    parser.add_argument("--summarize", action="store_true", help="Generate summary.")
    parser.add_argument("--summary_length", type=int, default=500, help="Maximum summary length.")
    
    args = parser.parse_args()
    
    # Convert input path to Path object
    input_path = Path(args.input_file)
    
    # Check if input file exists
    if not input_path.exists():
        logging.error(f"Input file does not exist: {input_path}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Process the input file
    if process_text_file(input_path, output_dir, args):
        logging.info("Text processing completed successfully.")
        return 0
    else:
        logging.error("Text processing failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
