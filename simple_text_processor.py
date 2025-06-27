#!/usr/bin/env python3
"""
Simplified text processor that extracts text from files and performs basic operations.
This script bypasses the transformers models to avoid segmentation faults.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from extract_transcript.text_extractor import TextExtractor
from extract_transcript.output_writer import OutputWriter

def configure_logging():
    """Configure logging format and level."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def process_text_file(input_file, output_dir, args):
    """
    Process a text file directly without using transformers models.
    
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
        output_writer.save(input_file.name, text_content, "_transcription.txt")
        logging.info(f"Extracted text saved to {output_dir / input_file.name}_transcription.txt")
        
        # Handle translation placeholder
        if args.translate_to:
            logging.info(f"Translation to {args.translate_to} is currently unavailable due to compatibility issues.")
            logging.info("To translate the text, you can use an online translation service.")
            placeholder = f"[Translation to {args.translate_to} not available in simplified mode]"
            output_writer.save(input_file.name, placeholder, f"_translation_{args.translate_to}.txt")
        
        # Handle summarization placeholder
        if args.summarize:
            logging.info("Summarization is currently unavailable due to compatibility issues.")
            logging.info("To summarize the text, you could use an online summarization tool.")
            placeholder = "[Summary not available in simplified mode]"
            output_writer.save(input_file.name, placeholder, "_summary.txt")
        
        return True
        
    except Exception as e:
        logging.error(f"Error processing {input_file}: {str(e)}")
        return False

def main():
    """Main entry point of the script."""
    # Configure logging
    configure_logging()
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process text files without using transformers models.")
    parser.add_argument("input_file", type=str, help="Path to the input text file.")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory.")
    parser.add_argument("--translate_to", type=str, help="Target language code for translation (placeholder).")
    parser.add_argument("--summarize", action="store_true", help="Generate summary (placeholder).")
    parser.add_argument("--summary_length", type=int, default=150, help="Summary length (placeholder).")
    
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
    # Set environment variable to avoid conflicts
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    sys.exit(main())
