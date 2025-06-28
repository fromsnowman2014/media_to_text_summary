#!/usr/bin/env python3
"""
Lightweight text processor with translation and summarization placeholders.
This script doesn't use any ML models to avoid segmentation faults.
"""

import os
import sys
import argparse
import logging
import re
from pathlib import Path
from extract_transcript.text_extractor import TextExtractor
from extract_transcript.output_writer import OutputWriter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class BasicTranslator:
    """A basic translator class that uses rule-based approaches for demo purposes."""
    
    def __init__(self):
        # Dictionary with some English to Korean translations for demo purposes
        self.en_ko_dict = {
            "hello": "안녕하세요",
            "thank you": "감사합니다",
            "good morning": "좋은 아침입니다",
            "good afternoon": "안녕하세요",
            "good evening": "안녕하세요",
            "goodbye": "안녕히 가세요",
            "yes": "네",
            "no": "아니오",
            "please": "부탁합니다",
            "sorry": "죄송합니다",
            "welcome": "환영합니다",
            "meeting": "회의",
            "document": "문서",
            "file": "파일",
            "computer": "컴퓨터",
            "software": "소프트웨어",
            "program": "프로그램",
            "data": "데이터",
            "information": "정보",
            "language": "언어"
        }
    
    def translate(self, text, src_lang="en", target_lang="ko"):
        """
        Basic translation function that replaces known words.
        
        This is just for demonstration purposes and doesn't do real translation.
        """
        if src_lang == "en" and target_lang == "ko":
            result = text
            for eng_word, kor_word in self.en_ko_dict.items():
                # Replace whole words with word boundaries
                pattern = r'\b' + re.escape(eng_word) + r'\b'
                result = re.sub(pattern, kor_word, result, flags=re.IGNORECASE)
            
            # Add explanation note
            result = f"[주의: 이것은 전문 번역이 아닙니다. ML 모델 로드 오류로 인한 기본 대체입니다.]\n\n{result}"
            return result
        else:
            return f"[Translation from {src_lang} to {target_lang} not supported in lightweight mode. Please use an online translation service.]"


class BasicSummarizer:
    """A basic summarizer class that uses rule-based approaches."""
    
    def summarize(self, text, max_length=500, min_length=100):
        """
        Basic summarization by extracting first few sentences.
        
        This is just for demonstration purposes and doesn't do real ML-based summarization.
        """
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Take the first sentence as the title/topic
        title = sentences[0] if sentences else ""
        
        # Take a few sentences from the beginning, middle, and end
        beginning = sentences[:3] if len(sentences) >= 3 else sentences
        
        if len(sentences) >= 6:
            middle_idx = len(sentences) // 2
            middle = sentences[middle_idx-1:middle_idx+2]
        else:
            middle = []
            
        if len(sentences) >= 9:
            end = sentences[-3:]
        else:
            end = [] if len(sentences) <= 3 else sentences[-1:]
        
        # Combine sentences for summary
        summary_sentences = beginning + middle + end
        
        # Remove duplicates while preserving order
        seen = set()
        unique_summary = []
        for s in summary_sentences:
            if s not in seen:
                seen.add(s)
                unique_summary.append(s)
        
        summary = " ".join(unique_summary)
        
        # Trim to max_length if needed
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
        
        # Add explanation note
        summary = f"[NOTE: This is a basic extract summary, not an AI-generated summary due to ML model loading errors]\n\n{summary}"
        
        return summary


def process_text_file(input_file, output_dir, args):
    """
    Process a text file with basic translation and summarization.
    
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
            translator = BasicTranslator()
            translated_text = translator.translate(text_content, args.language, args.translate_to)
            translation_path = output_writer.save(input_file.name, translated_text, f"_translation_{args.translate_to}.txt")
            logging.info(f"Basic translation saved to {translation_path}")
        
        # Handle summarization if requested
        if args.summarize:
            logging.info("Generating summary...")
            summarizer = BasicSummarizer()
            summary_text = summarizer.summarize(text_content, args.summary_length, args.summary_length // 3)
            summary_path = output_writer.save(input_file.name, summary_text, "_summary.txt")
            logging.info(f"Basic summary saved to {summary_path}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error processing {input_file}: {str(e)}")
        return False

def main():
    """Main entry point of the script."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Lightweight text processor with basic translation and summarization.")
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
