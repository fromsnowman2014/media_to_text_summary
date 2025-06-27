"""
CLI module for handling command-line interface for the extract_transcript tool.
"""

import argparse
import logging
from pathlib import Path

from .transcriber import Transcriber
from .translator import Translator
from .summarizer import Summarizer
from .subtitle_generator import SubtitleGenerator, SubtitleFormat
from .output_writer import OutputWriter

def configure_logging():
    """Configure logging format and level."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def process_file(
    file_path: Path, 
    output_dir: Path, 
    args,
    transcriber: Transcriber,
    translator: Translator = None,
    summarizer: Summarizer = None,
    subtitle_generator: SubtitleGenerator = None,
    output_writer: OutputWriter = None
):
    """
    Process a single audio/video file.
    
    Args:
        file_path: Path to the audio/video file
        output_dir: Directory where output files will be saved
        args: Command-line arguments
        transcriber: Transcriber instance
        translator: Translator instance (optional)
        summarizer: Summarizer instance (optional)
        subtitle_generator: SubtitleGenerator instance (optional)
        output_writer: OutputWriter instance (optional)
    
    Returns:
        True if processing was successful, False otherwise
    """
    if output_writer is None:
        output_writer = OutputWriter(output_dir)
    else:
        output_writer.set_output_dir(output_dir)
        
    # Transcribe the file
    full_text, segments, detected_lang = transcriber.transcribe(file_path, args.language)
    if not full_text:
        logging.error(f"Failed to transcribe {file_path}")
        return False
    
    # Save the transcription
    output_writer.save(file_path.name, full_text, "_transcription.txt")
    
    # Translate if requested
    if args.translate_to and translator:
        logging.info(f"Translating from {detected_lang} to {args.translate_to}...")
        translated_text = translator.translate(full_text, detected_lang, args.translate_to)
        if translated_text:
            output_writer.save(file_path.name, translated_text, f"_translation_{args.translate_to}.txt")
    
    # Summarize if requested
    if args.summarize and summarizer:
        logging.info("Generating summary of transcription...")
        summary_text = summarizer.summarize(full_text, max_length=args.summary_length)
        if summary_text:
            output_writer.save(file_path.name, summary_text, "_summary.txt")
    
    # Generate subtitles if requested
    if args.generate_subtitles and subtitle_generator and segments:
        logging.info(f"Generating subtitles in {args.subtitle_format} format...")
        if args.subtitle_format == "srt" or args.subtitle_format == "both":
            srt_content = subtitle_generator.generate(segments, SubtitleFormat.SRT)
            output_writer.save(file_path.name, srt_content, ".srt")
            
        if args.subtitle_format == "vtt" or args.subtitle_format == "both":
            vtt_content = subtitle_generator.generate(segments, SubtitleFormat.VTT)
            output_writer.save(file_path.name, vtt_content, ".vtt")
    
    return True

def main():
    """Main entry point for the CLI application."""
    configure_logging()
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Transcribe and process audio/video files.")
    parser.add_argument("input_file", type=str, help="Path to the audio/video file or directory.")
    parser.add_argument("--model", type=str, default="base", help="Whisper model size.")
    parser.add_argument("--language", type=str, default=None, help="Language code for transcription.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--translate_to", type=str, default=None, help="Language to translate the text to (e.g., 'en', 'ko').")
    parser.add_argument("--summarize", action="store_true", help="Generate a summary of the transcription.")
    parser.add_argument("--summary_length", type=int, default=150, help="Maximum length of summary in words.")
    parser.add_argument("--generate_subtitles", action="store_true", help="Generate subtitle files from the transcription.")
    parser.add_argument("--subtitle_format", type=str, default="srt", choices=["srt", "vtt", "both"], 
                       help="Format of subtitle file to generate (srt, vtt, or both).")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Default to a directory with the same name as the input in the current directory
        if input_path.is_dir():
            output_dir = Path("output") / input_path.name
        else:
            output_dir = Path("output")
    
    # Initialize components
    transcriber = Transcriber(model_name=args.model)
    output_writer = OutputWriter(output_dir)
    
    translator = None
    if args.translate_to:
        translator = Translator()
    
    summarizer = None
    if args.summarize:
        summarizer = Summarizer()
    
    subtitle_generator = None
    if args.generate_subtitles:
        subtitle_generator = SubtitleGenerator()
    
    # Process files
    if input_path.is_dir():
        # Process all audio/video files in the directory
        success_count = 0
        total_files = 0
        
        for ext in ["*.mp3", "*.mp4", "*.wav", "*.m4a", "*.mov", "*.flac", "*.ogg"]:
            for file_path in input_path.glob(ext):
                total_files += 1
                file_output_dir = output_dir / file_path.stem
                if process_file(
                    file_path, 
                    file_output_dir, 
                    args, 
                    transcriber, 
                    translator, 
                    summarizer, 
                    subtitle_generator,
                    output_writer
                ):
                    success_count += 1
        
        if total_files > 0:
            logging.info(f"Processed {success_count}/{total_files} files successfully.")
        else:
            logging.warning(f"No audio/video files found in {input_path}")
    else:
        # Process a single file
        if not input_path.exists() or not input_path.is_file():
            logging.error(f"File not found: {input_path}")
        else:
            if process_file(
                input_path, 
                output_dir, 
                args, 
                transcriber, 
                translator, 
                summarizer, 
                subtitle_generator,
                output_writer
            ):
                logging.info("Processing completed successfully.")
            else:
                logging.error("Failed to process file.")

if __name__ == "__main__":
    main()
