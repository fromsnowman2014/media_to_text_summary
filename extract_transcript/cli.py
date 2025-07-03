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
from .text_extractor import TextExtractor

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
    output_writer: OutputWriter = None,
    text_extractor: TextExtractor = None
):
    """
    Process a single file (audio, video, text, or PDF).
    
    Args:
        file_path: Path to the input file
        output_dir: Directory where output files will be saved
        args: Command-line arguments
        transcriber: Transcriber instance
        translator: Translator instance (optional)
        summarizer: Summarizer instance (optional)
        subtitle_generator: SubtitleGenerator instance (optional)
        output_writer: OutputWriter instance (optional)
        text_extractor: TextExtractor instance (optional)
    
    Returns:
        True if processing was successful, False otherwise
    """
    if output_writer is None:
        output_writer = OutputWriter(output_dir)
    else:
        output_writer.set_output_dir(output_dir)
        
    # Determine input type by extension or explicit argument
    input_type = args.input_type
    if not input_type:
        # Auto-detect based on file extension
        file_ext = file_path.suffix.lower()
        if file_ext in ['.mp3', '.wav', '.m4a', '.mp4', '.mov', '.flac', '.ogg']:
            input_type = 'audio'
        elif file_ext in ['.txt', '.srt']:
            input_type = 'text'
        elif file_ext in ['.pdf']:
            input_type = 'pdf'
        else:
            logging.warning(f"Unrecognized file extension: {file_ext}. Attempting audio processing.")
            input_type = 'audio'
    
    full_text = None
    segments = None
    detected_lang = args.language
    
    if input_type in ['audio', 'video']:
        # Process as audio/video through transcription
        full_text, segments, detected_lang = transcriber.transcribe(file_path, args.language)
        if not full_text:
            logging.error(f"Failed to transcribe {file_path}")
            return False
    elif input_type in ['text', 'pdf']:
        # Direct document text extraction
        if text_extractor is None:
            text_extractor = TextExtractor()
            
        try:
            full_text = text_extractor.extract(file_path)
            if not detected_lang:
                detected_lang = 'en'  # Default language for text files
            
            # No segments for text files
            segments = None
        except Exception as e:
            logging.error(f"Failed to extract text from {file_path}: {str(e)}")
            return False
    
    # Save the transcription
    output_writer.save(file_path.name, full_text, "_transcription.txt")
    
    # Translate if requested
    if args.translate_to and translator:
        try:
            logging.info(f"Translating from {detected_lang} to {args.translate_to}...")
            translated_text = translator.translate(full_text, detected_lang, args.translate_to)
            if translated_text:
                output_writer.save(file_path.name, translated_text, f"_translation_{args.translate_to}.txt")
            else:
                logging.error("Translation failed or produced empty results.")
                translated_text = None  # Ensure this is explicitly None if translation failed
        except Exception as e:
            logging.error(f"Translation process failed with error: {e}")
            translated_text = None  # Ensure this is explicitly None if translation failed
    
    # Summarize if requested
    if args.summarize and summarizer:
        try:
            logging.info("Generating summary of transcription...")
            summary_text = summarizer.summarize(full_text, max_length=args.summary_length)
            if summary_text:
                output_writer.save(file_path.name, summary_text, "_summary.txt")
                
                # Also translate the summary if translation was requested
                if args.translate_to and translator and translated_text:  # Only if text was already translated
                    try:
                        logging.info(f"Translating summary to {args.translate_to}...")
                        
                        # Force memory cleanup before summary translation
                        if hasattr(translator, '_cleanup_memory'):
                            logging.debug("Performing extra memory cleanup before summary translation")
                            translator._cleanup_memory()
                            
                        # Limit summary size to prevent memory issues
                        if len(summary_text) > 500:
                            logging.warning(f"Summary too large ({len(summary_text)} chars), truncating to 500 chars")
                            summary_to_translate = summary_text[:500] + "..."
                        else:
                            summary_to_translate = summary_text
                            
                        # Use a separate try-except for the actual translation call
                        try:
                            translated_summary = translator.translate(summary_to_translate, detected_lang, args.translate_to)
                            if translated_summary:
                                output_writer.save(file_path.name, translated_summary, f"_summary_{args.translate_to}.txt")
                            else:
                                logging.error("Summary translation failed or produced empty results.")
                                # Create a fallback message so the process can continue
                                output_writer.save(file_path.name, 
                                                 "[Summary translation failed - see log for details]", 
                                                 f"_summary_{args.translate_to}.txt")
                        except Exception as trans_err:
                            logging.error(f"Error during summary translation: {trans_err}")
                            # Create fallback output to prevent process failure
                            output_writer.save(file_path.name, 
                                             f"[Summary translation error: {str(trans_err)[:100]}...]", 
                                             f"_summary_{args.translate_to}.txt")
                    except Exception as e:
                        logging.error(f"Failed to translate summary: {e}")
                        # Continue execution despite summary translation failure
            else:
                logging.warning("Summary generation failed or produced empty results.")
        except Exception as e:
            logging.error(f"Failed to generate summary: {e}")
            # Continue execution despite summary failure
            
    # Generate subtitles if requested (only for audio/video with segments)
    if args.generate_subtitles and subtitle_generator and segments and input_type in ['audio', 'video']:
        logging.info(f"Generating subtitles in {args.subtitle_format} format...")
        if args.subtitle_format == "srt" or args.subtitle_format == "both":
            srt_content = subtitle_generator.generate(segments, SubtitleFormat.SRT)
            output_writer.save(file_path.name, srt_content, ".srt")
            
        if args.subtitle_format == "vtt" or args.subtitle_format == "both":
            vtt_content = subtitle_generator.generate(segments, SubtitleFormat.VTT)
            output_writer.save(file_path.name, vtt_content, ".vtt")
    elif args.generate_subtitles and input_type in ['text', 'pdf']:
        logging.info("Subtitle generation skipped for document input.")
    
    return True

def main():
    """Main entry point for the CLI application."""
    configure_logging()
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Transcribe and process media and text files.")
    parser.add_argument("input_file", type=str, help="Path to the input file or directory.")
    parser.add_argument("--input_type", type=str, choices=['audio', 'video', 'text', 'pdf'], 
                      help="Type of input file (auto-detected if not specified).")
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
        
    text_extractor = TextExtractor()
    
    # Process files
    if input_path.is_dir():
        # Process all audio/video files in the directory
        success_count = 0
        total_files = 0
        
        # Process all supported files in the directory
        media_extensions = ["*.mp3", "*.mp4", "*.wav", "*.m4a", "*.mov", "*.flac", "*.ogg"]
        text_extensions = ["*.txt", "*.srt", "*.pdf"]
        all_extensions = media_extensions + text_extensions
        
        for ext in all_extensions:
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
                    output_writer,
                    text_extractor
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
                output_writer,
                text_extractor
            ):
                logging.info("Processing completed successfully.")
            else:
                logging.error("Failed to process file.")

if __name__ == "__main__":
    main()
