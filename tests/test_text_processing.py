"""
Integration tests for text file processing.
"""

import os
import pytest
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock

from extract_transcript.cli import process_file
from extract_transcript.text_extractor import TextExtractor
from extract_transcript.output_writer import OutputWriter
from extract_transcript.summarizer import Summarizer
from extract_transcript.translator import Translator

class TestTextProcessing:
    """Integration tests for text file processing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        self.output_dir = self.test_dir / "output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create a mock text file
        self.text_file = self.test_dir / "sample.txt"
        with open(self.text_file, 'w', encoding='utf-8') as f:
            f.write("This is a sample text file for testing.\n"
                   "It has multiple lines and should be processed directly.\n"
                   "No transcription should happen.")
            
    def teardown_method(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_text_file_processing(self):
        """Test direct processing of a text file."""
        # Create mock args
        args = MagicMock()
        args.input_type = 'text'
        args.language = 'en'
        args.translate_to = None
        args.summarize = False
        args.generate_subtitles = False
        args.subtitle_format = 'srt'
        
        # Create components
        output_writer = OutputWriter(self.output_dir)
        text_extractor = TextExtractor()
        
        # Process the file
        result = process_file(
            self.text_file,
            self.output_dir,
            args,
            transcriber=None,  # Not needed for text processing
            translator=None,
            summarizer=None,
            subtitle_generator=None,
            output_writer=output_writer,
            text_extractor=text_extractor
        )
        
        # Check result
        assert result is True
        
        # Verify output file exists and contains the correct text
        output_file = self.output_dir / f"{self.text_file.stem}_transcription.txt"
        assert output_file.exists()
        
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        assert "This is a sample text file for testing." in content
        assert "No transcription should happen." in content
        
    @patch('extract_transcript.translator.Translator.translate')
    def test_text_file_with_translation(self, mock_translate):
        """Test text file processing with translation."""
        # Set up translation mock
        mock_translate.return_value = "이것은 번역된 텍스트입니다."
        
        # Create mock args
        args = MagicMock()
        args.input_type = 'text'
        args.language = 'en'
        args.translate_to = 'ko'
        args.summarize = False
        args.generate_subtitles = False
        
        # Create components
        output_writer = OutputWriter(self.output_dir)
        text_extractor = TextExtractor()
        translator = Translator()
        
        # Process the file
        result = process_file(
            self.text_file,
            self.output_dir,
            args,
            transcriber=None,
            translator=translator,
            summarizer=None,
            subtitle_generator=None,
            output_writer=output_writer,
            text_extractor=text_extractor
        )
        
        # Check result
        assert result is True
        
        # Verify translation file exists
        translation_file = self.output_dir / f"{self.text_file.stem}_translation_ko.txt"
        assert translation_file.exists()
        
        with open(translation_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        assert "이것은 번역된 텍스트입니다." in content
        
    @patch('extract_transcript.summarizer.Summarizer.summarize')
    def test_text_file_with_summarization(self, mock_summarize):
        """Test text file processing with summarization."""
        # Set up summarization mock
        mock_summarize.return_value = "This is a summary of the text."
        
        # Create mock args
        args = MagicMock()
        args.input_type = 'text'
        args.language = 'en'
        args.translate_to = None
        args.summarize = True
        args.summary_length = 100
        args.generate_subtitles = False
        
        # Create components
        output_writer = OutputWriter(self.output_dir)
        text_extractor = TextExtractor()
        summarizer = Summarizer()
        
        # Process the file
        result = process_file(
            self.text_file,
            self.output_dir,
            args,
            transcriber=None,
            translator=None,
            summarizer=summarizer,
            subtitle_generator=None,
            output_writer=output_writer,
            text_extractor=text_extractor
        )
        
        # Check result
        assert result is True
        
        # Verify summary file exists
        summary_file = self.output_dir / f"{self.text_file.stem}_summary.txt"
        assert summary_file.exists()
        
        with open(summary_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        assert "This is a summary of the text." in content
    
    def test_subtitle_option_ignored_for_text(self):
        """Test that subtitle generation is ignored for text files."""
        # Create mock args with subtitles enabled (should be ignored)
        args = MagicMock()
        args.input_type = 'text'
        args.language = 'en'
        args.translate_to = None
        args.summarize = False
        args.generate_subtitles = True
        args.subtitle_format = 'srt'
        
        # Create components
        output_writer = OutputWriter(self.output_dir)
        text_extractor = TextExtractor()
        subtitle_generator = MagicMock()  # Will verify this is not called
        
        # Process the file
        result = process_file(
            self.text_file,
            self.output_dir,
            args,
            transcriber=None,
            translator=None,
            summarizer=None,
            subtitle_generator=subtitle_generator,
            output_writer=output_writer,
            text_extractor=text_extractor
        )
        
        # Check result
        assert result is True
        
        # Verify no subtitle file was created
        srt_file = self.output_dir / f"{self.text_file.stem}.srt"
        assert not srt_file.exists()
        
        # Verify subtitle generator was not used
        subtitle_generator.generate.assert_not_called()
