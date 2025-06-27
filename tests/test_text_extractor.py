"""
Unit tests for the TextExtractor class.
"""

import os
import pytest
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock

from extract_transcript.text_extractor import TextExtractor

class TestTextExtractor:
    """Test cases for the TextExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = TextExtractor()
        
        # Create temp dir for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        
        # Create test files
        self.text_file = self.test_dir / "test.txt"
        with open(self.text_file, 'w', encoding='utf-8') as f:
            f.write("This is a test text file.\nIt has multiple lines.\n")
            
        self.srt_file = self.test_dir / "test.srt"
        with open(self.srt_file, 'w', encoding='utf-8') as f:
            f.write("""1
00:00:01,000 --> 00:00:04,000
This is a test subtitle

2
00:00:05,000 --> 00:00:09,000
With multiple entries
""")

    def teardown_method(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
        
    def test_extract_txt_file(self):
        """Test extraction from a .txt file."""
        result = self.extractor.extract(self.text_file)
        assert result == "This is a test text file.\nIt has multiple lines.\n"
        
    def test_extract_srt_file(self):
        """Test extraction from a .srt file."""
        result = self.extractor.extract(self.srt_file)
        assert "This is a test subtitle" in result
        assert "With multiple entries" in result
        
    def test_unsupported_extension(self):
        """Test handling of unsupported file extensions."""
        unsupported_file = self.test_dir / "test.unknown"
        with open(unsupported_file, 'w') as f:
            f.write("test")
            
        with pytest.raises(ValueError) as excinfo:
            self.extractor.extract(unsupported_file)
        assert "Unsupported file format" in str(excinfo.value)
        
    def test_nonexistent_file(self):
        """Test handling of nonexistent files."""
        nonexistent_file = self.test_dir / "nonexistent.txt"
        
        with pytest.raises(FileNotFoundError):
            self.extractor.extract(nonexistent_file)
            
    @patch('extract_transcript.text_extractor.pdfplumber')
    def test_pdf_extraction(self, mock_pdfplumber):
        """Test PDF extraction using mock."""
        # Create mock PDF with pages
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1 content"
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page 2 content"
        
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page1, mock_page2]
        mock_pdf.__enter__.return_value = mock_pdf
        
        mock_pdfplumber.open.return_value = mock_pdf
        
        # Create a dummy PDF file (contents don't matter since we're mocking)
        pdf_file = self.test_dir / "test.pdf"
        with open(pdf_file, 'wb') as f:
            f.write(b'%PDF-1.5\n%mock PDF file\n')
            
        result = self.extractor.extract(pdf_file)
        assert result == "Page 1 content\n\nPage 2 content"
        mock_pdfplumber.open.assert_called_once_with(pdf_file)
        
    @patch('extract_transcript.text_extractor.pdfplumber')
    def test_pdf_extraction_error(self, mock_pdfplumber):
        """Test handling of PDF extraction errors."""
        mock_pdfplumber.open.side_effect = Exception("PDF error")
        
        pdf_file = self.test_dir / "test.pdf"
        with open(pdf_file, 'wb') as f:
            f.write(b'%PDF-1.5\n%mock PDF file\n')
            
        with pytest.raises(RuntimeError) as excinfo:
            self.extractor.extract(pdf_file)
        assert "Error extracting text" in str(excinfo.value)
