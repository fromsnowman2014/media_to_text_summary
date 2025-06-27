"""
Module for extracting text from various document formats including .txt, .srt, and .pdf.
"""

import pathlib
import logging
try:
    import pdfplumber
except ImportError:
    logging.warning("pdfplumber not installed. PDF extraction will not be available.")
    pdfplumber = None

class TextExtractor:
    """Extracts text from document files (.txt, .srt, .pdf)."""

    def __init__(self):
        self._supported_extensions = {
            '.txt': self._extract_from_txt,
            '.srt': self._extract_from_txt,
            '.pdf': self._extract_from_pdf,
        }
        
        # Check PDF support
        if pdfplumber is None:
            del self._supported_extensions['.pdf']
            logging.warning("PDF extraction disabled: pdfplumber module not found.")

    def extract(self, file_path: pathlib.Path) -> str:
        """
        Extract text from the given document file.
        
        Args:
            file_path: Path to the document file (.txt, .srt, or .pdf)
            
        Returns:
            The extracted text as a string.
            
        Raises:
            ValueError: If the file format is not supported.
            FileNotFoundError: If the file doesn't exist.
            RuntimeError: If there's an error during extraction.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = file_path.suffix.lower()
        if file_ext not in self._supported_extensions:
            supported_formats = ', '.join(self._supported_extensions.keys())
            raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: {supported_formats}")
        
        try:
            extract_method = self._supported_extensions[file_ext]
            return extract_method(file_path)
        except Exception as e:
            raise RuntimeError(f"Error extracting text from {file_path}: {str(e)}")
            
    def _extract_from_txt(self, file_path: pathlib.Path) -> str:
        """Extract text from a plain text file (.txt or .srt)."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
            
    def _extract_from_pdf(self, file_path: pathlib.Path) -> str:
        """Extract text from a PDF file using pdfplumber."""
        if pdfplumber is None:
            raise RuntimeError("Cannot extract PDF: pdfplumber module not available.")
            
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text_parts.append(extracted)
                    
        return "\n\n".join(text_parts)
