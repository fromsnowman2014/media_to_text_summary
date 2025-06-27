"""
OutputWriter module for saving processed text to files.
"""

import logging
from pathlib import Path

class OutputWriter:
    """A class to handle saving outputs to files."""
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize an OutputWriter instance.
        
        Args:
            output_dir: Directory where output files will be saved
        """
        self.output_dir = output_dir
        
    def set_output_dir(self, output_dir: Path):
        """
        Set the output directory.
        
        Args:
            output_dir: Directory where output files will be saved
        """
        self.output_dir = output_dir
        
    def ensure_output_dir(self):
        """Ensure the output directory exists."""
        if self.output_dir and not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created output directory: {self.output_dir}")
    
    def save(self, original_filename: str, text: str, suffix: str) -> Path:
        """
        Save text content to a file.
        
        Args:
            original_filename: The original file name used to derive the output filename
            text: The text content to save
            suffix: The suffix to append to the filename (e.g., "_transcription.txt")
            
        Returns:
            Path to the saved file or None if failed
        """
        if not self.output_dir:
            logging.error("Output directory not set.")
            return None
            
        self.ensure_output_dir()
        
        # Generate output filename based on the original filename and suffix
        output_filename = Path(original_filename).stem + suffix
        output_path = self.output_dir / output_filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            logging.info(f"Output saved to {output_path}")
            return output_path
        except Exception as e:
            logging.error(f"Error saving output to {output_path}: {e}")
            return None
