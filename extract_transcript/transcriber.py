"""
Transcriber module for handling audio/video transcription using faster-whisper.
"""

import logging
from pathlib import Path
from faster_whisper import WhisperModel

class Transcriber:
    """A class to handle audio/video transcription tasks."""
    
    def __init__(self, model_name: str = "base", device: str = "cpu", compute_type: str = "int8"):
        """
        Initialize a Transcriber instance.
        
        Args:
            model_name: The name of the whisper model to use (e.g., tiny, base, small, medium, large-v3)
            device: Device to use for computation (cpu or cuda)
            compute_type: Compute type for the model (int8, float16, float32)
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.model = None
    
    def load_model(self):
        """Load the whisper model."""
        if self.model is None:
            logging.info(f"Loading model '{self.model_name}'...")
            self.model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
            logging.info(f"Model '{self.model_name}' loaded successfully.")
        return self.model
    
    def transcribe(self, file_path: str, language: str = None):
        """
        Transcribe an audio/video file.
        
        Args:
            file_path: Path to the audio/video file
            language: Language code of the audio (e.g., ko, en). Auto-detect if not specified.
            
        Returns:
            Tuple of (full_text, segments, detected_language) or (None, None, None) if failed
        """
        path_obj = Path(file_path)
        if not path_obj.exists():
            logging.error(f"File not found: {file_path}")
            return None, None, None
        
        try:
            model = self.load_model()
            logging.info(f"Transcribing file: {file_path}...")
            segments_generator, info = model.transcribe(str(path_obj), beam_size=5, language=language)
            
            segment_list = list(segments_generator)
            full_text = "".join(segment.text for segment in segment_list)
            
            detected_lang = info.language
            logging.info(f"Detected language '{detected_lang}' with probability {info.language_probability:.2f}")
            
            return full_text, segment_list, detected_lang
        
        except Exception as e:
            logging.error(f"An error occurred during transcription: {e}")
            return None, None, None
