"""
Tests for the Transcriber module.
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from extract_transcript.transcriber import Transcriber

@pytest.fixture
def mock_whisper_model():
    """Creates a mock WhisperModel for testing."""
    mock_model = MagicMock()
    
    # Mock transcribe method
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.95
    
    mock_segment1 = MagicMock()
    mock_segment1.text = "This is segment one."
    
    mock_segment2 = MagicMock()
    mock_segment2.text = "This is segment two."
    
    mock_model.transcribe.return_value = (
        [mock_segment1, mock_segment2],  # Generator mock
        mock_info
    )
    
    return mock_model

@pytest.fixture
def transcriber():
    """Creates a Transcriber instance for testing."""
    return Transcriber(model_name="base")

@patch("extract_transcript.transcriber.WhisperModel")
def test_transcriber_init(mock_whisper_class):
    """Tests Transcriber initialization."""
    transcriber = Transcriber(model_name="test_model", device="test_device", compute_type="test_type")
    
    assert transcriber.model_name == "test_model"
    assert transcriber.device == "test_device"
    assert transcriber.compute_type == "test_type"
    assert transcriber.model is None  # Model not loaded on init

@patch("extract_transcript.transcriber.WhisperModel")
def test_load_model(mock_whisper_class, transcriber):
    """Tests the load_model method."""
    mock_model = MagicMock()
    mock_whisper_class.return_value = mock_model
    
    # First call should create the model
    model = transcriber.load_model()
    
    mock_whisper_class.assert_called_once_with(
        "base", device="cpu", compute_type="int8"
    )
    assert model == mock_model
    assert transcriber.model == mock_model
    
    # Subsequent calls should reuse the model
    mock_whisper_class.reset_mock()
    model2 = transcriber.load_model()
    
    mock_whisper_class.assert_not_called()
    assert model2 == model

@patch("extract_transcript.transcriber.Path")
@patch("extract_transcript.transcriber.WhisperModel")
def test_transcribe_file_not_found(mock_whisper_class, mock_path, transcriber):
    """Tests transcription when the file is not found."""
    mock_path_instance = MagicMock()
    mock_path_instance.exists.return_value = False
    mock_path.return_value = mock_path_instance
    
    result = transcriber.transcribe("nonexistent_file.mp3")
    
    assert result == (None, None, None)
    mock_whisper_class.assert_not_called()

@patch("extract_transcript.transcriber.WhisperModel")
def test_transcribe_success(mock_whisper_class, mock_whisper_model, transcriber):
    """Tests successful transcription."""
    mock_whisper_class.return_value = mock_whisper_model
    
    # Set up mock segments and info for the return value
    mock_segment1 = MagicMock()
    mock_segment1.text = "This is segment one."
    
    mock_segment2 = MagicMock()
    mock_segment2.text = "This is segment two."
    
    mock_segments = [mock_segment1, mock_segment2]
    
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.95
    
    mock_whisper_model.transcribe.return_value = (mock_segments, mock_info)
    
    # Mock Path.exists to return True, and capture the path instance
    with patch("extract_transcript.transcriber.Path") as mock_path:
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.__str__.return_value = "test_file.mp3"
        mock_path.return_value = mock_path_instance
    
        full_text, segments, lang = transcriber.transcribe("test_file.mp3", language="en")
    
    assert full_text == "This is segment one.This is segment two."
    assert len(segments) == 2
    assert lang == "en"
    # Use the str representation of the mock path to check correct call
    mock_whisper_model.transcribe.assert_called_once_with(
        mock_path_instance.__str__(), beam_size=5, language="en"
    )

@patch("extract_transcript.transcriber.WhisperModel")
def test_transcribe_exception(mock_whisper_class, transcriber):
    """Tests transcription when an exception occurs."""
    mock_model = MagicMock()
    mock_model.transcribe.side_effect = Exception("Test error")
    mock_whisper_class.return_value = mock_model
    
    # Mock Path.exists to return True
    with patch("extract_transcript.transcriber.Path") as mock_path:
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        result = transcriber.transcribe("test_file.mp3")
    
    assert result == (None, None, None)
