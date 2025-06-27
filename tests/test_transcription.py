import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from extract_transcript.transcriber import Transcriber

# Mock segment data that faster-whisper would return
mock_segments_data = [
    MagicMock(text="This is a test. "),
    MagicMock(text="Hello world.")
]
# Mock info data
mock_info_data = MagicMock(language='en', language_probability=0.99)

@pytest.fixture
def mock_whisper_model(monkeypatch):
    """Mocks the WhisperModel and its transcribe method."""
    mock_model_instance = MagicMock()
    mock_model_instance.transcribe.return_value = (mock_segments_data, mock_info_data)
    
    mock_model_class = MagicMock(return_value=mock_model_instance)
    monkeypatch.setattr('extract_transcript.transcriber.WhisperModel', mock_model_class)
    
    return mock_model_class, mock_model_instance

def test_transcribe_audio_success(mock_whisper_model, monkeypatch):
    # Create a mock for Path that returns True for exists()
    mock_path = MagicMock()
    mock_path.exists.return_value = True
    monkeypatch.setattr(Path, '__new__', MagicMock(return_value=mock_path))
    """
    Tests the transcribe function with a mocked backend to ensure it processes data correctly.
    """
    # Setup mock model
    mock_model_class, mock_model_instance = mock_whisper_model
    
    # Mock segments and info
    mock_segment1 = MagicMock()
    mock_segment1.text = "This is a test. "
    
    mock_segment2 = MagicMock()
    mock_segment2.text = "Hello world."
    
    mock_segments = [mock_segment1, mock_segment2]
    
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.99
    
    mock_model_instance.transcribe.return_value = (mock_segments, mock_info)
    
    # Call the function
    transcriber = Transcriber(model_name="base")
    full_text, segments, detected_lang = transcriber.transcribe("fake/path/to/audio.mp3")

    # Assertions
    assert full_text == "This is a test. Hello world."
    assert len(segments) == 2
    assert segments[0].text == "This is a test. "
    assert segments[1].text == "Hello world."
    assert mock_path.exists.called
    
    # Check if the model was initialized and used correctly
    mock_model_class, mock_model_instance = mock_whisper_model
    mock_model_class.assert_called_once_with("base", device="cpu", compute_type="int8")
    mock_model_instance.transcribe.assert_called_once()

@patch('extract_transcript.transcriber.Path.exists', return_value=False)
def test_transcribe_audio_file_not_found(mock_path_exists):
    """
    Tests that transcribe returns None, None, None when the file does not exist.
    """
    transcriber = Transcriber(model_name="base")
    full_text, segments, detected_lang = transcriber.transcribe("non_existent_file.mp3")
    assert full_text is None
    assert segments is None
    assert detected_lang is None
    mock_path_exists.assert_called_once()
