"""
Tests for the SubtitleGenerator module.
"""

import pytest
from unittest.mock import MagicMock
from extract_transcript.subtitle_generator import SubtitleGenerator, SubtitleFormat

@pytest.fixture
def subtitle_generator():
    """Creates a SubtitleGenerator instance for testing."""
    return SubtitleGenerator()

@pytest.fixture
def mock_segments():
    """Creates mock segments for testing subtitle generation."""
    mock_segment1 = MagicMock()
    mock_segment1.start = 0.0
    mock_segment1.end = 5.32
    mock_segment1.text = "This is the first segment."
    
    mock_segment2 = MagicMock()
    mock_segment2.start = 5.5
    mock_segment2.end = 10.2
    mock_segment2.text = "This is the second segment."
    
    mock_segment3 = MagicMock()
    mock_segment3.start = 10.5
    mock_segment3.end = 15.0
    mock_segment3.text = " This has leading and trailing spaces. "
    
    return [mock_segment1, mock_segment2, mock_segment3]

def test_format_timestamp(subtitle_generator):
    """Tests the timestamp formatting."""
    # Test with seconds
    result = subtitle_generator._format_timestamp(5.32, delimiter=",")
    assert result == "0:00:05,320"
    
    # Test with minutes
    result = subtitle_generator._format_timestamp(65.5, delimiter=".")
    assert result == "0:01:05.500"
    
    # Test with hours
    result = subtitle_generator._format_timestamp(3661.5, delimiter=",")
    assert result == "1:01:01,500"
    
    # Test with no decimal part
    result = subtitle_generator._format_timestamp(10.0, delimiter=",")
    assert result == "0:00:10,000"

def test_generate_srt(subtitle_generator, mock_segments):
    """Tests SRT generation."""
    srt_content = subtitle_generator.generate(mock_segments, SubtitleFormat.SRT)
    
    # Verify format
    lines = srt_content.strip().split("\n")
    assert lines[0] == "1"  # First segment number
    assert lines[1] == "0:00:00,000 --> 0:00:05,320"  # First timestamp
    assert lines[2] == "This is the first segment."  # First text
    
    assert lines[4] == "2"  # Second segment number
    assert lines[5] == "0:00:05,500 --> 0:00:10,200"  # Second timestamp
    
    assert lines[8] == "3"  # Third segment number
    assert lines[10] == "This has leading and trailing spaces."  # Third text (trimmed)

def test_generate_vtt(subtitle_generator, mock_segments):
    """Tests VTT generation."""
    vtt_content = subtitle_generator.generate(mock_segments, SubtitleFormat.VTT)
    
    # Verify format
    lines = vtt_content.strip().split("\n")
    assert lines[0] == "WEBVTT"  # Header
    assert lines[2] == "0:00:00.000 --> 0:00:05.320"  # First timestamp
    assert lines[3] == "This is the first segment."  # First text
    
    assert lines[5] == "0:00:05.500 --> 0:00:10.200"  # Second timestamp
    
    assert lines[9] == "This has leading and trailing spaces."  # Third text (trimmed)

def test_generate_empty_segments(subtitle_generator):
    """Tests subtitle generation with empty segments."""
    srt_content = subtitle_generator.generate([], SubtitleFormat.SRT)
    vtt_content = subtitle_generator.generate([], SubtitleFormat.VTT)
    
    assert srt_content == ""
    assert vtt_content == "WEBVTT\n\n"

def test_generate_unsupported_format(subtitle_generator, mock_segments):
    """Tests subtitle generation with unsupported format."""
    class UnsupportedFormat:
        pass
        
    result = subtitle_generator.generate(mock_segments, UnsupportedFormat())
    assert result == ""
