import pytest
from unittest.mock import MagicMock
from extract_transcript.main import segments_to_srt, segments_to_vtt

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

def test_segments_to_srt(mock_segments):
    """Tests SRT format conversion from segments."""
    srt_content = segments_to_srt(mock_segments)
    
    # Check if the output has the correct SRT format
    srt_lines = srt_content.strip().split("\n\n")
    assert len(srt_lines) == 3  # 3 segments
    
    # Check first segment
    first_segment = srt_lines[0].split("\n")
    assert first_segment[0] == "1"  # Segment number
    assert first_segment[1] == "00:00:00,000 --> 00:00:05,320"  # Timestamps
    assert first_segment[2] == "This is the first segment."  # Text
    
    # Check second segment
    second_segment = srt_lines[1].split("\n")
    assert second_segment[0] == "2"
    assert second_segment[1] == "00:00:05,500 --> 00:00:10,200"
    assert second_segment[2] == "This is the second segment."
    
    # Check that the third segment has trimmed spaces
    third_segment = srt_lines[2].split("\n")
    assert third_segment[2] == "This has leading and trailing spaces."

def test_segments_to_vtt(mock_segments):
    """Tests WebVTT format conversion from segments."""
    vtt_content = segments_to_vtt(mock_segments)
    
    # Check if the output has the correct WebVTT format
    vtt_lines = vtt_content.split("\n\n")
    assert vtt_lines[0] == "WEBVTT"  # Header
    assert len(vtt_lines) == 5  # Header + 3 segments + empty line at end
    
    # Check first segment
    first_segment = vtt_lines[1].split("\n")
    assert "00:00:00.000 --> 00:00:05.320" in first_segment[0]  # Timestamps with period instead of comma
    assert first_segment[1] == "This is the first segment."  # Text
    
    # Check second segment
    second_segment = vtt_lines[2].split("\n")
    assert "00:00:05.500 --> 00:00:10.200" in second_segment[0]
    assert second_segment[1] == "This is the second segment."
    
    # Check that the third segment has trimmed spaces
    third_segment = vtt_lines[3].split("\n")
    assert third_segment[1] == "This has leading and trailing spaces."

def test_empty_segments():
    """Tests subtitle generation with empty segments."""
    assert segments_to_srt([]) == ""
    assert segments_to_vtt([]) == "WEBVTT\n\n"

def test_long_duration_formatting():
    """Test formatting of timestamps over an hour."""
    segments = [{
        'start': 3661.5,  # 1 hour, 1 minute, 1.5 seconds
        'end': 7322.75,   # 2 hours, 2 minutes, 2.75 seconds
        'text': 'This is a very long video.'
    }]
    
    srt_result = segments_to_srt(segments)
    assert '01:01:01,500 --> 02:02:02,750' in srt_result
    
    vtt_result = segments_to_vtt(segments)
    assert '01:01:01.500 --> 02:02:02.750' in vtt_result
