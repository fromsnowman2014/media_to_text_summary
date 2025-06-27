"""
Tests for the OutputWriter module.
"""

import pytest
import os
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
from extract_transcript.output_writer import OutputWriter

@pytest.fixture
def output_writer():
    """Creates an OutputWriter instance for testing."""
    return OutputWriter(Path("/test/output"))

def test_output_writer_init():
    """Tests OutputWriter initialization."""
    # Initialize with no output directory
    writer1 = OutputWriter()
    assert writer1.output_dir is None
    
    # Initialize with output directory
    output_dir = Path("/test/output")
    writer2 = OutputWriter(output_dir)
    assert writer2.output_dir == output_dir

def test_set_output_dir(output_writer):
    """Tests setting the output directory."""
    new_path = Path("/new/path")
    output_writer.set_output_dir(new_path)
    assert output_writer.output_dir == new_path

def test_ensure_output_dir_none():
    """Tests ensuring output directory when directory is not set."""
    writer = OutputWriter()
    writer.ensure_output_dir()  # Should not raise any errors

@patch('pathlib.Path.mkdir')
@patch('pathlib.Path.exists')
def test_ensure_output_dir_exists(mock_exists, mock_mkdir, output_writer):
    """Tests ensuring output directory when directory exists."""
    mock_exists.return_value = True
    
    output_writer.ensure_output_dir()
    
    mock_exists.assert_called_once()
    mock_mkdir.assert_not_called()

@patch('pathlib.Path.mkdir')
@patch('pathlib.Path.exists')
def test_ensure_output_dir_not_exists(mock_exists, mock_mkdir, output_writer):
    """Tests ensuring output directory when directory doesn't exist."""
    mock_exists.return_value = False
    
    output_writer.ensure_output_dir()
    
    mock_exists.assert_called_once()
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

def test_save_no_output_dir():
    """Tests saving when no output directory is set."""
    writer = OutputWriter()
    result = writer.save("test.mp3", "test content", ".txt")
    
    assert result is None

@patch('builtins.open', new_callable=mock_open)
@patch('pathlib.Path.exists')
def test_save_success(mock_exists, mock_open_file, output_writer):
    """Tests successful saving of content."""
    mock_exists.return_value = True
    
    result = output_writer.save("test.mp3", "test content", "_transcript.txt")
    
    assert result == Path("/test/output/test_transcript.txt")
    mock_open_file.assert_called_once_with(
        Path("/test/output/test_transcript.txt"), 'w', encoding='utf-8'
    )
    mock_open_file().write.assert_called_once_with("test content")

@patch('builtins.open')
@patch('pathlib.Path.exists')
def test_save_exception(mock_exists, mock_open_file, output_writer):
    """Tests saving when an exception occurs."""
    mock_exists.return_value = True
    mock_open_file.side_effect = Exception("Test error")
    
    result = output_writer.save("test.mp3", "test content", "_transcript.txt")
    
    assert result is None
