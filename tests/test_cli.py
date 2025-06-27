import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path, PurePath
from extract_transcript.cli import main
from extract_transcript.transcriber import Transcriber
from extract_transcript.output_writer import OutputWriter

@pytest.fixture
def mock_dependencies(monkeypatch):
    """Mocks all external dependencies for CLI tests."""
    # Mock transcriber.transcribe
    mock_transcribe = MagicMock(return_value=("test transcription", [], "en"))
    monkeypatch.setattr(Transcriber, 'transcribe', mock_transcribe)

    # Mock OutputWriter.save
    mock_save = MagicMock()
    monkeypatch.setattr(OutputWriter, 'save', mock_save)
    
    # Mock the process_file function
    mock_process_file = MagicMock(return_value=True)
    monkeypatch.setattr('extract_transcript.cli.process_file', mock_process_file)
    
    return mock_transcribe, mock_save, mock_process_file

@patch('extract_transcript.transcriber.WhisperModel')
@patch('pathlib.Path.exists', return_value=True)
@patch('pathlib.Path.is_file', return_value=True)
@patch('sys.argv')
def test_cli_basic_flow(mock_argv, mock_path_exists, mock_is_file, mock_whisper, mock_dependencies):
    """Tests the CLI with basic arguments."""
    _, _, mock_process_file = mock_dependencies
    
    test_args = ['cli.py', 'audio.mp3']
    mock_argv.__getitem__.side_effect = lambda i: test_args[i]
    mock_argv.__len__.return_value = len(test_args)
    
    main()
    
    # Check that process_file was called
    mock_process_file.assert_called_once()

@pytest.mark.usefixtures("mock_dependencies")
@patch('extract_transcript.transcriber.WhisperModel')
@patch('extract_transcript.translator.AutoTokenizer')
@patch('extract_transcript.translator.AutoModelForSeq2SeqLM')
@patch('extract_transcript.summarizer.pipeline')
@patch('pathlib.Path.exists', return_value=True)
@patch('pathlib.Path.is_file', return_value=True)
@patch('sys.argv')
def test_cli_with_all_options(mock_argv, mock_path_exists, mock_is_file, mock_pipeline, mock_model, mock_tokenizer, mock_whisper, mock_dependencies):
    """Tests the CLI with all optional arguments specified."""
    _, _, mock_process_file = mock_dependencies

    test_args = [
        'cli.py', 
        'audio.mp3', 
        '--model', 'large-v3',
        '--language', 'ko',
        '--output_dir', '/custom/output',
        '--translate_to', 'en',
        '--summarize',
        '--generate_subtitles'
    ]
    mock_argv.__getitem__.side_effect = lambda i: test_args[i]
    mock_argv.__len__.return_value = len(test_args)
    
    main()
    
    # Check that process_file was called
    mock_process_file.assert_called_once()

@patch('extract_transcript.transcriber.WhisperModel')
@patch('sys.argv')
def test_cli_file_not_found(mock_argv, mock_whisper, mock_dependencies):
    """Tests the CLI behavior when the input file does not exist."""
    _, _, mock_process_file = mock_dependencies
    
    test_args = ['cli.py', 'nonexistent.mp3']
    mock_argv.__getitem__.side_effect = lambda i: test_args[i]
    mock_argv.__len__.return_value = len(test_args)
    
    main()
    
    # process_file should not be called for nonexistent files
    mock_process_file.assert_not_called()
