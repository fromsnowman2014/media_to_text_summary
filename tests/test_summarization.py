import pytest
from unittest.mock import patch, MagicMock
from extract_transcript.main import summarize_text

@pytest.fixture
def mock_pipeline(monkeypatch):
    """Mocks the transformers pipeline for summarization."""
    mock_summarizer = MagicMock()
    mock_summarizer.return_value = [{'summary_text': 'This is a mock summary.'}]
    
    mock_pipeline_fn = MagicMock(return_value=mock_summarizer)
    monkeypatch.setattr("extract_transcript.main.pipeline", mock_pipeline_fn)
    
    return mock_pipeline_fn, mock_summarizer

def test_summarize_text_success(mock_pipeline):
    """Tests successful text summarization."""
    mock_pipeline_fn, mock_summarizer = mock_pipeline
    
    result = summarize_text("This is a long text that needs to be summarized.")
    
    assert result == "This is a mock summary."
    mock_pipeline_fn.assert_called_once_with("summarization", model="facebook/bart-large-cnn")
    mock_summarizer.assert_called_once_with(
        "This is a long text that needs to be summarized.",
        max_length=150,
        min_length=30,
        do_sample=False
    )

def test_summarize_text_with_custom_length(mock_pipeline):
    """Tests summarization with custom max_length parameter."""
    mock_pipeline_fn, mock_summarizer = mock_pipeline
    
    result = summarize_text("This is a long text that needs to be summarized.", max_length=100)
    
    assert result == "This is a mock summary."
    mock_summarizer.assert_called_once_with(
        "This is a long text that needs to be summarized.",
        max_length=100,
        min_length=30,
        do_sample=False
    )

def test_summarize_text_empty_result(monkeypatch):
    """Tests handling of empty result from summarizer."""
    mock_summarizer = MagicMock(return_value=[])
    mock_pipeline_fn = MagicMock(return_value=mock_summarizer)
    monkeypatch.setattr("extract_transcript.main.pipeline", mock_pipeline_fn)
    
    result = summarize_text("This text will yield an empty result.")
    
    assert result is None

@patch("extract_transcript.main.pipeline", MagicMock(side_effect=Exception("Model loading failed")))
def test_summarize_text_exception_handling():
    """Tests that exceptions during summarization are handled gracefully."""
    result = summarize_text("This will fail to summarize.")
    assert result is None
