"""
Tests for the Summarizer module.
"""

import pytest
from unittest.mock import MagicMock, patch
from extract_transcript.summarizer import Summarizer

@pytest.fixture
def summarizer():
    """Creates a Summarizer instance for testing."""
    return Summarizer()

@pytest.fixture
def mock_pipeline():
    """Creates a mock pipeline for testing."""
    mock = MagicMock()
    mock.return_value = [{"summary_text": "This is a summary."}]
    return mock

def test_summarizer_init():
    """Tests Summarizer initialization."""
    summarizer = Summarizer(model_name="custom_model")
    
    assert summarizer.model_name == "custom_model"
    assert summarizer.summarizer is None

@patch("extract_transcript.summarizer.pipeline")
def test_load_model(mock_pipeline_fn, summarizer, mock_pipeline):
    """Tests the load_model method."""
    mock_pipeline_fn.return_value = mock_pipeline
    
    # First call should create the pipeline
    model = summarizer.load_model()
    
    mock_pipeline_fn.assert_called_once_with(
        "summarization", model="facebook/bart-large-cnn"
    )
    assert model == mock_pipeline
    
    # Subsequent calls should reuse the pipeline
    mock_pipeline_fn.reset_mock()
    model2 = summarizer.load_model()
    
    assert model2 == model
    mock_pipeline_fn.assert_not_called()

@patch("extract_transcript.summarizer.pipeline")
def test_summarize_success(mock_pipeline_fn, summarizer, mock_pipeline):
    """Tests successful summarization."""
    mock_pipeline_fn.return_value = mock_pipeline
    
    result = summarizer.summarize("This is a long text that needs to be summarized.", max_length=50)
    
    assert result == "This is a summary."
    mock_pipeline.assert_called_once_with(
        "This is a long text that needs to be summarized.",
        max_length=50,
        min_length=30,
        do_sample=False
    )

@patch("extract_transcript.summarizer.pipeline")
def test_summarize_with_custom_length(mock_pipeline_fn, summarizer, mock_pipeline):
    """Tests summarization with custom max and min lengths."""
    mock_pipeline_fn.return_value = mock_pipeline
    
    result = summarizer.summarize("Test text", max_length=100, min_length=20)
    
    assert result == "This is a summary."
    mock_pipeline.assert_called_once_with(
        "Test text", max_length=100, min_length=20, do_sample=False
    )

@patch("extract_transcript.summarizer.pipeline")
def test_summarize_empty_result(mock_pipeline_fn, summarizer):
    """Tests summarization when the result is empty."""
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = []  # Empty result
    mock_pipeline_fn.return_value = mock_pipeline
    
    result = summarizer.summarize("Test text")
    
    assert result is None

@patch("extract_transcript.summarizer.pipeline")
def test_summarize_exception(mock_pipeline_fn, summarizer):
    """Tests summarization when an exception occurs."""
    mock_pipeline = MagicMock()
    mock_pipeline.side_effect = Exception("Test error")
    mock_pipeline_fn.return_value = mock_pipeline
    
    result = summarizer.summarize("Test text")
    
    assert result is None
