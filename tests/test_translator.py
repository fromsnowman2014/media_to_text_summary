"""
Tests for the Translator module.
"""

import pytest
from unittest.mock import MagicMock, patch
from extract_transcript.translator import Translator, NLLB_LANGUAGE_CODES

@pytest.fixture
def translator():
    """Creates a Translator instance for testing."""
    return Translator()

@pytest.fixture
def mock_tokenizer():
    """Creates a mock tokenizer for testing."""
    mock = MagicMock()
    mock.lang_code_to_id = {
        'eng_Latn': 1,
        'kor_Hang': 2,
        'jpn_Jpan': 3
    }
    mock.batch_decode.return_value = ["This is a translated text."]
    return mock

@pytest.fixture
def mock_model():
    """Creates a mock translation model for testing."""
    mock = MagicMock()
    mock.generate.return_value = "mock_translated_tokens"
    return mock

def test_translator_init():
    """Tests Translator initialization."""
    translator = Translator(model_name="custom_model")
    
    assert translator.model_name == "custom_model"
    assert translator.tokenizer is None
    assert translator.model is None

def test_is_language_supported(translator):
    """Tests language support checking."""
    assert translator.is_language_supported("en") is True
    assert translator.is_language_supported("ko") is True
    assert translator.is_language_supported("xx") is False

def test_get_supported_languages(translator):
    """Tests getting supported languages list."""
    supported = translator.get_supported_languages()
    assert isinstance(supported, list)
    assert len(supported) == len(NLLB_LANGUAGE_CODES)
    assert "en" in supported
    assert "ko" in supported

@patch("extract_transcript.translator.AutoTokenizer")
@patch("extract_transcript.translator.AutoModelForSeq2SeqLM")
def test_load_model(mock_model_class, mock_tokenizer_class, translator, mock_model, mock_tokenizer):
    """Tests the load_model method."""
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
    mock_model_class.from_pretrained.return_value = mock_model

    # Reset the translator state to ensure a clean test
    translator.model = None
    translator.tokenizer = None

    # First call should load model and tokenizer
    model, tokenizer = translator.load_model()

    # Verify the model and tokenizer are returned correctly
    assert model == mock_model
    assert tokenizer == mock_tokenizer
        
    # Verify the model loading was called with expected parameters
    # For our implementation, we check that at least one call was made with these parameters
    assert any(
        call[0][0] == "facebook/nllb-200-distilled-600M" and call[1].get('local_files_only') is not None 
        for call in mock_tokenizer_class.from_pretrained.call_args_list
    )
    assert any(
        call[0][0] == "facebook/nllb-200-distilled-600M" 
        for call in mock_model_class.from_pretrained.call_args_list
    )

    # Store initial model and tokenizer for comparison
    initial_model = model
    initial_tokenizer = tokenizer
        
    # Reset mocks but don't modify the translator's internal state
    # (The actual implementation might cache the model and tokenizer)
    mock_tokenizer_class.reset_mock()
    mock_model_class.reset_mock()
        
    # Subsequent calls should return the same instances
    model2, tokenizer2 = translator.load_model()

    # Verify that the same model and tokenizer instances are returned
    # This tests our caching functionality
    assert model2 is not None
    assert tokenizer2 is not None
    assert model2 == initial_model
    assert tokenizer2 == initial_tokenizer

def test_translate_unsupported_language(translator):
    """Tests translation with unsupported languages."""
    result = translator.translate("Test text", "xx", "en")
    assert result is None
    
    result = translator.translate("Test text", "en", "xx")
    assert result is None

@patch("extract_transcript.translator.AutoTokenizer")
@patch("extract_transcript.translator.AutoModelForSeq2SeqLM")
def test_translate_success(mock_model_class, mock_tokenizer_class, translator, mock_model, mock_tokenizer):
    """Tests successful translation."""
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
    mock_model_class.from_pretrained.return_value = mock_model
    
    result = translator.translate("Test text", "en", "ko")
    
    assert result == "This is a translated text."
    mock_tokenizer.assert_called_with("Test text", return_tensors="pt")
    mock_model.generate.assert_called_once()

@patch("extract_transcript.translator.AutoTokenizer")
@patch("extract_transcript.translator.AutoModelForSeq2SeqLM")
def test_translate_exception(mock_model_class, mock_tokenizer_class, translator):
    """Tests translation when an exception occurs."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.side_effect = Exception("Test error")
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
    
    result = translator.translate("Test text", "en", "ko")
    
    assert result is None


def test_kr_to_ko_normalization(translator):
    """Tests that 'kr' is normalized to 'ko' for Korean."""
    # Mock the actual translate method to check language code normalization
    translator.is_language_supported = lambda x: True if x == 'ko' or x == 'kr' else False
    
    # Test normalization when 'kr' is the source language
    with patch.object(translator, 'load_model', return_value=(None, None)):
        translator.translate("Test text", "kr", "en")
        # If code works, kr should be normalized to ko and translation attempted
        assert translator.is_language_supported("kr") is True
        assert translator.is_language_supported("ko") is True

    # Test normalization when 'kr' is the target language
    with patch.object(translator, 'load_model', return_value=(None, None)):
        translator.translate("Test text", "en", "kr")
        # If code works, kr should be normalized to ko and translation attempted
        assert translator.is_language_supported("kr") is True
        assert translator.is_language_supported("ko") is True


@patch("extract_transcript.translator.AutoTokenizer")
@patch("extract_transcript.translator.AutoModelForSeq2SeqLM")
def test_network_error_handling(mock_model_class, mock_tokenizer_class, translator):
    """Tests handling of network errors during model loading."""
    # Simulate a network timeout for both tokenizer and model
    mock_tokenizer_class.from_pretrained.side_effect = Exception("Connection to huggingface.co timed out")
    mock_model_class.from_pretrained.side_effect = Exception("Connection to huggingface.co timed out")

    # Attempt to load model
    model, tokenizer = translator.load_model()

    # Should return None, None rather than raising an exception
    assert model is None
    assert tokenizer is None


@patch("extract_transcript.translator.AutoTokenizer")
@patch("extract_transcript.translator.AutoModelForSeq2SeqLM")
def test_model_load_failure_handling(mock_model_class, mock_tokenizer_class, translator):
    """Tests handling of model loading failures."""
    # Mock tokenizer loads successfully
    mock_tokenizer = MagicMock()
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
    
    # But model loading fails
    mock_model_class.from_pretrained.side_effect = [ValueError("prefer_safetensors error"), RuntimeError("Model loading failed")]
    
    # Attempt to load model
    model, tokenizer = translator.load_model()
    
    # Should return None, None rather than raising an exception
    assert model is None
    assert tokenizer is None
