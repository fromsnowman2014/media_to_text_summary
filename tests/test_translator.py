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
    
    # First call should load model and tokenizer
    model, tokenizer = translator.load_model()
    
    assert model == mock_model
    assert tokenizer == mock_tokenizer
    mock_tokenizer_class.from_pretrained.assert_called_once_with(
        "facebook/nllb-200-distilled-600M"
    )
    mock_model_class.from_pretrained.assert_called_once_with(
        "facebook/nllb-200-distilled-600M"
    )
    
    # Subsequent calls should reuse loaded instances
    mock_tokenizer_class.from_pretrained.reset_mock()
    mock_model_class.from_pretrained.reset_mock()
    
    model2, tokenizer2 = translator.load_model()
    
    assert model2 == model
    assert tokenizer2 == tokenizer
    mock_tokenizer_class.from_pretrained.assert_not_called()
    mock_model_class.from_pretrained.assert_not_called()

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
