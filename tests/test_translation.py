import pytest
from unittest.mock import patch, MagicMock
from extract_transcript.main import translate_text, NLLB_LANGUAGE_CODES

@pytest.fixture
def mock_transformers(monkeypatch):
    """Mocks the transformers library components."""
    mock_tokenizer_instance = MagicMock()
    mock_tokenizer_instance.return_value = {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
    mock_tokenizer_instance.lang_code_to_id = {"eng_Latn": 256047, "kor_Hang": 256022}
    mock_tokenizer_class = MagicMock(return_value=mock_tokenizer_instance)
    
    mock_model_instance = MagicMock()
    mock_model_instance.generate.return_value = [[256022, 4, 5, 6]] # Dummy translated token IDs
    mock_model_class = MagicMock(return_value=mock_model_instance)

    # Mock the tokenizer's batch_decode to return a specific string
    mock_tokenizer_instance.batch_decode.return_value = ["translated text"]

    monkeypatch.setattr("extract_transcript.main.AutoTokenizer.from_pretrained", mock_tokenizer_class)
    monkeypatch.setattr("extract_transcript.main.AutoModelForSeq2SeqLM.from_pretrained", mock_model_class)

    return mock_tokenizer_class, mock_model_class

def test_translate_text_success(mock_transformers):
    """Tests successful translation."""
    mock_tokenizer, mock_model = mock_transformers
    result = translate_text("Hello world", src_lang='en', target_lang='ko')
    
    assert result == "translated text"
    mock_tokenizer.assert_called_once()
    mock_model.assert_called_once()

def test_translate_text_unsupported_language():
    """Tests translation with an unsupported language code."""
    # Test unsupported source language
    result_unsupported_src = translate_text("Hello", src_lang="unsupported", target_lang="ko")
    assert result_unsupported_src is None

    # Test unsupported target language
    result_unsupported_target = translate_text("Hello", src_lang="en", target_lang="unsupported")
    assert result_unsupported_target is None

@patch("extract_transcript.main.AutoModelForSeq2SeqLM.from_pretrained", MagicMock(side_effect=Exception("Model loading failed")))
def test_translate_text_exception_handling():
    """Tests that exceptions during translation are handled gracefully."""
    result = translate_text("This will fail", src_lang="en", target_lang="ko")
    assert result is None
