"""
Translator module for translating transcribed text using NLLB models.
"""

import logging
import os
import torch
import gc
import time
import subprocess
from typing import List, Tuple, Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, M2M100ForConditionalGeneration

# Set environment variable to disable the torch version check
# This is a temporary solution until torch 2.6.0 becomes available
os.environ['TRANSFORMERS_IGNORE_TORCH_VERSION'] = '1'

# Set environment variables to optimize memory usage
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

# Define language code normalizations
LANGUAGE_ALIASES = {
    'kr': 'ko',  # Korean
    'cn': 'zh',  # Chinese
    'jp': 'ja',  # Japanese
    'gb': 'en',  # British English
    'mx': 'es',  # Mexican Spanish (Spanish)
    'br': 'pt',  # Brazilian Portuguese
    'tw': 'zh'   # Traditional Chinese (mapped to Chinese)
}

# NLLB model uses different language codes
# Mapping from ISO 639-1 codes to NLLB codes
NLLB_LANGUAGE_CODES = {
    'en': 'eng_Latn',
    'ko': 'kor_Hang',
    'kr': 'kor_Hang',  # Support 'kr' as an alias for Korean
    'zh': 'zho_Hans',
    'ja': 'jpn_Jpan',
    'fr': 'fra_Latn',
    'de': 'deu_Latn',
    'es': 'spa_Latn',
}

# Language code mapping for M2M100 models
M2M100_LANGUAGE_CODES = {
    'en': 'en',
    'ko': 'ko',
    'kr': 'ko',  # Support 'kr' as an alias for Korean
    'zh': 'zh',
    'ja': 'ja',
    'fr': 'fr',
    'de': 'de',
    'es': 'es',
    'nl': 'nld_Latn',  # Dutch
    'pl': 'pol_Latn',  # Polish
    'pt': 'por_Latn',  # Portuguese
    'ru': 'rus_Cyrl',  # Russian
    'sv': 'swe_Latn',  # Swedish
    'tr': 'tur_Latn',  # Turkish
    'uk': 'ukr_Cyrl',  # Ukrainian
    'zh': 'zho_Hans'   # Chinese (Simplified)
}

# Define smaller, more memory-efficient models for each language
# Updated to use models that exist on Hugging Face
EFFICIENT_MODELS = {
    # Use small models for better stability
    'ko': 'facebook/m2m100_418M',  # Smaller than NLLB
    'ja': 'facebook/m2m100_418M',
    'zh': 'facebook/m2m100_418M',
    'fr': 'facebook/m2m100_418M',
    'de': 'facebook/m2m100_418M',
    'es': 'facebook/m2m100_418M',
}

# Universal fallback model - significantly smaller than NLLB 600M
FALLBACK_MODEL = 'facebook/m2m100_418M'

# Ultra-lightweight fallback for emergency use
EMERGENCY_MODEL = 'Helsinki-NLP/opus-mt-en-ROMANCE'  # Very small, limited language support

class Translator:
    """A class to handle translation of transcribed text using memory-efficient techniques."""
    
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M"):
        """
        Initialize a Translator instance.
        
        Args:
            model_name: The name of the primary model to use for translation
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.current_loaded_model = None
        
        # Define chunk size for translation (in characters)
        self.max_chunk_size = 300  # Very conservative for better stability
    
    def _cleanup_memory(self):
        """Clean up memory to prevent OOM errors and segmentation faults.
        
        This aggressively cleans memory before loading potentially large models.
        """
        # Free any loaded models from memory
        if hasattr(self, 'model') and self.model is not None:
            try:
                del self.model
            except Exception as e:
                logging.warning(f"Error when cleaning up model: {e}")
            self.model = None
            
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            try:
                del self.tokenizer
            except Exception as e:
                logging.warning(f"Error when cleaning up tokenizer: {e}")
            self.tokenizer = None
            
        # Force multiple garbage collection passes
        for _ in range(3):
            gc.collect()
        
        # Free GPU memory if available
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception as e:
                logging.warning(f"Error when cleaning CUDA memory: {e}")
                
        # Set memory limits to prevent crashes
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable parallelism to reduce memory usage
            
        # Give system some time to free memory
        time.sleep(0.5)
        
    def load_model(self, model_name: str = None, target_lang: str = None) -> Tuple[Optional[Any], Optional[Any]]:
        """Load a translation model optimized for the target language.
        
        Args:
            model_name: Override the model to load
            target_lang: Target language code to help select optimal model
            
        Returns:
            Tuple of (model, tokenizer) or (None, None) if loading failed
        """
        # First clean up existing models to free memory
        self._cleanup_memory()
        
        # Force CPU mode for model loading - more stable than GPU
        use_cpu = True
        original_cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        if use_cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        # Select the best model for this language
        if target_lang in EFFICIENT_MODELS and not model_name:
            # Use language-specific efficient model if available
            model_to_load = EFFICIENT_MODELS[target_lang]
        else:
            # Fall back to specified or default model
            model_to_load = model_name or self.model_name
            
        # Track which model we're loading
        self.current_loaded_model = model_to_load
        logging.info(f"Loading translation model '{model_to_load}' with memory-efficient settings...")
        
        # Check if offline mode is enabled
        offline_mode = os.environ.get('TRANSFORMERS_OFFLINE', '0') == '1'
        
        try:
            # Configure model loading with conservative memory settings
            memory_efficient_settings = {
                'local_files_only': offline_mode,
                'low_cpu_mem_usage': True,
                'torch_dtype': torch.float32,  # Use 32-bit precision for stability
            }
            
            # First load the tokenizer with error handling
            try:
                logging.info("Loading tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_to_load,
                    local_files_only=offline_mode,
                    use_fast=False  # Use slower but more reliable tokenizer
                )
                logging.info("Tokenizer loaded successfully")
            except Exception as tokenizer_err:
                logging.warning(f"Standard tokenizer loading failed: {tokenizer_err}")
                self.tokenizer = None
                return None, None
            
            # Then load the model with conservative settings and error handling
            try:
                logging.info("Loading model with conservative settings...")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_to_load,
                    **memory_efficient_settings,
                    device_map="auto" if not use_cpu else "cpu"  # Use CPU for stability
                )
                logging.info(f"Successfully loaded model {model_to_load}")
            except Exception as basic_err:
                logging.warning(f"Model loading with standard settings failed: {basic_err}")
                
                # Try again with more aggressive memory optimization
                try:
                    logging.warning("Attempting to load with more aggressive memory optimization")
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_to_load,
                        **memory_efficient_settings,
                        device_map="cpu",  # Force CPU
                        torch_dtype=torch.float16,  # Try half precision
                        offload_folder="tmp_offload"  # Offload to disk if needed
                    )
                    logging.info("Successfully loaded model with aggressive memory optimization")
                except Exception as memory_err:
                    logging.error(f"Memory-optimized model loading failed: {memory_err}")
                    self._cleanup_memory()
                    return None, None
            
            # For NLLB models, ensure we have the tokenizer configured properly
            if "nllb" in model_to_load.lower() and self.tokenizer:
                # Ensure the NLLB tokenizer is properly configured for the specific languages
                if target_lang in NLLB_LANGUAGE_CODES:
                    logging.info(f"Configuring NLLB tokenizer for {target_lang}: {NLLB_LANGUAGE_CODES[target_lang]}")
            
            # Force model to CPU mode if needed
            if use_cpu and self.model is not None:
                try:
                    self.model = self.model.to("cpu")
                    logging.info("Model moved to CPU")
                except Exception as cpu_err:
                    logging.warning(f"Failed to move model to CPU: {cpu_err}")
                
            return self.model, self.tokenizer
            
        except Exception as e:
            logging.error(f"Failed to load model {model_to_load}: {e}")
            
            # Try fallback models if we haven't already tried the fallback
            if model_to_load != FALLBACK_MODEL:
                logging.info(f"Trying fallback model: {FALLBACK_MODEL}")
                return self.load_model(model_name=FALLBACK_MODEL)
                    
            # All fallbacks failed
            self._cleanup_memory()
            return None, None
        finally:
            # Restore original CUDA settings
            if use_cpu and original_cuda_visible_devices is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible_devices
        
    def is_language_supported(self, language_code: str) -> bool:
        """Check if a language code is supported for translation."""
        return language_code in NLLB_LANGUAGE_CODES
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """
        Split a large text into very small chunks for more stable translation.
        
        We use a character-based approach rather than word-based to ensure
        consistent chunk sizes that won't overwhelm the translation model.
        
        Args:
            text: The text to split
            
        Returns:
            List of text chunks
        """
        # Handle empty text
        if not text or not text.strip():
            return []
            
        chunks = []
        sentences = text.replace('\n', ' ').split('.')
        current_chunk = ''
        
        for current_sentence in sentences:
            # Clean up the sentence
            current_sentence = current_sentence.strip() + '.'
            if len(current_sentence) <= 1:  # Just a period or empty
                continue
                
            # If adding this sentence would exceed chunk size, start a new chunk
            if len(current_chunk) + len(current_sentence) > self.max_chunk_size:
                if current_chunk:  # Don't add empty chunks
                    chunks.append(current_chunk)
                current_chunk = current_sentence
            else:
                current_chunk += ' ' + current_sentence if current_chunk else current_sentence
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    def _translate_single_chunk_safely(self, chunk: str, src_lang: str, target_lang: str) -> Optional[str]:
        """
        Translate a single small chunk of text with aggressive memory management.
        
        Args:
            chunk: Text chunk to translate
            src_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text or None if translation failed
        """
        if not chunk or not chunk.strip():
            return ""
        
        # If chunk is too large, split it immediately to prevent memory issues
        if len(chunk) > 100:  # Use much smaller chunks to prevent crashes
            logging.info(f"Pre-emptively splitting large chunk of size {len(chunk)}")
            half = len(chunk) // 2
            first = self._translate_single_chunk_safely(chunk[:half], src_lang, target_lang)
            # Force memory cleanup between chunks
            self._cleanup_memory()
            second = self._translate_single_chunk_safely(chunk[half:], src_lang, target_lang)
            if first is not None and second is not None:
                return first + " " + second
            elif first is not None:
                return first
            elif second is not None:
                return second
            else:
                return None
        
        # For small chunks, proceed with translation
        try:
            # Load model if needed
            model, tokenizer = self.load_model(target_lang=target_lang)
            
            if model is None or tokenizer is None:
                logging.error("Cannot translate: model or tokenizer failed to load.")
                return None
            
            # Convert language codes to model-specific format if needed
            model_src_lang = src_lang
            model_target_lang = target_lang
            
            # For NLLB models, convert language codes to NLLB format
            if "nllb" in self.current_loaded_model.lower():
                if src_lang in NLLB_LANGUAGE_CODES:
                    model_src_lang = NLLB_LANGUAGE_CODES[src_lang]
                if target_lang in NLLB_LANGUAGE_CODES:
                    model_target_lang = NLLB_LANGUAGE_CODES[target_lang]
                    
            try:
                # Use extremely conservative settings for translation
                if "nllb" in self.current_loaded_model.lower():
                    # For NLLB models with special language code handling
                    logging.info(f"Using NLLB translation from {model_src_lang} to {model_target_lang}")
                    
                    # Process inputs
                    try:
                        inputs = tokenizer(chunk, return_tensors="pt")
                        
                        # Move inputs to CPU for safety
                        for k in inputs.keys():
                            if hasattr(inputs[k], 'to'):
                                inputs[k] = inputs[k].to('cpu')
                    except Exception as input_err:
                        logging.error(f"Error processing inputs: {input_err}")
                        return None
                    
                    # Generate with conservative settings
                    try:
                        generation_kwargs = {
                            'max_length': 100,  # Smaller max length
                            'num_beams': 1,    # Greedy search (faster)
                            'do_sample': False, # Deterministic output
                            'early_stopping': True
                        }
                        
                        # Add forced BOS token if available
                        if hasattr(tokenizer, 'lang_code_to_id') and model_target_lang in tokenizer.lang_code_to_id:
                            generation_kwargs['forced_bos_token_id'] = tokenizer.lang_code_to_id[model_target_lang]
                            
                        # Disable gradient calculation for inference
                        with torch.no_grad():
                            translated_tokens = model.generate(**inputs, **generation_kwargs)
                            
                    except Exception as gen_err:
                        logging.error(f"Error during generation: {gen_err}")
                        return None
                        
                elif "m2m100" in self.current_loaded_model.lower():
                    # For M2M100 models with different language code handling
                    model_src_lang = M2M100_LANGUAGE_CODES.get(src_lang, src_lang)
                    model_target_lang = M2M100_LANGUAGE_CODES.get(target_lang, target_lang)
                    logging.info(f"Using M2M100 translation from {model_src_lang} to {model_target_lang}")
                    
                    try:
                        # Special handling for M2M100 models with safer memory handling
                        tokenizer.src_lang = model_src_lang
                        
                        # Process input with explicit CPU placement and error handling
                        try:
                            # Explicitly limit the input length to prevent memory issues
                            max_input_length = 50  # Very conservative for stability
                            if len(chunk) > max_input_length:
                                logging.warning(f"Input chunk too long ({len(chunk)} chars), truncating to {max_input_length}")
                                chunk = chunk[:max_input_length]
                            
                            # Tokenize with safety limits
                            inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=60)
                            
                            # Force inputs to CPU to prevent CUDA memory issues
                            for key in inputs:
                                if hasattr(inputs[key], 'to'):
                                    inputs[key] = inputs[key].to('cpu')
                        except Exception as tok_err:
                            logging.error(f"Tokenization error: {tok_err}")
                            return f"[Translation error: {str(tok_err)[:50]}...]" 
                        
                        # M2M100 requires forced_bos_token_id to be set to the target language token
                        # Use ultra-conservative settings to prevent crashes
                        generation_kwargs = {
                            'max_length': 50,     # Much smaller for stability
                            'num_beams': 1,      # Greedy search (minimal memory)
                            'do_sample': False,  # Deterministic output
                            'early_stopping': True,
                            'use_cache': False   # Disable caching to reduce memory usage
                        }
                        
                        # Add forced BOS token with error handling
                        try:
                            token_id = tokenizer.get_lang_id(model_target_lang)
                            generation_kwargs['forced_bos_token_id'] = token_id
                        except Exception as token_err:
                            logging.warning(f"Could not set language token ID: {token_err}")
                            # Continue without it as fallback
                        
                        # Generate with enhanced error handling and memory management
                        try:
                            # Ensure model is in eval mode and using CPU
                            if hasattr(model, 'eval'):
                                model.eval()
                                
                            # Force model to CPU if not already there
                            if hasattr(model, 'to') and torch.cuda.is_available():
                                model = model.to('cpu')
                                
                            # Use no_grad and set deterministic mode to reduce memory errors
                            torch.set_grad_enabled(False)  # Redundant with no_grad but safer
                            with torch.no_grad():
                                # Add timeout protection to detect potential hangs
                                # This won't actually timeout but will help debug hangs
                                start_time = time.time()
                                logging.debug("Starting model.generate() call")
                                
                                # Set model to inference mode explicitly
                                if hasattr(model, 'config'):
                                    old_use_cache = getattr(model.config, 'use_cache', None)
                                    model.config.use_cache = False
                                
                                # Perform generation with minimal settings
                                translated_tokens = model.generate(**inputs, **generation_kwargs)
                                
                                # Restore model settings
                                if hasattr(model, 'config') and old_use_cache is not None:
                                    model.config.use_cache = old_use_cache
                                    
                                end_time = time.time()
                                logging.debug(f"model.generate() completed in {end_time - start_time:.2f} seconds")
                        except Exception as gen_err:
                            logging.error(f"Generation error: {gen_err}")
                            # Return a meaningful error message instead of crashing
                            return f"[Translation generation error: {str(gen_err)[:50]}...]"
                            
                    except Exception as m2m_err:
                        logging.error(f"Error during M2M100 translation: {m2m_err}")
                        # Try emergency translation if available
                        return self._emergency_translate(chunk)
                        
                else:
                    # Generic approach for other models
                    inputs = tokenizer(chunk, return_tensors="pt")
                    with torch.no_grad():
                        translated_tokens = model.generate(
                            **inputs, 
                            max_length=100, 
                            num_beams=1
                        )
                
                # Decode the translation safely
                try:
                    translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                    return translation
                except Exception as decode_err:
                    logging.error(f"Error decoding translation: {decode_err}")
                    return None
                    
            except Exception as translation_err:
                logging.error(f"Translation error: {translation_err}")
                return None
                
        except Exception as e:
            logging.error(f"Unexpected error in translation: {e}")
            # Last resort: return the original text if we can't translate
            logging.warning("Returning original text as fallback")
            return chunk
        finally:
            # Aggressive memory cleanup after each chunk
            self._cleanup_memory()
    
    def _emergency_translate(self, text):
        """Last-resort minimal translation using an ultra-lightweight model or service.
        
        This is only used when all other translation attempts have failed.
        It may only provide partial translation or return the original text.
        """
        if not text or len(text) < 5:
            return text
            
        try:
            logging.warning("Attempting emergency translation with ultra-lightweight model")
            
            # Clean memory before attempting emergency translation
            self._cleanup_memory()
            
            # Try to load the emergency model (very small model)
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    EMERGENCY_MODEL, 
                    local_files_only=False,
                    use_fast=False
                )
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    EMERGENCY_MODEL,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16
                )
                
                # Ultra-conservative translation: trim text if needed
                if len(text) > 50:
                    text = text[:50] + "..."
                    
                # Try minimal translation
                inputs = tokenizer(text, return_tensors="pt", max_length=50, truncation=True)
                translated_tokens = model.generate(**inputs, max_length=60)
                translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                
                self._cleanup_memory()
                if translation:
                    return f"[Partial emergency translation] {translation}"
                    
            except Exception as model_err:
                logging.error(f"Emergency model failed: {model_err}")
        except Exception as e:
            logging.error(f"Emergency translation failed: {e}")
            
        # Last resort: return original text with warning
        return f"[Translation unavailable] {text}"
    
    def translate_large_text(self, text: str, src_lang: str, target_lang: str) -> str:
        """Split very long text into chunks for translation."""
        if not text or not text.strip():
            return ""
        
        # DEBUGGING: Skip actual translation to diagnose hanging issue
        logging.warning("DEBUGGING: Skipping actual translation to diagnose hanging issue")
        return "[DEBUG MODE: Translation skipped to diagnose hanging issue. Original text would be translated here.]"
        
        # Code below is temporarily disabled for debugging
        # Set debug mode for testing - only translate one chunk
        os.environ['TRANSLATE_DEBUG'] = '1'
            
        # Use extremely small chunks to reduce memory pressure
        self.max_chunk_size = 50
        
        # Calculate number of chunks
        n = len(text)
        chunk_size = self.max_chunk_size
        chunks = [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]
        num_chunks = len(chunks)
        
        logging.info(f"Split text into {num_chunks} chunks for translation with size {chunk_size}")
        
        # Translate each chunk
        translated_chunks = []
        for i, (start, end) in enumerate(chunks):
            logging.info(f"Translating chunk {i+1}/{num_chunks}")
            chunk = text[start:end]
            logging.debug(f"CHUNK CONTENT: {chunk}")
            
            logging.debug("About to call _translate_single_chunk_safely")
            translated_chunk = self._translate_single_chunk_safely(chunk, src_lang, target_lang)
            logging.debug("Returned from _translate_single_chunk_safely")
            
            if translated_chunk is None:
                logging.warning(f"Failed to translate chunk {i+1}/{num_chunks}, using fallback")
                translated_chunk = f"[Translation failed for this segment] {chunk}"
                
            translated_chunks.append(translated_chunk)
            
            # Clean memory after each chunk
            self._cleanup_memory()
            
            # Stop after first chunk during debugging to prevent crashes
            if i == 0 and os.environ.get('TRANSLATE_DEBUG') == '1':
                logging.info("Debug mode: stopping after first chunk")
                translated_chunks.append("[Debug mode: translation truncated to first chunk]")
                break
                
        # Combine the chunks
        return " ".join(translated_chunks)
    
    def translate(self, text: str, src_lang: str, target_lang: str) -> Optional[str]:
        """
        Translate text from source language to target language using memory-efficient chunked translation.
        
        Args:
            text: The text to translate
            src_lang: Source language code (e.g., 'en')
            target_lang: Target language code (e.g., 'ko')
            
        Returns:
            Translated text or None if translation failed
        """
        # Validate input languages
        if not self.is_language_supported(src_lang):
            logging.error(f"Source language '{src_lang}' is not supported for translation.")
            return None
            
        if not self.is_language_supported(target_lang):
            logging.error(f"Target language '{target_lang}' is not supported for translation.")
            return None
            
        # No need to translate if source and target languages are the same
        if src_lang == target_lang:
            logging.info("Source and target languages are the same. No translation needed.")
            return text
            
        # For small texts, translate directly without chunking
        if len(text) < self.max_chunk_size * 2:
            logging.info("Text is small enough for direct translation")
            return self._translate_single_chunk_safely(text, src_lang, target_lang)
        
        # For longer texts, use chunked translation
        logging.info("Long text detected, using chunked translation...")
        return self.translate_large_text(text, src_lang, target_lang)
            
    def get_supported_languages(self):
        """Return a list of supported language codes."""
        return list(NLLB_LANGUAGE_CODES.keys())
