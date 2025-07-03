"""
Summarizer module for generating summaries from transcribed text.
"""

import logging
import os
import gc
import time
import torch
from transformers import pipeline

# Set environment variables for better stability
os.environ['TRANSFORMERS_IGNORE_TORCH_VERSION'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable parallelism to reduce memory usage

# Set a flag to enable debug mode for summarization
SUMMARIZE_DEBUG = os.environ.get('SUMMARIZE_DEBUG', '1') == '1'

class Summarizer:
    """A class to handle text summarization."""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        Initialize a Summarizer instance.
        
        Args:
            model_name: The name of the summarization model to use
        """
        self.model_name = model_name
        self.summarizer = None
    
    def _cleanup_memory(self):
        """Clean up memory to prevent OOM errors and segmentation faults.
        
        This aggressively cleans memory before loading potentially large models.
        """
        # Free any loaded models from memory
        if hasattr(self, 'summarizer') and self.summarizer is not None:
            try:
                del self.summarizer
            except Exception as e:
                logging.warning(f"Error when cleaning up summarizer: {e}")
            self.summarizer = None
        
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
        
        # Give system some time to free memory
        time.sleep(0.5)
    
    def load_model(self):
        """Load the summarization model with memory management."""
        # Clean up memory first
        self._cleanup_memory()
        
        if self.summarizer is None:
            logging.info(f"Loading summarization model '{self.model_name}' with memory-efficient settings...")
            try:
                # First try loading with memory-efficient settings
                self.summarizer = pipeline(
                    "summarization", 
                    model=self.model_name, 
                    device=-1,  # Force CPU usage to avoid GPU memory issues
                    model_kwargs={
                        'low_cpu_mem_usage': True,
                        'torch_dtype': torch.float32  # Use 32-bit precision for stability
                    }
                )
                logging.info("Summarization model loaded successfully.")
            except Exception as e:
                logging.warning(f"First attempt to load summarization model failed: {e}")
                self._cleanup_memory()  # Clean memory before next attempt
                try:
                    # Try with more aggressive memory settings
                    self.summarizer = pipeline(
                        "summarization", 
                        model=self.model_name,
                        device=-1,  # Force CPU
                        truncation=True,
                        max_length=512,  # Limit max length to save memory
                        model_kwargs={
                            'low_cpu_mem_usage': True,
                            'torch_dtype': torch.float16,  # Try half precision
                            'offload_folder': "tmp_offload"  # Offload to disk if needed
                        }
                    )
                    logging.info("Summarization model loaded successfully with aggressive memory settings.")
                except Exception as e3:
                    logging.error(f"All attempts to load summarization model failed: {e3}")
                    self._cleanup_memory()
                    return None  # Return None instead of raising exception
        return self.summarizer
    
    def split_text_for_summarization(self, text, max_chunk_size=1000):
        """
        Split text into manageable chunks for summarization.
        
        Args:
            text: The text to split
            max_chunk_size: Maximum chunk size in characters
            
        Returns:
            List of text chunks
        """
        # Simple splitting by paragraphs
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def summarize(self, text: str, max_length: int = 150, min_length: int = 30):
        """
        Generate a summary of the given text with careful memory management.
        
        Args:
            text: The text to summarize
            max_length: Maximum length of the summary in tokens
            min_length: Minimum length of the summary in tokens
            
        Returns:
            Summary text or None if summarization failed
        """
        # Initial memory cleanup
        self._cleanup_memory()
        
        try:
            logging.debug("About to load summarization model")
            summarizer = self.load_model()
            logging.debug("Summarization model loaded successfully")
            
            if summarizer is None:
                logging.error("Cannot summarize: model failed to load.")
                return None
                
            # Debug mode: Set a smaller threshold for splitting
            chunk_threshold = 1000 if SUMMARIZE_DEBUG else 5000
                
            # For long texts, summarize in chunks
            if len(text) > chunk_threshold:
                logging.info(f"Long text detected ({len(text)} chars), using chunked summarization...")
                
                # Use smaller chunks for better stability
                chunks = self.split_text_for_summarization(text, max_chunk_size=500)
                logging.info(f"Split text into {len(chunks)} smaller chunks for summarization")
                
                # In debug mode, only process the first chunk
                if SUMMARIZE_DEBUG and len(chunks) > 2:
                    logging.info("Debug mode enabled: only summarizing first chunk")
                    chunks = chunks[:1]
                
                # Summarize each chunk with memory management
                summaries = []
                for i, chunk in enumerate(chunks):
                    logging.info(f"Summarizing chunk {i+1}/{len(chunks)}")
                    try:
                        # Very conservative settings for stability
                        chunk_max_length = min(60, max(30, max_length // (len(chunks) or 1)))
                        
                        # Use a try-except block with memory cleanup
                        try:
                            logging.debug(f"About to run inference on chunk {i+1}")
                            with torch.no_grad():  # Disable gradient calculation for inference
                                result = summarizer(chunk, 
                                                   max_length=chunk_max_length, 
                                                   min_length=min(15, min_length//2), 
                                                   do_sample=False,
                                                   truncation=True)
                            logging.debug(f"Inference completed for chunk {i+1}")
                            
                            if result and len(result) > 0:
                                summaries.append(result[0]['summary_text'])
                            else:
                                logging.warning(f"Empty result for chunk {i+1}")
                                summaries.append(f"[Summary failed for segment {i+1}]")
                        except Exception as chunk_err:
                            logging.error(f"Error summarizing chunk {i+1}: {chunk_err}")
                            summaries.append(f"[Error summarizing segment {i+1}]")
                            
                        # Clean memory after each chunk
                        logging.debug(f"Cleaning memory after chunk {i+1}")
                        self._cleanup_memory()
                    except Exception as e:
                        logging.error(f"Outer error summarizing chunk {i+1}: {e}")
                        summaries.append(f"[Error processing segment {i+1}]")
                        self._cleanup_memory()
                
                # Return combined summary without re-summarizing (more stable)
                combined_summary = " ".join(summaries)
                return combined_summary
            else:
                # For shorter texts, use direct summarization with memory safety
                try:
                    logging.info(f"Summarizing short text ({len(text)} chars)")
                    logging.debug("About to run inference on short text")
                    with torch.no_grad():
                        result = summarizer(text, 
                                           max_length=min(max_length, 100), 
                                           min_length=min_length, 
                                           do_sample=False,
                                           truncation=True)
                    logging.debug("Inference completed for short text")
                    
                    if result and len(result) > 0:
                        return result[0]['summary_text']
                    return None
                except Exception as short_err:
                    logging.error(f"Error summarizing short text: {short_err}")
                    return f"[Summary generation failed: {str(short_err)[:100]}...]"
        except Exception as e:
            logging.error(f"An error occurred during summarization: {e}")
            return f"[Summarization error: {str(e)[:100]}...]"
        finally:
            # Final memory cleanup
            logging.debug("Final memory cleanup")
            self._cleanup_memory()
