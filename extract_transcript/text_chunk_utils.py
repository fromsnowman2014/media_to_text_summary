"""
Utilities for splitting and merging text chunks for improved translation performance.
"""

import re
import logging
from typing import List

def estimate_token_count(text: str) -> int:
    """
    Estimate the token count for a given text.
    This is a simple approximation - actual tokens may vary depending on the tokenizer.
    
    Args:
        text: The text to estimate tokens for
    
    Returns:
        Estimated token count
    """
    # Simple approximation: one token per word + 20% for punctuation and special tokens
    words = text.split()
    return int(len(words) * 1.2)

def split_text_into_chunks(text: str, max_tokens: int = 400) -> List[str]:
    """
    Split text into smaller chunks suitable for translation.
    Attempts to preserve paragraph structure and sentence boundaries.
    
    Args:
        text: Text to split
        max_tokens: Maximum estimated tokens per chunk
        
    Returns:
        List of text chunks
    """
    # First split by paragraphs (double newlines)
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = ""
    current_token_count = 0
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        paragraph_tokens = estimate_token_count(paragraph)
        
        # If adding this paragraph would exceed the limit, save current chunk and start a new one
        if current_token_count + paragraph_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph
            current_token_count = paragraph_tokens
            
        # If the paragraph itself exceeds the limit, split by sentences
        elif paragraph_tokens > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_token_count = 0
                
            # Split the paragraph into sentences
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            
            sentence_chunk = ""
            sentence_token_count = 0
            
            for sentence in sentences:
                sentence_tokens = estimate_token_count(sentence)
                
                # If adding this sentence would exceed the limit, save current sentence chunk
                if sentence_token_count + sentence_tokens > max_tokens and sentence_chunk:
                    chunks.append(sentence_chunk.strip())
                    sentence_chunk = sentence
                    sentence_token_count = sentence_tokens
                # Otherwise, add the sentence to the current chunk
                else:
                    sentence_chunk += " " + sentence if sentence_chunk else sentence
                    sentence_token_count += sentence_tokens
                    
            # Add the last sentence chunk
            if sentence_chunk:
                chunks.append(sentence_chunk.strip())
        # Otherwise, add the paragraph to the current chunk
        else:
            current_chunk += "\n\n" + paragraph if current_chunk else paragraph
            current_token_count += paragraph_tokens
    
    # Add the last chunk if there is one
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

def merge_translated_chunks(chunks: List[str]) -> str:
    """
    Merge translated chunks into a single text.
    
    Args:
        chunks: List of translated text chunks
        
    Returns:
        Merged text
    """
    return "\n\n".join(chunks)
