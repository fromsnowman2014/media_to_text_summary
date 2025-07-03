"""
Tests for the text chunking utilities.
"""

import unittest
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from extract_transcript.text_chunk_utils import (
    estimate_token_count,
    split_text_into_chunks,
    merge_translated_chunks
)

class TestTextChunkUtils(unittest.TestCase):
    def test_token_estimation(self):
        """Test token estimation function"""
        # Simple cases
        self.assertEqual(estimate_token_count("hello world"), 2)
        self.assertEqual(estimate_token_count(""), 0)
        
        # More complex case
        text = "This is a longer text with multiple words. It should have around 15 tokens."
        estimated = estimate_token_count(text)
        # We expect around 15 tokens (12 words * 1.2 = ~14-15)
        self.assertTrue(14 <= estimated <= 16)
        
    def test_split_text_small_chunks(self):
        """Test splitting text into small chunks"""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = split_text_into_chunks(text, max_tokens=5)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], "First paragraph.")
        self.assertEqual(chunks[1], "Second paragraph.")
        self.assertEqual(chunks[2], "Third paragraph.")
        
    def test_split_large_paragraph(self):
        """Test splitting a large paragraph into sentence chunks"""
        large_paragraph = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        chunks = split_text_into_chunks(large_paragraph, max_tokens=5)
        self.assertEqual(len(chunks), 2)
        self.assertTrue("sentence one" in chunks[0])
        self.assertTrue("sentence two" in chunks[0])
        self.assertTrue("sentence three" in chunks[1])
        self.assertTrue("sentence four" in chunks[1])
        
    def test_merge_chunks(self):
        """Test merging translated chunks"""
        chunks = ["First chunk", "Second chunk", "Third chunk"]
        merged = merge_translated_chunks(chunks)
        self.assertEqual(merged, "First chunk\n\nSecond chunk\n\nThird chunk")
        
    def test_realistic_text_splitting(self):
        """Test with a more realistic text example"""
        text = """
        This is the first paragraph of the text. It contains multiple sentences.
        Each sentence should be properly analyzed for tokens.
        
        Here is the second paragraph. It's a bit shorter.
        
        And finally the third paragraph, which is also not very long.
        
        But the fourth paragraph is much longer. It contains many sentences that should probably be split.
        This is because we want to ensure that the algorithm correctly breaks down paragraphs that exceed
        the specified token limit. In real-world scenarios, we might have transcripts with very long paragraphs
        that need to be handled gracefully. This paragraph should definitely be split if our max tokens is around 20-30.
        """
        
        # Test with a low token limit to force splitting
        chunks = split_text_into_chunks(text, max_tokens=25)
        # We expect more chunks than paragraphs
        self.assertTrue(len(chunks) > 4)
        
        # Test that we can put it back together
        merged = merge_translated_chunks(chunks)
        # The merged text should contain the same content, though spacing may vary
        for phrase in ["first paragraph", "second paragraph", "third paragraph", "fourth paragraph"]:
            self.assertIn(phrase, merged)


if __name__ == "__main__":
    unittest.main()
