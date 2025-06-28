"""
extract_transcript package.
A tool for processing media and text files, with features including:
- Transcribing audio/video files to text
- Extracting text from document files (.txt, .srt, .pdf)
- Translating content to different languages
- Summarizing content
- Generating subtitles for audio/video
"""

from .transcriber import Transcriber
from .translator import Translator
from .summarizer import Summarizer
from .subtitle_generator import SubtitleGenerator, SubtitleFormat
from .output_writer import OutputWriter
from .text_extractor import TextExtractor

__version__ = '0.1.0'
