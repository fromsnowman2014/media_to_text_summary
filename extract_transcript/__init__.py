"""
extract_transcript package.
A tool for transcribing audio/video files, translating, summarizing, and generating subtitles.
"""

from .transcriber import Transcriber
from .translator import Translator
from .summarizer import Summarizer
from .subtitle_generator import SubtitleGenerator, SubtitleFormat
from .output_writer import OutputWriter

__version__ = '0.1.0'
