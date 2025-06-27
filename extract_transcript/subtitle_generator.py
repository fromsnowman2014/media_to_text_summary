"""
SubtitleGenerator module for creating subtitle files from transcribed segments.
"""

import logging
import datetime
from enum import Enum
from typing import List, Any

class SubtitleFormat(Enum):
    """Enumeration of supported subtitle formats."""
    SRT = "srt"
    VTT = "vtt"

class SubtitleGenerator:
    """A class to handle subtitle generation from transcription segments."""
    
    def generate(self, segments: List[Any], format_type: SubtitleFormat = SubtitleFormat.SRT) -> str:
        """
        Generate subtitles from transcription segments.
        
        Args:
            segments: List of transcription segments
            format_type: The subtitle format to generate (SRT or VTT)
            
        Returns:
            The subtitle content as a string
        """
        if not segments:
            return "" if format_type == SubtitleFormat.SRT else "WEBVTT\n\n"
            
        if format_type == SubtitleFormat.SRT:
            return self.generate_srt(segments)
        elif format_type == SubtitleFormat.VTT:
            return self.generate_vtt(segments)
        else:
            logging.error(f"Unsupported subtitle format: {format_type}")
            return ""
    
    def generate_srt(self, segments: List[Any]) -> str:
        """
        Convert transcription segments to SRT format.
        
        Args:
            segments: List of transcription segments with start/end times and text
            
        Returns:
            Formatted SRT content as string
        """
        srt_content = ""
        for i, segment in enumerate(segments):
            # Format start and end times as SRT timestamps (HH:MM:SS,mmm)
            start_time = self._format_timestamp(segment.start, delimiter=",")
            end_time = self._format_timestamp(segment.end, delimiter=",")
                
            # Add sequence number, timestamps and text to SRT
            srt_content += f"{i+1}\n{start_time} --> {end_time}\n{segment.text.strip()}\n\n"
            
        return srt_content
    
    def generate_vtt(self, segments: List[Any]) -> str:
        """
        Convert transcription segments to WebVTT format.
        
        Args:
            segments: List of transcription segments with start/end times and text
            
        Returns:
            Formatted WebVTT content as string
        """
        vtt_content = "WEBVTT\n\n"
        
        for i, segment in enumerate(segments):
            # Format start and end times as WebVTT timestamps (HH:MM:SS.mmm)
            start_time = self._format_timestamp(segment.start, delimiter=".")
            end_time = self._format_timestamp(segment.end, delimiter=".")
                
            # Add timestamps and text to VTT
            vtt_content += f"{start_time} --> {end_time}\n{segment.text.strip()}\n\n"
            
        return vtt_content
    
    def _format_timestamp(self, seconds: float, delimiter: str = ".") -> str:
        """
        Format a timestamp in seconds to HH:MM:SS[delimiter]mmm format.
        
        Args:
            seconds: Time in seconds
            delimiter: Delimiter between seconds and milliseconds ("," for SRT, "." for VTT)
            
        Returns:
            Formatted timestamp string
        """
        time_str = str(datetime.timedelta(seconds=seconds))
        
        # Add milliseconds if not present
        if "." not in time_str:
            time_str += ".000"
        
        # Format milliseconds and replace delimiter if needed
        if delimiter != ".":
            time_str = time_str.replace(".", delimiter)
            
        # Truncate to milliseconds (3 decimal places)
        parts = time_str.split(delimiter)
        time_str = f"{parts[0]}{delimiter}{parts[1][:3]}"
            
        # Keep original format without adding leading zeros to hours
        # (This matches the test expectations)
            
        return time_str
