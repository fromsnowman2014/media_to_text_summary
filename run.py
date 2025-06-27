#!/usr/bin/env python3
"""
Wrapper script to run the extract_transcript CLI with proper environment settings.
This script sets KMP_DUPLICATE_LIB_OK=TRUE to avoid OpenMP runtime conflicts.
"""

import os
import sys
import subprocess

# Set the environment variable to avoid OpenMP conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import and run the CLI module
if __name__ == "__main__":
    from extract_transcript.cli import main
    main()
