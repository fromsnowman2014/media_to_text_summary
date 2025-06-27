#!/bin/bash
# Wrapper script to run the extract_transcript CLI with proper environment settings.
# This script sets KMP_DUPLICATE_LIB_OK=TRUE to avoid OpenMP runtime conflicts.

export KMP_DUPLICATE_LIB_OK=TRUE
python -m extract_transcript.cli "$@"
