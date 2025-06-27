# Media-to-Text CLI Tool

A powerful command-line tool for transcribing, translating, summarizing, and subtitle generation from audio/video files. The tool leverages state-of-the-art AI models for high-quality text extraction and processing.

## Features

- **Transcription**: Convert audio/video files to text using Faster Whisper
- **Translation**: Translate transcribed text to different languages using Hugging Face Transformers
- **Summarization**: Generate concise summaries of transcripts
- **Subtitle Generation**: Create SRT and VTT subtitle files from transcriptions
- **Batch Processing**: Process multiple files in a directory
- **Flexible Output**: Save results to customizable locations

## Installation

### Requirements

- Python 3.8+
- FFmpeg (for audio processing)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd extract_transcript
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python -m extract_transcript.cli input_file.mp3
```

This will transcribe the audio file and save the result to the `output` directory.

### Advanced Options

```bash
python -m extract_transcript.cli input_file.mp3 \
    --model medium \
    --language en \
    --output_dir my_output \
    --translate_to ko \
    --summarize \
    --summary_length 200 \
    --generate_subtitles \
    --subtitle_format both
```

### Command-line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `input_file` | Path to audio/video file or directory | Required |
| `--model` | Whisper model size (tiny, base, small, medium, large) | base |
| `--language` | Language code for transcription | Auto-detect |
| `--output_dir` | Output directory for results | `output/` |
| `--translate_to` | Target language for translation | None |
| `--summarize` | Generate a summary of the transcript | False |
| `--summary_length` | Maximum summary length in words | 150 |
| `--generate_subtitles` | Generate subtitle files | False |
| `--subtitle_format` | Subtitle format (srt, vtt, or both) | srt |

## Directory Processing

Process all supported audio/video files in a directory:

```bash
python -m extract_transcript.cli path/to/media/folder --generate_subtitles
```

## Architecture

The codebase follows SOLID design principles with separate modules for each responsibility:

- **Transcriber**: Handles audio/video to text conversion
- **Translator**: Manages text translation between languages
- **Summarizer**: Creates concise summaries from transcripts
- **SubtitleGenerator**: Converts segments to SRT/VTT formats
- **OutputWriter**: Manages file system operations for output
- **CLI**: Coordinates the workflow and user interaction

## Supported Formats

### Input Formats
- Audio: mp3, wav, flac, ogg, m4a
- Video: mp4, mov

### Output Formats
- Transcription: Plain text (.txt)
- Translation: Plain text (.txt)
- Summary: Plain text (.txt)
- Subtitles: SRT (.srt) and/or WebVTT (.vtt)

## Development

### Testing

Run the test suite:

```bash
pytest -xvs
```

### Adding New Features

The modular architecture makes it easy to extend functionality:

1. Add a new module in the `extract_transcript` package
2. Update the CLI interface in `cli.py`
3. Write unit tests in the `tests` directory

## License

[MIT License](LICENSE)
