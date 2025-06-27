import argparse
import logging
import datetime
from pathlib import Path
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define language codes for NLLB
NLLB_LANGUAGE_CODES = {
    'en': 'eng_Latn',
    'ko': 'kor_Hang',
    'ja': 'jpn_Jpan',
    'zh': 'zho_Hans',
    'es': 'spa_Latn',
    'fr': 'fra_Latn',
    'de': 'deu_Latn',
}

def transcribe_audio(file_path: str, model_name: str = "base", language: str = None, device: str = "cpu", compute_type: str = "int8"):
    """Transcribes an audio file using faster-whisper."""
    path_obj = Path(file_path)
    if not path_obj.exists():
        logging.error(f"File not found: {file_path}")
        return None, None, None

    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        segments_generator, info = model.transcribe(str(path_obj), beam_size=5, language=language)
        
        segment_list = list(segments_generator)
        full_text = "".join(segment.text for segment in segment_list)
        
        detected_lang = info.language
        logging.info(f"Detected language '{detected_lang}' with probability {info.language_probability:.2f}")
        
        return full_text, segment_list, detected_lang
    except Exception as e:
        logging.error(f"An error occurred during transcription: {e}")
        return None, None, None

def translate_text(text: str, src_lang: str, target_lang: str, model_name: str = "facebook/nllb-200-distilled-600M"):
    """Translates text using a specified NLLB model."""
    if src_lang not in NLLB_LANGUAGE_CODES or target_lang not in NLLB_LANGUAGE_CODES:
        logging.error(f"Translation from '{src_lang}' to '{target_lang}' is not supported.")
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        src_code = NLLB_LANGUAGE_CODES[src_lang]
        target_code = NLLB_LANGUAGE_CODES[target_lang]

        inputs = tokenizer(text, return_tensors="pt")
        translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[target_code])
        
        translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return translated_text
    except Exception as e:
        logging.error(f"An error occurred during translation: {e}")
        return None


def summarize_text(text: str, max_length: int = 150, min_length: int = 30, model_name: str = "facebook/bart-large-cnn"):
    """Summarizes text using a pretrained summarization model."""
    try:
        summarizer = pipeline("summarization", model=model_name)
        result = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        if result and len(result) > 0:
            return result[0]['summary_text']
        return None
    except Exception as e:
        logging.error(f"An error occurred during summarization: {e}")
        return None

def save_output(output_dir: Path, original_filename: str, text: str, suffix: str):
    """Saves the output text to a file with a given suffix."""
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    output_filename = Path(original_filename).stem + suffix
    output_path = output_dir / output_filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    logging.info(f"Output saved to {output_path}")


def segments_to_srt(segments):
    """Convert transcription segments to SRT format."""
    srt_content = ""
    for i, segment in enumerate(segments):
        # Get start and end times from segment (handle both object and dict segments)
        if hasattr(segment, 'start'):
            start_seconds = segment.start
            end_seconds = segment.end
            segment_text = segment.text
        elif isinstance(segment, dict):
            start_seconds = segment['start']
            end_seconds = segment['end']
            segment_text = segment['text']
        else:
            continue  # Skip invalid segments
            
        # Format start and end times as SRT timestamps (HH:MM:SS,mmm)
        start_time = str(datetime.timedelta(seconds=start_seconds))
        if "." not in start_time:
            start_time += ",000"
        else:
            start_time = start_time.replace(".", ",")[:11]
            
        end_time = str(datetime.timedelta(seconds=end_seconds))
        if "." not in end_time:
            end_time += ",000"
        else:
            end_time = end_time.replace(".", ",")[:11]
            
        # Ensure both timestamps have proper formatting (00:00:00,000)
        while len(start_time.split(":")[0]) < 2:
            start_time = "0" + start_time
        while len(end_time.split(":")[0]) < 2:
            end_time = "0" + end_time
            
        # Add sequence number, timestamps and text to SRT
        srt_content += f"{i+1}\n{start_time} --> {end_time}\n{segment_text.strip()}\n\n"
        
    return srt_content


def segments_to_vtt(segments):
    """Convert transcription segments to WebVTT format."""
    vtt_content = "WEBVTT\n\n"
    
    for i, segment in enumerate(segments):
        # Get start and end times from segment (handle both object and dict segments)
        if hasattr(segment, 'start'):
            start_seconds = segment.start
            end_seconds = segment.end
            segment_text = segment.text
        elif isinstance(segment, dict):
            start_seconds = segment['start']
            end_seconds = segment['end']
            segment_text = segment['text']
        else:
            continue  # Skip invalid segments
            
        # Format start and end times as WebVTT timestamps (HH:MM:SS.mmm)
        start_time = str(datetime.timedelta(seconds=start_seconds))
        if "." not in start_time:
            start_time += ".000"
        else:
            start_time = start_time[:11]  # WebVTT uses . instead of , for milliseconds
            
        end_time = str(datetime.timedelta(seconds=end_seconds))
        if "." not in end_time:
            end_time += ".000"
        else:
            end_time = end_time[:11]  # WebVTT uses . instead of , for milliseconds
            
        # Ensure both timestamps have proper formatting (00:00:00.000)
        while len(start_time.split(":")[0]) < 2:
            start_time = "0" + start_time
        while len(end_time.split(":")[0]) < 2:
            end_time = "0" + end_time
            
        # Add timestamps and text to VTT
        vtt_content += f"{start_time} --> {end_time}\n{segment_text.strip()}\n\n"
        
    return vtt_content

def main():
    parser = argparse.ArgumentParser(description="Transcribe and optionally translate audio/video files.")
    parser.add_argument("input_file", type=str, help="Path to the audio/video file or directory.")
    parser.add_argument("--model", type=str, default="base", help="Whisper model size.")
    parser.add_argument("--language", type=str, default=None, help="Language code for transcription.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--translate_to", type=str, default=None, help="Language to translate the text to (e.g., 'en', 'ko').")
    parser.add_argument("--summarize", action="store_true", help="Generate a summary of the transcription.")
    parser.add_argument("--summary_length", type=int, default=150, help="Maximum length of summary in words.")
    parser.add_argument("--generate_subtitles", action="store_true", help="Generate subtitle files from the transcription.")
    parser.add_argument("--subtitle_format", type=str, default="srt", choices=["srt", "vtt", "both"], 
                      help="Format of subtitle file to generate (srt, vtt, or both).")

    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir) if args.output_dir else (input_path.parent if input_path.is_file() else input_path)

    if not input_path.exists():
        logging.error(f"Input path does not exist: {input_path}")
        return

    files_to_process = [input_path] if input_path.is_file() else [f for f in input_path.iterdir() if f.is_file()]

    for file_path in files_to_process:
        logging.info(f"Processing {file_path.name}...")
        full_text, _, detected_lang = transcribe_audio(str(file_path), args.model, args.language)

        if full_text:
            save_output(output_dir, file_path.name, full_text, "_transcription.txt")

            if args.translate_to:
                logging.info(f"Translating from {detected_lang} to {args.translate_to}...")
                translated_text = translate_text(full_text, detected_lang, args.translate_to)
                if translated_text:
                    save_output(output_dir, file_path.name, translated_text, f"_translation_{args.translate_to}.txt")
            
            if args.summarize:
                logging.info("Generating summary of transcription...")
                summary_text = summarize_text(full_text, max_length=args.summary_length)
                if summary_text:
                    save_output(output_dir, file_path.name, summary_text, "_summary.txt")
                    
            if args.generate_subtitles and segments:
                logging.info(f"Generating subtitles in {args.subtitle_format} format...")
                if args.subtitle_format == "srt" or args.subtitle_format == "both":
                    srt_content = segments_to_srt(segments)
                    save_output(output_dir, file_path.name, srt_content, ".srt")
                    
                if args.subtitle_format == "vtt" or args.subtitle_format == "both":
                    vtt_content = segments_to_vtt(segments)
                    save_output(output_dir, file_path.name, vtt_content, ".vtt")

if __name__ == "__main__":
    main()
