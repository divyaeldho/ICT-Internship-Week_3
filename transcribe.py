import json
import os
import re
from google.cloud import speech
from google.cloud import translate_v2 as translate
from pydub import AudioSegment


#  CONFIG 
AUDIO_PATH = "audio/extracted.wav"
CHUNK_FOLDER = "audio_chunks"
OUTPUT_PATH = "transcripts/transcript.json"

CHUNK_LENGTH_MS = 30 * 1000  # 30 seconds

LANGUAGE_MAP = {
    "Malayalam": "ml-IN",
    "Tamil": "ta-IN",
    "Hindi": "hi-IN",
    "English": "en-IN"
}



#  AUDIO CHUNKING 
def chunk_audio(audio_path):
    audio = AudioSegment.from_wav(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1)

    os.makedirs(CHUNK_FOLDER, exist_ok=True)
    chunks = []

    for i in range(0, len(audio), CHUNK_LENGTH_MS):
        chunk = audio[i:i + CHUNK_LENGTH_MS]
        chunk_path = f"{CHUNK_FOLDER}/chunk_{i//1000}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append((chunk_path, i / 1000))

    print(f" Created {len(chunks)} audio chunks")
    return chunks


#  GENERAL CLEANING 
def clean_english(text: str) -> str:
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"&[^;]+;", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text.split()) < 3:
        return ""

    return text


#  TRANSLATION 
def translate_to_english(text, translate_client):
    if not text.strip():
        return ""

    result = translate_client.translate(text, target_language="en")
    return result["translatedText"]


#  TRANSCRIPTION CORE 
def transcribe_chunks(chunks, language_code):
    speech_client = speech.SpeechClient()
    translate_client = translate.Client()

    segments = []

    print(f" Transcribing using language: {language_code}")

    for chunk_path, offset in chunks:
        print(f"   → Processing {chunk_path}")

        with open(chunk_path, "rb") as f:
            content = f.read()

        audio = speech.RecognitionAudio(content=content)

        config = speech.RecognitionConfig(
            language_code=language_code,
            enable_automatic_punctuation=True,
            audio_channel_count=1
        )

        response = speech_client.recognize(config=config, audio=audio)

        for result in response.results:
            original_text = result.alternatives[0].transcript.strip()
            if not original_text:
                continue

            english_raw = translate_to_english(original_text, translate_client)
            english_cleaned = clean_english(english_raw)

            segments.append({
                "start": offset,
                "end": offset + 3,
                "original_language_text": original_text,
                "english_translation_raw": english_raw,
                "english_cleaned": english_cleaned
            })

    return segments


# PIPELINE FUNCTION 
def transcribe(language):
    """
    Pipeline-friendly transcription function.
    language: Malayalam | Tamil | Hindi | English
    """

    if language not in LANGUAGE_MAP:
        raise ValueError(f"Unsupported language: {language}")

    language_code = LANGUAGE_MAP[language]
    print(f"\n Selected language: {language}\n")

    chunks = chunk_audio(AUDIO_PATH)
    segments = transcribe_chunks(chunks, language_code)

    if not segments:
        print(" No transcription generated.")
        return

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)

    print(" Transcription + Translation + Cleaning completed")
    print(f" Output saved at: {OUTPUT_PATH}")


# OPTIONAL: direct terminal test
if __name__ == "__main__":
    transcribe("Malayalam")
