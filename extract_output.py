import json
import subprocess
import os
import re

from sentence_transformers import SentenceTransformer, util


#  Configuration 
TRANSCRIPT_PATH = "transcripts/transcript.json"
VIDEO_PATH = "video/source_video.webm"

BUFFER_SECONDS = 15
OUTPUT_DURATION = 30   # seconds


#  NLP Model 
print(" Loading semantic model...")
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")


#  Keyword Definitions 
QUESTION_KEYWORDS = [
    "what", "why", "how", "when", "where",
    "who", "which", "can you", "could you",
    "do you", "is it", "are you"
]

AGREEMENT_KEYWORDS = [
    "i agree", "yes", "right", "correct",
    "exactly", "true", "makes sense", "i think so"
]

DISAGREEMENT_KEYWORDS = [
    "i disagree", "no", "not correct",
    "not true", "wrong", "i dont think so"
]


#  Semantic Intent Templates 
INTENT_TEMPLATES = {
    "agreement": [
        "I agree with that",
        "That is correct",
        "Yes, absolutely",
        "That makes sense"
    ],
    "disagreement": [
        "I disagree with that",
        "That is wrong",
        "I do not agree",
        "That is not correct"
    ]
}

# Precompute intent embeddings (speed)
INTENT_EMBEDDINGS = {
    intent: semantic_model.encode(phrases)
    for intent, phrases in INTENT_TEMPLATES.items()
}


#  Utility Functions 
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text


def load_transcript(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_output_folders():
    os.makedirs("outputs/question_answer", exist_ok=True)
    os.makedirs("outputs/agreement", exist_ok=True)
    os.makedirs("outputs/disagreement", exist_ok=True)


def semantic_score(sentence_emb, intent):
    scores = util.cos_sim(sentence_emb, INTENT_EMBEDDINGS[intent])
    return float(scores.max())


# ---------------- AUDIO FIX (CRITICAL) ----------------
def extract_audio_once():
    """
    Extract full audio ONCE as WAV.
    This avoids WebM + Opus seek issues.
    """
    if not os.path.exists("temp_audio.wav"):
        print("🔹 Extracting master audio...")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", VIDEO_PATH,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "2",
            "temp_audio.wav"
        ])


def extract_video_output(start_time, category, index):
    """
    Guaranteed audio method:
    1) Cut video-only
    2) Cut audio-only
    3) Merge both
    """
    output_start = max(0, start_time - BUFFER_SECONDS)
    output_path = f"outputs/{category}/{category}_{index}.mp4"

    temp_video = "temp_video.mp4"
    temp_audio = "temp_audio_clip.wav"

    #  Video-only clip
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(output_start),
        "-i", VIDEO_PATH,
        "-t", str(OUTPUT_DURATION),
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        temp_video
    ])

    #  Audio-only clip
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(output_start),
        "-i", "temp_audio.wav",
        "-t", str(OUTPUT_DURATION),
        temp_audio
    ])

    #  Merge video + audio
    subprocess.run([
        "ffmpeg", "-y",
        "-i", temp_video,
        "-i", temp_audio,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "128k",
        output_path
    ])

    os.remove(temp_video)
    os.remove(temp_audio)


#  Main Processing 
print("🔹 Loading transcript...")
segments = load_transcript(TRANSCRIPT_PATH)

create_output_folders()
extract_audio_once()

qa_count = 1
agreement_count = 1
disagreement_count = 1

print(" Detecting moments (HYBRID + AUDIO SAFE)...")

for i, segment in enumerate(segments):
    raw_text = segment.get("english_cleaned", "")
    if not raw_text:
        continue

    cleaned_text = clean_text(raw_text)
    start_time = segment["start"]

    #  Question Detection 
    if any(q in cleaned_text for q in QUESTION_KEYWORDS):
        extract_video_output(start_time, "question_answer", qa_count)
        qa_count += 1

        # Answer follows question closely
        if i + 1 < len(segments):
            next_seg = segments[i + 1]
            if next_seg["start"] - segment["end"] < 2.5:
                extract_video_output(
                    next_seg["start"],
                    "question_answer",
                    qa_count
                )
                qa_count += 1
        continue

    # Agreement Detection 
    rule_agree = any(w in cleaned_text for w in AGREEMENT_KEYWORDS)

    if rule_agree:
        agreement_score = 1.0
    else:
        emb = semantic_model.encode(cleaned_text)
        agreement_score = 0.4 * semantic_score(emb, "agreement")

    if agreement_score > 0.6:
        extract_video_output(start_time, "agreement", agreement_count)
        agreement_count += 1
        continue

    #  Disagreement Detection 
    rule_disagree = any(w in cleaned_text for w in DISAGREEMENT_KEYWORDS)

    if rule_disagree:
        disagreement_score = 1.0
    else:
        emb = semantic_model.encode(cleaned_text)
        disagreement_score = 0.4 * semantic_score(emb, "disagreement")

    if disagreement_score > 0.6:
        extract_video_output(start_time, "disagreement", disagreement_count)
        disagreement_count += 1


print(" Extraction completed WITH AUDIO!")
