import subprocess
import os


def extract_audio(
    video_path="video/source_video.webm",
    output_audio="audio/extracted.wav"
):
    """
    Extract audio from a locally downloaded video using FFmpeg.
    Designed for pipeline / Streamlit use.
    """

    print("🎵 Extracting audio from video...")

    os.makedirs(os.path.dirname(output_audio), exist_ok=True)

    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "44100",
                "-ac", "2",
                output_audio
            ],
            check=True
        )

        print("✅ Audio extraction completed!")

    except subprocess.CalledProcessError as error:
        print(" Audio extraction failed.")
        print(error)


# OPTIONAL: direct test
if __name__ == "__main__":
    extract_audio()
