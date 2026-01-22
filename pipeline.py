from download_video import download_video
from download_audio import extract_audio
from transcribe import transcribe
from extract_output import extract_output


def run_pipeline(youtube_url, language):
    """
    Central AI pipeline controller.
    """

    print("\n Starting AI Moment Extraction Pipeline\n")

    # Step 1: Download video
    download_video(youtube_url)

    # Step 2: Extract audio from video
    extract_audio()

    # Step 3: Transcription + Translation
    transcribe(language)

    # Step 4: Moment extraction + clip generation
    extract_output()

    print("\n Pipeline completed successfully!\n")


# OPTIONAL: terminal testing
if __name__ == "__main__":
    test_url = "https://youtu.be/hFr7UsovOBQ"
    test_language = "Malayalam"

    run_pipeline(test_url, test_language)
