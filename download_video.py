import yt_dlp


def download_video(video_url, save_path="video/source_video"):
    """
    Downloads a YouTube video and saves it locally.
    Reusable function for pipeline / Streamlit.
    """

    print(" Downloading YouTube video...")

    ydl_options = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": f"{save_path}.%(ext)s",
        "merge_output_format": "webm",
        "quiet": False
    }

    try:
        with yt_dlp.YoutubeDL(ydl_options) as downloader:
            downloader.download([video_url])

        print("✅ Video download completed!")

    except Exception as error:
        print(" Video download failed.")
        print(f"Error details: {error}")


# OPTIONAL: run only if file is executed directly
if __name__ == "__main__":
    test_url = "https://youtu.be/hFr7UsovOBQ"
    download_video(test_url)
