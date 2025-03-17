"""A module for managing media files with downloading, playback, recording, and screenshot capabilities.

This module defines the `Media` class, which provides a unified interface for handling various media operations.
It supports downloading media from URLs (e.g., YouTube videos), playing audio and video files, recording audio from
a microphone, capturing video from a webcam, and taking screenshots. The module is designed for ease of use in
media-related applications, with non-blocking playback and error handling.

Key Features:
    - Download media files from URLs using yt-dlp with MP4 output.
    - Play media files asynchronously using VLC with stop control.
    - Record audio (WAV) and video (AVI) with specified durations.
    - Capture screenshots as PNG images.
    - Configurable output directory and optional verbose logging.

Dependencies:
    - yt_dlp: For downloading media from online sources.
    - vlc: For media playback using the VLC library.
    - pyaudio: For recording audio from the microphone.
    - wave: For saving audio recordings in WAV format.
    - cv2: For video recording via OpenCV.
    - pyautogui: For capturing screenshots.
    - threading: For non-blocking media playback.
    - os: For file and directory management.
    - time: For timing operations and delays.

Classes:
    Media: The primary class for interacting with media functionalities.

Constants:
    None explicitly defined, but methods use internal defaults (e.g., output path "../output").

Usage Example:
    ```python
    from media import Media  # Assuming the file is named 'media.py'
    media = Media()
    # Download a video
    file_path = media.download("https://www.youtube.com/watch?v=example")
    # Play the downloaded file
    media.play(file_path)
    # Record 5 seconds of audio
    media.record_audio("output.wav", 5)
    # Record 5 seconds of video
    media.record_video("output.avi", 5)
    # Take a screenshot
    media.screenshot("capture.png")
    # Stop playback if needed
    media.stop_playback()
"""
import yt_dlp
import os
from time import time, sleep
import vlc
import pyaudio
import wave
import cv2
import pyautogui
import threading

class Media(object):
    def __init__(self):
        """Initializes the Media class for handling media-related operations.

        Attributes:
            player (vlc.MediaPlayer or None): VLC player instance for media playback.
            instance (vlc.Instance or None): VLC instance for configuring playback.
            playback_thread (threading.Thread or None): Thread for non-blocking playback.
            stop_event (threading.Event): Event to signal playback termination.
            out_path (str): Directory path for saving output files (default: "../output").
            verbose (bool): Flag for enabling verbose output (default: False).

        Notes:
            - The output directory is not created until a method requiring it is called.
            - Playback-related attributes are initialized as None and set during playback.
        """
        self.player = None  # Store the VLC player instance
        self.instance = None  # Store the VLC instance
        self.playback_thread = None  # Store the playback thread
        self.stop_event = threading.Event()  # Event to signal stop
        self.out_path = "../../output"
        self.verbose = False

    def download(self, url=None):
        """Downloads media from a URL using yt-dlp and saves it to the output directory.

        Args:
            url (str, optional): The URL of the media to download (e.g., YouTube link).
                Defaults to None.

        Returns:
            str or None: The filepath of the downloaded media if successful, None otherwise.

        Raises:
            yt_dlp.DownloadError: If the download fails due to a yt-dlp issue.
            yt_dlp.ExtractorError: If URL extraction fails.
            Exception: For unexpected errors during download.

        Notes:
            - Creates the output directory if it doesn’t exist.
            - Downloads in MP4 format with the best available video and audio.
            - Verbose mode prints download time and filepath.
        """
        if not url or not isinstance(url, str):
            print("A valid URL string is required to download media.")
            return None
        output_path =  self.out_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        ydl_opts = {
            'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'merge_output_format': 'mp4',
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
            'verbose': False,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                if self.verbose:
                    start_time = time()
                    print("Downloading media...", end=" ")
                info = ydl.extract_info(url, download=False)
                file_path = ydl.prepare_filename(info)
                ydl.download(url)
                if self.verbose:
                    download_time = time() - start_time
                    print(f"\rDownload Completed in {download_time:.2f} seconds! File saved as: {file_path}")
                return file_path
        except yt_dlp.DownloadError as e:
            print(f"Download Error: {e}")
            return None
        except yt_dlp.ExtractorError as e:
            print(f"Extractor Error: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

    def play(self, filename):
        """Plays a media file using VLC in a non-blocking thread.

        Args:
            filename (str): Path to the media file to play (e.g., "video.mp4").

        Returns:
            bool: True if playback starts successfully, False if the file is missing or another playback is active.

        Notes:
            - Playback runs in a separate thread to avoid blocking the main process.
            - Use `stop_playback()` to stop the media before starting a new one.
            - Verbose mode prints the absolute filepath being played.
        """
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            return False

        if self.playback_thread and self.playback_thread.is_alive():
            print("Another playback is already in progress. Stop it first with stop_playback().")
            return False

        if self.verbose:
            print(f"Attempting to play: {os.path.abspath(filename)}")

        self.stop_event.clear()  # Ensure stop event is reset
        self.playback_thread = threading.Thread(target=self._play_in_thread, args=(filename,))
        self.playback_thread.start()
        return True

    def _play_in_thread(self, filename, audio_only=False):
        """Helper method to handle media playback in a separate thread.

        Args:
            filename (str): Path to the media file to play.
            audio_only (bool, optional): If True, plays audio only (no video). Defaults to False.

        Notes:
            - Internal method; not intended for direct use.
            - Configures VLC with minimal verbosity and Direct3D11 output.
            - Cleans up resources (player and instance) after playback ends or is stopped.
            - Verbose mode provides detailed playback status updates.
        """
        try:
            # Configure VLC instance
            vlc_args = ['--verbose=-1', '--vout=direct3d11', '--no-spu']
            if audio_only:
                vlc_args.append('--no-video')
            if self.verbose:
                print(f"Creating VLC instance with args: {vlc_args}")
            self.instance = vlc.Instance(*vlc_args)
            self.player = self.instance.media_player_new()
            media = self.instance.media_new(filename)
            self.player.set_media(media)
            if self.verbose:
                print(f"Starting playback in thread: {filename}")

            if self.player.play() == -1:
                print("VLC failed to start playback.")
                return

            # Keep playing until stopped or finished
            while not self.stop_event.is_set():
                state = self.player.get_state()
                if state in [vlc.State.Ended, vlc.State.Error]:
                    break
                sleep(1)
            if self.verbose:
                if self.player.get_state() == vlc.State.Ended:
                    print("Playback completed.")
                elif self.stop_event.is_set():
                    print("Playback interrupted by user.")
                else:
                    print(f"Playback stopped with state: {self.player.get_state()}")

        except Exception as e:
            print(f"Playback error in thread: {e}")
        finally:
            if self.player:
                self.player.stop()
                self.player.release()
            if self.instance:
                self.instance.release()
            self.player = None
            self.instance = None
            self.stop_event.clear()  # Reset stop event

    def stop_playback(self):
        """Stops the currently playing media.

        Returns:
            bool: True if playback was stopped successfully, False if no playback was active.

        Notes:
            - Waits for the playback thread to finish before returning.
            - Verbose mode confirms when playback is stopped.
        """
        if not self.playback_thread or not self.playback_thread.is_alive():
            print("No playback is currently active.")
            return False

        self.stop_event.set()  # Signal the thread to stop
        self.playback_thread.join()  # Wait for the thread to finish
        if self.verbose:
            print("Playback stopped.")
        return True

    def record_audio(self, filename, duration):
        """Records audio from the microphone and saves it as a WAV file.

        Args:
            filename (str): Name of the output audio file (saved in `out_path`).
            duration (int or float): Recording duration in seconds.

        Returns:
            bool: True if recording succeeds, False if duration is invalid or an error occurs.

        Notes:
            - Audio is recorded at 44.1 kHz, 16-bit, mono.
            - The full filepath is constructed as `out_path/filename`.
            - Prints recording status and filepath on completion.
        """
        if not isinstance(duration, (int, float)) or duration <= 0:
            print("Duration must be a positive number.")
            return False

        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100

        try:
            audio = pyaudio.PyAudio()
            stream = audio.open(format=FORMAT, channels=CHANNELS,
                                rate=RATE, input=True,
                                frames_per_buffer=CHUNK)
            print(f"Recording audio for {duration} seconds...")
            frames = []

            for _ in range(0, int(RATE / CHUNK * duration)):
                data = stream.read(CHUNK)
                frames.append(data)

            stream.stop_stream()
            stream.close()
            audio.terminate()
            file_path = self.out_path + "/" + filename
            with wave.open(file_path, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))

            print(f"Audio saved as: {file_path}")
            return True
        except Exception as e:
            print(f"Audio recording error: {e}")
            return False

    def record_video(self, filename, duration):
        """Records video from the default webcam and saves it as an AVI file.

        Args:
            filename (str): Name of the output video file (saved in `out_path`).
            duration (int or float): Recording duration in seconds.

        Returns:
            bool: True if recording succeeds, False if duration is invalid or webcam access fails.

        Notes:
            - Video is recorded at 20 FPS with a 640x480 resolution using the XVID codec.
            - The full filepath is constructed as `out_path/filename`.
            - Prints recording status and filepath on completion.
        """
        if not isinstance(duration, (int, float)) or duration <= 0:
            print("Duration must be a positive number.")
            return False

        try:
            cap = cv2.VideoCapture(0)  # Open default webcam
            if not cap.isOpened():
                print("Cannot access webcam.")
                return False

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            file_path = self.out_path + "/" + filename
            out = cv2.VideoWriter(file_path, fourcc, 20.0, (640, 480))

            print(f"Recording video for {duration} seconds...")
            start_time = time()
            while time() - start_time < duration:
                ret, frame = cap.read()
                if ret:
                    out.write(frame)
                else:
                    break

            cap.release()
            out.release()
            cv2.destroyAllWindows()
            print(f"Video saved as: {file_path}")
            return True
        except Exception as e:
            print(f"Video recording error: {e}")
            return False

    def screenshot(self, filename="screenshot.png"):
        """Captures a screenshot of the screen and saves it as a PNG file.

        Args:
            filename (str, optional): Name of the output image file (saved in `out_path`).
                Defaults to "screenshot.png".

        Returns:
            bool: True if the screenshot is saved successfully, False if an error occurs.

        Notes:
            - The full filepath is constructed as `out_path/filename`.
            - Verbose mode prints the saved filepath.
        """
        try:
            file_path = self.out_path + "/" + filename
            screenshot = pyautogui.screenshot()
            screenshot.save(file_path)
            if self.verbose:
                print(f"Screenshot saved as: {file_path}")
            return True
        except Exception as e:
            print(f"Screenshot error: {e}")
            return False


if __name__ == "__main__":
    media = Media()
    # Test download
    downloaded_file = media.download(url="https://www.youtube.com/watch?v=ZYRfUoR9Q4Y&ab_channel=RobinFrancis")
    # Test play (assuming download worked)
    media.play(downloaded_file)
    # Simulate doing other things while playback runs
    print("Playback started. Waiting 5 seconds before stopping...")
    sleep(5)
    media.stop_playback()
    print("Script continues after stopping playback.")
    # Test screenshot
    media.screenshot()
