"""A Streamlit web application integrating LLM and Media functionalities.

This module creates an interactive web interface for interacting with an AI assistant (via `llm.LLM`) and managing
media files (via `media.Media`). Users can chat with the AI, download media from URLs, play media files, record
audio or video, and capture screenshots. The application uses Streamlit for a simple, reactive UI.

Key Features:
    - Chat interface with the AI assistant with conversation history.
    - Media download from URLs with file display.
    - Media playback control (start/stop).
    - Audio and video recording with configurable durations.
    - Screenshot capture with file display.

Dependencies:
    - streamlit: For creating the web application and UI components.
    - llm: Custom module providing the `LLM` class for AI interactions.
    - media: Custom module providing the `Media` class for media operations.
    - os: For file and directory management.

Usage:
    Run the script directly (`streamlit run app_streamlit.py`) to start the server, then access it at
    `http://localhost:8501/`.
"""

import streamlit as st
import lib.llm as llm  # Importing from llm.py
import lib.media as media  # Importing from media.py
import os
import time

# Initialize LLM and Media instances
if 'llm' not in st.session_state:
    st.session_state['llm'] = llm
    st.session_state['llm'].verbose = False
if 'media' not in st.session_state:
    st.session_state['media'] = media
    st.session_state['media'].verbose = False

# Set output directory
UPLOAD_FOLDER = st.session_state['media'].out_path
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Main app layout
st.title("AI & Media Web App")

# Chat Section
st.header("Chat with AI")
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

chat_input = st.text_area("Type your message...", key="chat_input")
col1, col2 = st.columns(2)
speak = False
with col1:
    # Chat with AI
    if st.button("Send", key="chat_send"):
        if chat_input:
            speak = not speak
            response = st.session_state['llm'].chat(chat_input, speak=False)
            #st.session_state['chat_history'].append(("You", chat_input))
            #st.session_state['chat_history'].append(("AI", response))
            st.session_state['chat_history'] = [("Speak", speak), ("You", chat_input), ("AI", response)]
            st.session_state['last_response'] = response  # Store last response for speaking
with col2:
    # Speak last response
    if st.button("Speak", key="chat_speak"):
        st.session_state['speak'] = not st.session_state['speak']

st.text_area("Chat History", value="\n".join([f"{role}: {msg}" for role, msg in st.session_state['chat_history']]), height=200)

# Download Media Section
st.header("Download Media")
media_url = st.text_input("Enter URL (e.g., YouTube link)", key="media_url")
if st.button("Download", key="download_btn"):
    if media_url:
        with st.spinner("Downloading..."):
            file_path = st.session_state['media'].download(media_url)
        if file_path:
            st.success(f"Downloaded: {file_path}")
            with open(file_path, "rb") as file:
                st.download_button("Download File", data=file, file_name=os.path.basename(file_path))
        else:
            st.error("Download failed.")

# Play Media Section
st.header("Play Media")
play_file = st.text_input("Enter file path", key="play_file")
col3, col4 = st.columns(2)
with col3:
    if st.button("Play", key="play_btn"):
        if play_file and os.path.exists(play_file):
            success = st.session_state['media'].play(play_file)
            if success:
                st.session_state['playing'] = True
                st.success("Playback started.")
            else:
                st.error("Playback failed.")
        else:
            st.error("Invalid or missing file path.")
with col4:
    if st.button("Stop", key="stop_btn"):
        success = st.session_state['media'].stop_playback()
        if success:
            st.session_state['playing'] = False
            st.success("Playback stopped.")
        else:
            st.error("No active playback.")

# Record Audio/Video Section
st.header("Record Audio/Video")
record_duration = st.number_input("Duration (seconds)", min_value=1, value=5, key="record_duration")
col5, col6 = st.columns(2)
with col5:
    if st.button("Record Audio", key="record_audio_btn"):
        filename = f"audio_{int(time.time())}.wav"
        with st.spinner("Recording audio..."):
            success = st.session_state['media'].record_audio(filename, record_duration)
        if success:
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            st.success(f"Recorded: {file_path}")
            with open(file_path, "rb") as file:
                st.download_button("Download Audio", data=file, file_name=filename)
        else:
            st.error("Audio recording failed.")
with col6:
    if st.button("Record Video", key="record_video_btn"):
        filename = f"video_{int(time.time())}.avi"
        with st.spinner("Recording video..."):
            success = st.session_state['media'].record_video(filename, record_duration)
        if success:
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            st.success(f"Recorded: {file_path}")
            with open(file_path, "rb") as file:
                st.download_button("Download Video", data=file, file_name=filename)
        else:
            st.error("Video recording failed.")

# Screenshot Section
st.header("Screenshot")
if st.button("Take Screenshot", key="screenshot_btn"):
    filename = f"screenshot_{int(time.time())}.png"
    with st.spinner("Capturing screenshot..."):
        success = st.session_state['media'].screenshot(filename)
    if success:
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        st.success(f"Captured: {file_path}")
        with open(file_path, "rb") as file:
            st.download_button("Download Screenshot", data=file, file_name=filename)
    else:
        st.error("Screenshot failed.")

if __name__ == "__main__":
    # Streamlit apps are run with `streamlit run app_streamlit.py`, so this block is typically unused
    st.write("Run this app with `streamlit run app_streamlit.py`")