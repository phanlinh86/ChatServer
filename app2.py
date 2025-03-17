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
            st.session_state['chat_history'].append(("You", chat_input))
            st.session_state['chat_history'].append(("AI", response))
            #st.session_state['chat_history'] = [("Speak", speak), ("You", chat_input), ("AI", response)]
            st.session_state['last_response'] = response  # Store last response for speaking
with col2:
    # Speak last response
    if st.button("Speak", key="chat_speak"):
        st.session_state['speak'] = not st.session_state['speak']

st.text_area("Chat History", value="\n".join([f"{role}: {msg}" for role, msg in st.session_state['chat_history']]), height=200)


if __name__ == "__main__":
    # Streamlit apps are run with `streamlit run app_streamlit.py`, so this block is typically unused
    st.write("Run this app with `streamlit run app_streamlit.py`")