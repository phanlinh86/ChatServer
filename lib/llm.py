"""A module implementing a Large Language Model (LLM) with chat, translation, transcription, and text-to-speech capabilities.

This module provides the `LLM` class, which integrates an AI assistant powered by a language model (via Ollama),
audio transcription using Whisper-based methods, and text-to-speech functionality using pyttsx3. It is designed
for interactive conversations, multilingual translation, and audio processing tasks.

Key Features:
    - Chat with an AI assistant with streaming responses and optional speech output.
    - Translate text into supported languages.

    - Transcribe audio files using various Whisper implementations.
    - Convert text to speech with configurable voices.

Dependencies:
    - openai: For interacting with the Ollama API.
    - pyttsx3: For text-to-speech functionality.
    - whisper, faster_whisper, whisperx: For audio transcription.
    - torch: For GPU acceleration and model handling.
    - os: For environment configuration.

Constants:
    OLLAMA_BASE_URL (str): Default base URL for the Ollama API ("http://127.0.0.1:11434/v1").
    PROMPT_ASSISTANT (str): Default system prompt for the AI assistant.
    LANG_DICT (dict): Mapping of language codes to full names.
    SPEAK_MIN_LENGTH (int): Minimum text length to trigger speech (10 characters).
    BASE_MODEL (str): Default language model ("llama3.2:latest").

Usage:
    ```python
    llm = LLM(verbose=True)
    response = llm.chat("Hello, how are you?")
    translation = llm.translate("Good morning", "es")
    transcription = llm.transcribe("audio.wav")
    llm.speak("Hello, world!")
"""
import os
import sys
from time import time
from openai import OpenAI
from faster_whisper import WhisperModel
import pyttsx3          # For text to speech
#import whisper          # For transcribe
#import whisperx


OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"  # Default Ollama address

PROMPT_ASSISTANT = ("The following is a conversation with an AI assistant."
                    "\nThe assistant is helpful, creative, clever, and very friendly."
                    "\n"
                    "\nUser: Hello, who are you?"
                    "\nAI: I am an AI assistant. How can I help you today?"
                    "\nUser: What can you do for me?"
                    "\nAI: I can help you with a variety of tasks. What do you need help with?"
                    "\nUser:")

LANG_DICT = {   "en": "english",
                "vi": "vietnamese",
                "es": "spanish",
                "fr": "french",
                "ko": "korean",
                "zh": "chinese",
                "ja": "japanese",
                "de": "german",
            }  # Language codes
SPEAK_MIN_LENGTH = 10  # Minimum length of text to be spoken
BASE_MODEL = "llama3.2:latest"


class LLM(object):
    def __init__(self, prompt=PROMPT_ASSISTANT, verbose=False):
        """Initializes the LLM class with a chat client, text-to-speech engine, and optimized settings.

        Args:
            prompt (str, optional): Initial system prompt for the AI assistant. Defines its behavior and tone.
                Defaults to `PROMPT_ASSISTANT`.
            verbose (bool, optional): If True, enables verbose output (e.g., transcription timing).
                Defaults to False.

        Attributes:
            _client (OpenAI): Client instance for interacting with the Ollama API.
            _messages (list): List of conversation messages, starting with the system prompt.
            verbose (bool): Flag for verbose output.
            engine (pyttsx3.Engine): Text-to-speech engine instance with configured rate and volume.

        Raises:
            Exception: If the Ollama API connection or text-to-speech initialization fails, an error may occur
                during instantiation (handled by the respective libraries).

        Notes:
            - The Ollama API is accessed at `OLLAMA_BASE_URL`.
            - Text-to-speech is initialized with a speech rate of 150 and maximum volume (1.0).
            - Environment variables `OMP_NUM_THREADS` and `MKL_NUM_THREADS` are set to 12 for optimized
              Whisper processing.
        """
        # Initialize the chat client
        self._client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        self._messages = [
            {"role": "system", "content": prompt},
        ]
        self.verbose = verbose
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()  # Initialize text-to-speech engine
        self.engine.setProperty('rate', 150)  # Set speech rate
        self.engine.setProperty('volume', 1)  # Set volume (0.0 to 1.0)
        # Optimize for faster whisper processing
        os.environ["OMP_NUM_THREADS"] = "12"  # At script start
        os.environ["MKL_NUM_THREADS"] = "12"  # For PyTorch/Whisper


    def chat(self,message, speak=False, speak_stream=True):
        """Engages in a conversation with the AI assistant and optionally speaks the response.

        Args:
            message (str): The user's input message to send to the AI assistant.
            speak (bool, optional): If True, the AI response is spoken aloud. Defaults to False.
            speak_stream (bool, optional): If True, speaks the response incrementally as it streams.
                If False, speaks the full response at once. Ignored if `speak` is False. Defaults to True.

        Returns:
            str: The AI's response to the user's message.

        Raises:
            Exception: If text-to-speech fails, an error message is printed, but execution continues.

        Notes:
            - The conversation history is maintained in `self._messages`.
            - Streaming is used for real-time response generation.
        """
        self._messages.append({"role": "user", "content": message})
        response = self._client.chat.completions.create(model=BASE_MODEL, stream=True, messages=self._messages)
        ai_reply = ""
        ai_speak = ""
        for chunk in response:
            print(chunk.choices[0].delta.content or "", end="", flush=True)
            ai_speak += chunk.choices[0].delta.content or ""
            ai_reply += chunk.choices[0].delta.content or ""
            if speak_stream:
                if speak and ((ai_speak != "") and ai_speak[-1] in ".,!?"):
                    try:
                        # Get available voices
                        self.speak(ai_speak or "")
                    except Exception as e:
                        print(f"Text-to-speech error: {e}")
                    ai_speak = ""
        print()
        if speak and speak_stream is False:
            try:
                # Get available voices
                self.engine.say(ai_reply or "")
                self.engine.runAndWait()
            except Exception as e:
                print(f"Text-to-speech error: {e}")

        self._messages.append({"role": "bot", "content": ai_reply})
        return ai_reply

    def translate(self, text, target_language="vi"):
        """Translates the provided text into the specified target language.

        Args:
            text (str): The text to be translated.
            target_language (str, optional): The language code for the target language (e.g., "vi" for Vietnamese).
                Must be a key in `LANG_DICT`. Defaults to "vi".

        Returns:
            str: The translated text in the target language.

        Notes:
            - Supported languages are defined in `LANG_DICT`.
            - Translation is performed by invoking the `chat` method with a translation prompt.
        """
        message = f"Translate the following to {LANG_DICT[target_language]}:\n\n{text}"
        return self.chat(message)

    def transcribe(self, audio_file=None,
                   language="en",
                   method="fast_whisper",
                   model="tiny",
                   compute_type="int8"):  # Add language parameter
        """Transcribes audio from a file into text using the specified method and model.

        Args:
            audio_file (str): Path to the audio file to transcribe (e.g., "audio.wav").
            language (str, optional): Language code of the audio (e.g., "en" for English).
                Must be a key in `LANG_DICT`. Defaults to "en".
            method (str, optional): Transcription method to use. Options are:
                - "whisper": Standard Whisper model.
                - "fast_whisper": Faster Whisper implementation.
                - "whisperx": WhisperX with enhanced features.
                Defaults to "fast_whisper".
            model (str, optional): Model size for transcription. Options are "tiny", "base", or "large".
                Defaults to "tiny".
            compute_type (str, optional): Compute precision for "fast_whisper" and "whisperx".
                Options are "float32" or "int8". Defaults to "int8".

        Returns:
            dict or None: A dictionary containing the transcription result with at least a "text" key.
                Returns None if transcription fails.

        Raises:
            Exception: If transcription fails, an error message is printed, and None is returned.

        Notes:
            - Verbose mode prints transcription time if enabled.
            - The result format may vary slightly depending on the method used.
        """
        if self.verbose:
            print("Transcribing audio... ", end="")
            start_time = time()

        try:
            if method == "whisper":
                if 'whisper' not in sys.modules:
                    whisper = __import__('whisper')
                whisper_model = whisper.load_model(model)
                result = whisper_model.transcribe(audio_file, fp16=False, language=language)  # Pass language to Whisper
            elif method == "fast_whisper":
                whisper_model = WhisperModel(model, compute_type=compute_type)
                segments, info = whisper_model.transcribe(audio_file, language=language, beam_size=1)
                result = {'text': " ".join([segment.text for segment in segments]), 'info': info}
            elif method == "whisperx":
                if 'whisperx' not in sys.modules:
                    whisperx = __import__('whisperx')
                # Load WhisperX model
                # device = "cuda" if torch.cuda.is_available() else "cpu"
                whisperx_model = whisperx.load_model(model, compute_type=compute_type, language=language)
                audio = whisperx.load_audio(audio_file)
                result = whisperx_model.transcribe(audio, language=language, task="transcribe")     # Do the transcription only
                # Combine segments into a single text output
                result['text'] = " ".join([segment["text"] for segment in result["segments"]])

            if self.verbose:
                transcription_time = time() - start_time
                print(f"Transcription Completed in {transcription_time:.2f} seconds!")

            return result

        except Exception as e:
            print(f"Transcription Error: {e}")
            return None

    def speak(self, text, language=None):
        """Converts the given text to speech using the text-to-speech engine.

        Args:
            text (str): The text to be spoken.
            language (str, optional): Language code to set the voice (e.g., "en" for English).
                If None, the current voice is used. Must be a key in `LANG_DICT`. Defaults to None.

        Returns:
            None

        Raises:
            Exception: If text-to-speech fails, an error message is printed, but execution continues.

        Notes:
            - The voice is changed only if a valid `language` is provided.
        """
        try:
            if language:
                self.change_voice(language)
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Text-to-speech error: {e}")
            return None

    def change_voice(self, language="English"):
        """Changes the text-to-speech voice based on the specified language.

        Args:
            language (str, optional): The language name or code to match a voice (e.g., "English" or "en").
                Case-insensitive. Defaults to "English".

        Returns:
            bool: True if the voice was successfully changed.

        Raises:
            RuntimeError: If no voice matching the specified language is found.

        Notes:
            - Searches available voices for a match based on the language string.
            - The voice change persists until changed again or the engine is reinitialized.

        """
        for voice in self.engine.getProperty('voices'):
            if language.lower() in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                return True
        raise RuntimeError("Language '{}' not found".format(language))