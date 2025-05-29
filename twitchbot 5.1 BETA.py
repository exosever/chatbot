import logging
import asyncio
import random
import json
import os
import time
import gc
import openai # Added for OpenAI integration

from twitchio.ext import commands
# Removed google.generativeai and HarmCategory/HarmBlockThreshold imports
from dotenv import load_dotenv

"""
--------------------------------------------------------------------------------
CORE FUNCTIONALITY BELOW - ADVANCED USERS

The code below constitutes the core functionality of the bot.
Regular users should not modify this section.
Only make changes if you have a thorough understanding of the APIs and intend to
alter the bot's fundamental behavior.

--------------------------------------------------------------------------------
"""

"""
These are the environmental variables for the API keys and other bot settings.
All of these variables should be stored in your .env file
"""

if not load_dotenv('chatbot_variables.env'):
    with open('chatbot_variables.env', 'w') as file:
        file.write('# BOT CONFIGURATION - STANDARD USERS\n'
                   '# Below are the main configuration settings for the bot.\n'
                   '# Please adjust these variables to match your preferences and setup requirements.\n'
                   '# Ensure that you review and modify the values '
                   'according to your needs before running the bot.\n\n'

                   '# Logging levels, default is INFO, change to DEBUG if you need additional feedback\n'
                   'LOGGING = True\n'
                   'LOGGING_LEVEL = INFO\n\n'

                   'TWITCH_OAUTH_TOKEN = "13456"\n' # Placeholder
                   'TWITCH_CLIENT_ID = "123465"\n' # Placeholder
                   'TWITCH_CHANNEL_NAME = "123456"\n' # Placeholder
                   # Removed GENAI_API_KEY = "13456"
                   'GOOGLE_APPLICATION_CREDENTIALS = "123456"\n\n' # For Google TTS

                   '# AUTHORIZED_USERS_LIST is a list of USER NAMES\n'
                   '# to run admin commands from within twitch chat\n'
                   'AUTHORIZED_USERS_LIST=["TheJoshinatah", "DirtyDan"]\n\n'

                   # Removed FILTER_THRESHOLD = HIGH (OpenAI handles moderation)

                   '# BOT_TWITCH_NAME is the name of the "BOT" twitch account.\n'
                   '# BOT_NICKNAME is the name the bot will respond to.\n'
                   "BOT_TWITCH_NAME='ultron9000'\n"
                   "BOT_NICKNAME='ultron'\n\n"

                   '# BOT_ONLINE_MESSAGE is the message that the bot will send when it comes online.\n'
                   "BOT_ONLINE_MESSAGE='Hello everyone! How are you all doing?'\n\n"

                   '# ADJUSTMENT_WEIGHT defines the amount of change required for the bot\n'
                   '# to make a single adjustment to its emotional state.\n'
                   "ADJUSTMENT_WEIGHT=3\n\n"

                   '# FEEDBACK_TIME_THRESHOLD adjusts how long the bot '
                   'will wait before sending its feedback message.\n'
                   "FEEDBACK_TIME_THRESHOLD=120\n\n"

                   '# AUTOMATED_RESPONSE_TIME_RANGE adjusts how long the bot will\n'
                   '# wait before sending its automated response.\n'
                   '# It takes a RANGE in seconds (min, max)\n'
                   '# This time also affects how often the bot has the chance to become bored, curious,\n'
                   '# or the low chance of a random emotion.\n'
                   "AUTOMATED_RESPONSE_TIME_RANGE=[600, 1200]\n\n"

                   '# AUTOMATED_MESSAGE is the message the bot will send after the provided TIME_RANGE\n'
                   "AUTOMATED_MESSAGE=("
                   "f\"Hey There! I'm {BOT_NICKNAME}, your friendly neighborhood racoon! "
                   "Feel free to chat with me by calling my name first ^.^ ie:"
                   " {BOT_NICKNAME}, why is Josh such a great name?\n\n\")"

                   '# TTS CONFIGURATION\n'
                   '# Using https://cloud.google.com/text-to-speech?hl=en to find your settings\n'
                   '# OUTPUT_TTS_DEVICE_INDEX adjusts which output the TTS will use\n'
                   "TTS_MODEL='en-US-Wavenet-I'\n"
                   "TTS_LANGUAGE='en-US'\n"
                   "TTS_PITCH=0.0\n"
                   "TTS_SPEAKING_RATE=1.0\n"
                   "OUTPUT_TTS_DEVICE_INDEX=1\n\n"


                   '# STT CONFIGURATION\n'
                   '# Owner is your name, so the SST function knows who is talking to it.\n'
                   '# This is necessary if you wrote yourself into the chatbot_instructions by name.\n'
                   '# STT_INITIAL_THRESHOLD adjusts how loud the audio '
                   'needs to be to trigger the STT function.\n'
                   '# STT_SILENCE_DURATION adjusts how long the bot will '
                   'wait for silence before it sends the audio to STT.\n'
                   '# INPUT_STT_DEVICE_INDEX adjusts which microphone the bot will use. '
                   'Set this to None if you want it to print out a list of input devices.\n'
                   '# STT_NOISE_BUFFER_SIZE adjusts how many samples of "white noise" '
                   'the bot will take before it dynamically adjusts its threshold.\n'
                   '# MUTE_KEY is a key combination to mute audio input '
                   'to the STT system while the Flag is set to True.\n'
                   "OWNER='Josh'\n"
                   "STT_INITIAL_THRESHOLD=600\n"
                   "STT_SILENCE_DURATION=1.5\n"
                   "STT_NOISE_BUFFER_SIZE=30\n"
                   "INPUT_STT_DEVICE_INDEX=1\n"
                   "MUTE_KEY='ctrl + m'\n\n"

                   '# FEATURE FLAGS - STANDARD USERS\n'
                   '# The following flags are used to enable or disable certain features of the bot.\n'
                   '# Use this to tailor the bot to your needs, to free up system resources,\n'
                   '# or to minimize network usage.\n'
                   '# These flags can also assist with DEBUGGING problems, along with\n'
                   '# setting the LOGGING level to DEBUG.\n'
                   # "AI_WIKIPEDIA_FEATURE=True\n" # Removed
                   # "AI_EMOTION_DETECTION_FEATURE=True\n" # Removed
                   # "AI_MOODS_FEATURE=True\n" # Removed
                   # "AI_MEMORY_FEATURE=True\n" # Already Removed
                   "AI_LEARNING_FEATURE=True\n" # Retained
                   "AI_TTS_FEATURE=True\n" # Retained
                   "AI_STT_FEATURE=True\n\n" # Retained

                   '# OpenAI Configuration - Add these to your actual chatbot_variables.env file\n'
                   'OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"\n'
                   'OPENAI_ASSISTANT_ID="YOUR_OPENAI_ASSISTANT_ID_HERE"\n'
                   'OPENAI_VECTOR_STORE_ID="YOUR_OPENAI_VECTOR_STORE_ID_HERE"\n'
                   'LONG_TERM_MEMORY_UPLOAD_INTERVAL_SECONDS=3600\n'
                   )

    print("No .env detected or file is empty. "
          "chatbot_variables.env created with default values. "
          "Please add your API keys to this file and run again.")
    input("Press ENTER to exit")
    exit()


TWITCH_OAUTH_TOKEN = os.getenv('TWITCH_OAUTH_TOKEN')
TWITCH_CLIENT_ID = os.getenv('TWITCH_CLIENT_ID')
TWITCH_CHANNEL_NAME = os.getenv('TWITCH_CHANNEL_NAME')
# Removed GENAI_API_KEY = os.getenv('GENAI_API_KEY')
AUTHORIZED_USERS_LIST = json.loads(os.getenv('AUTHORIZED_USERS_LIST', '[]'))
BOT_TWITCH_NAME = os.getenv('BOT_TWITCH_NAME')
BOT_NICKNAME = os.getenv('BOT_NICKNAME')
BOT_ONLINE_MESSAGE = os.getenv('BOT_ONLINE_MESSAGE')
ADJUSTMENT_WEIGHT = int(os.getenv('ADJUSTMENT_WEIGHT'))
FEEDBACK_TIME_THRESHOLD = int(os.getenv('FEEDBACK_TIME_THRESHOLD'))
AUTOMATED_RESPONSE_TIME_RANGE = tuple(json.loads(os.getenv('AUTOMATED_RESPONSE_TIME_RANGE', '[]')))
AUTOMATED_MESSAGE = os.getenv('AUTOMATED_MESSAGE')
TTS_MODEL = os.getenv('TTS_MODEL')
TTS_LANGUAGE = os.getenv('TTS_LANGUAGE')
TTS_PITCH = float(os.getenv('TTS_PITCH'))
TTS_SPEAKING_RATE = float(os.getenv('TTS_SPEAKING_RATE'))
OWNER = os.getenv('OWNER')
STT_INITIAL_THRESHOLD = int(os.getenv('STT_INITIAL_THRESHOLD'))
STT_SILENCE_DURATION = float(os.getenv('STT_SILENCE_DURATION'))
INPUT_STT_DEVICE_INDEX = int(os.getenv('INPUT_STT_DEVICE_INDEX'))
OUTPUT_TTS_DEVICE_INDEX = int(os.getenv('OUTPUT_TTS_DEVICE_INDEX'))
STT_NOISE_BUFFER_SIZE = int(os.getenv('STT_NOISE_BUFFER_SIZE'))
# AI_WIKIPEDIA_FEATURE = os.getenv('AI_WIKIPEDIA_FEATURE', 'false').lower() in ['true', '1', 't', 'y', 'yes'] # Removed
# AI_EMOTION_DETECTION_FEATURE = os.getenv('AI_EMOTION_DETECTION_FEATURE', 'false').lower() in ['true', '1', 't', 'y', 'yes'] # Removed
# AI_MOODS_FEATURE = os.getenv('AI_MOODS_FEATURE', 'false').lower() in ['true', '1', 't', 'y', 'yes'] # Removed
# AI_MEMORY_FEATURE = os.getenv('AI_MEMORY_FEATURE', 'false').lower() in ['true', '1', 't', 'y', 'yes'] # Already Removed
AI_LEARNING_FEATURE = os.getenv('AI_LEARNING_FEATURE', 'false').lower() in ['true', '1', 't', 'y', 'yes'] # Retained
AI_TTS_FEATURE = os.getenv('AI_TTS_FEATURE', 'false').lower() in ['true', '1', 't', 'y', 'yes'] # Retained
AI_STT_FEATURE = os.getenv('AI_STT_FEATURE', 'false').lower() in ['true', '1', 't', 'y', 'yes']
MUTE_KEY = os.getenv('MUTE_KEY')
LOGGING_LEVEL = os.getenv('LOGGING_LEVEL')
LOGGING = os.getenv('LOGGING', 'false').lower() in ['true', '1', 't', 'y', 'yes']

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_ASSISTANT_ID = os.getenv('OPENAI_ASSISTANT_ID')
OPENAI_VECTOR_STORE_ID = os.getenv('OPENAI_VECTOR_STORE_ID') # Added
LONG_TERM_MEMORY_UPLOAD_INTERVAL_SECONDS = int(os.getenv('LONG_TERM_MEMORY_UPLOAD_INTERVAL_SECONDS', 3600))


"""
Logging Configuration
"""

if LOGGING:
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    LOGGING_LEVEL = level_map[LOGGING_LEVEL]
    logging.basicConfig(level=LOGGING_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

    # Removed threshold_map and FILTER_THRESHOLD assignment

if AI_TTS_FEATURE:
    google_credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

    if google_credentials_path:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(
            os.getcwd(), google_credentials_path)
    else:
        print("No Gemini credentials found in chatbot_variables.env.")
        print("Please add your Gemini credentials to chatbot_variables.env and run again.")
        input("Press ENTER to exit")
        exit()

# Updated to remove GENAI_API_KEY from the check
if not all([TWITCH_OAUTH_TOKEN, TWITCH_CLIENT_ID, TWITCH_CHANNEL_NAME, OPENAI_API_KEY, OPENAI_ASSISTANT_ID]):
    print("Please verify all API keys (Twitch, OpenAI) are present in chatbot_variables.env and run again.")
    input("Press ENTER to exit")
    exit()

# Initialize OpenAI Client
if OPENAI_API_KEY:
    openaiclient = openai.OpenAI(api_key=OPENAI_API_KEY) # Renamed to openaiclient to avoid conflict
    logging.info("OpenAI client initialized successfully.")
else:
    logging.error("OPENAI_API_KEY not found. OpenAI features will be disabled.")
    # Decide if to exit or disable features - for now, just log an error


# Removed genai.configure(api_key=GENAI_API_KEY)

if AI_STT_FEATURE and not AI_TTS_FEATURE:
    AI_TTS_FEATURE = True

"""
load the generation config from a JSON file
If a config is not present, a default one will be created
"""

# Removed generation_config.json loading section

"""
Initialize the bot, emotion detection, and wikipedia API
"""
try:
    bot = commands.Bot(
        token=TWITCH_OAUTH_TOKEN,
        client_id=TWITCH_CLIENT_ID,
        nick=BOT_TWITCH_NAME,
        prefix='!',
        initial_channels=[TWITCH_CHANNEL_NAME]
    )
    logging.info("Bot instance created successfully.")
except Exception as e:
    logging.error("Failed to create bot instance, error:", f"{e}")
    print("An error occurred while creating the bot instance. Check the log for details.")

# Removed AI_EMOTION_DETECTION_FEATURE block including transformers import and emotion_classifier
# Removed AI_WIKIPEDIA_FEATURE block including nltk and wikipediaapi imports and wiki_wiki instance

"""
Process time function for performance debugging
"""


def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper


"""
This block handles the Google TTS API
Initializes model, language, pitch, speed, etc.
It also handles the audio buffer and queue
"""

if AI_TTS_FEATURE or AI_STT_FEATURE:
    import pyaudio
    import wave

if AI_TTS_FEATURE:
    import emoji
    from collections import deque
    from google.cloud import texttospeech
    from pydub import AudioSegment # Uncommented: Required for play_audio_from_buffer
    import io

    tts_queue = deque()
    is_playing = False

    # Note: The variable 'client' was previously used for Google TTS.
    # If Google TTS is still active, this variable name might need to be managed
    # to avoid conflict with the OpenAI client, which I named 'openaiclient'.
    # For now, assuming Google TTS client init is still valid.
    google_tts_client = texttospeech.TextToSpeechClient() # Corrected: Initialize only once

    logging.info("Google TTS API initialized successfully.")

    # OpenAI Client initialization is handled earlier in the script.

    @time_it
    def synthesize_speech(text, pitch=TTS_PITCH, speaking_rate=TTS_SPEAKING_RATE):
        input_text = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code=TTS_LANGUAGE,
            name=TTS_MODEL,
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            pitch=pitch,
            speaking_rate=speaking_rate,
        )

        response = google_tts_client.synthesize_speech( # Corrected: Use renamed google_tts_client
            input=input_text, voice=voice, audio_config=audio_config
        )

        audio_buffer = io.BytesIO(response.audio_content)
        logging.info("Audio content written to in-memory buffer")

        return audio_buffer

    @time_it
    async def handle_tts_request(text):
        audio_buffer = synthesize_speech(text)
        tts_queue.append(audio_buffer)

        if not is_playing:
            await play_next_in_queue()

    @time_it
    async def play_next_in_queue():
        global is_playing
        while tts_queue:
            audio_buffer = tts_queue.popleft()
            is_playing = True
            play_audio_from_buffer(audio_buffer)
            audio_buffer.close()

        is_playing = False

    @time_it
    def play_audio_from_buffer(audio_buffer):
        audio_buffer.seek(0)

        audio_segment = AudioSegment.from_mp3(audio_buffer)
        pcm_buffer = io.BytesIO()
        audio_segment.export(pcm_buffer, format="wav")
        pcm_buffer.seek(0)

        with wave.open(pcm_buffer, "rb") as wf:
            p = pyaudio.PyAudio()

            stream = p.open(format=pyaudio.paInt16,
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True,
                            output_device_index=OUTPUT_TTS_DEVICE_INDEX)

            data = wf.readframes(1024)
            while data:
                stream.write(data)
                data = wf.readframes(1024)

            stream.stop_stream()
            stream.close()
            p.terminate()


"""
These are the experimental STT Gemini query functions
"""

# Global dictionary to store user_id to thread_id mapping
user_threads = {}

async def append_to_long_term_memory(user_id: str, prompt: str, response: str):
    """Appends a user interaction to the long-term memory file in JSONL format."""
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "user_id": user_id,
        "prompt": prompt,
        "response": response
    }
    try:
        # Define the file writing operation as a synchronous function
        def _write_sync():
            # Ensure the directory exists (optional, if file is in root)
            # os.makedirs(os.path.dirname("long_term_memory.txt"), exist_ok=True)
            with open("long_term_memory.txt", "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        
        # Run the synchronous file writing operation in a separate thread
        await asyncio.to_thread(_write_sync)
        logging.info("Successfully appended interaction for user %s to long_term_memory.txt", user_id) # Changed to INFO
    except Exception as e:
        logging.error("Error appending to long_term_memory.txt for user %s: %s", user_id, e)

async def query_openai_assistant(user_id: str, prompt: str) -> str:
    global user_threads
    assistant_id = OPENAI_ASSISTANT_ID # Loaded from environment variables

    if not openaiclient:
        logging.error("OpenAI client not initialized. Cannot query assistant for user_id: %s", user_id)
        return "Sorry, I'm having trouble connecting to the AI service."

    try:
        # Check if user_id has an existing thread
        thread_id = user_threads.get(user_id)
        if not thread_id:
            logging.info("No existing thread found for user_id: %s. Creating new thread.", user_id)
            thread = await asyncio.to_thread(openaiclient.beta.threads.create)
            user_threads[user_id] = thread.id
            thread_id = thread.id
            logging.info("New thread created with ID: %s for user_id: %s", thread_id, user_id)
        else:
            logging.info("Using existing thread ID: %s for user_id: %s", thread_id, user_id)

        # Add message to thread
        logging.info("Adding message to thread %s for user_id %s. Prompt: '%s'", thread_id, user_id, prompt)
        await asyncio.to_thread(
            openaiclient.beta.threads.messages.create,
            thread_id=thread_id,
            role="user",
            content=prompt
        )

        # Create a run
        logging.info("Creating run for thread_id: %s with assistant_id: %s", thread_id, assistant_id)
        run = await asyncio.to_thread(
            openaiclient.beta.threads.runs.create,
            thread_id=thread_id,
            assistant_id=assistant_id
        )
        logging.info("Run created with ID: %s for thread_id: %s", run.id, thread_id)

        # Poll for response
        logging.info("Polling for run completion, run_id: %s, thread_id: %s", run.id, thread_id)
        start_time = time.time()
        timeout_seconds = 60 # Set a timeout for the run completion

        while True:
            if time.time() - start_time > timeout_seconds:
                logging.error("Run %s timed out after %d seconds for thread_id: %s.", run.id, timeout_seconds, thread_id)
                # Optionally, remove thread_id if run timed out
                # if user_id in user_threads: del user_threads[user_id]
                return "Sorry, the AI operation timed out. Please try again."

            run_status = await asyncio.to_thread(
                openaiclient.beta.threads.runs.retrieve,
                thread_id=thread_id,
                run_id=run.id
            )
            logging.debug("Run %s status: %s for thread_id: %s", run.id, run_status.status, thread_id)

            if run_status.status == 'completed':
                logging.info("Run %s completed for thread_id: %s", run.id, thread_id)
                break
            elif run_status.status in ['queued', 'in_progress']:
                await asyncio.sleep(1)  # Wait for 1 second before polling again
            elif run_status.status in ['failed', 'cancelled', 'expired']:
                logging.error("Run %s %s. Error: %s for thread_id: %s", run.id, run_status.status, run_status.last_error, thread_id)
                # Optionally, remove thread_id if run failed permanently
                # if user_id in user_threads: del user_threads[user_id]
                return f"Sorry, the AI operation {run_status.status}. Please try again."
            elif run_status.status == 'requires_action':
                logging.warning("Run %s requires action. This is not handled in this version for thread_id: %s.", run.id, thread_id)
                # This part would require handling function calls from the assistant if any are defined.
                # For now, we will attempt to submit empty tool outputs to see if it resolves.
                if run_status.required_action and run_status.required_action.type == "submit_tool_outputs":
                    logging.info("Run %s requires tool outputs. Submitting empty outputs for thread_id: %s.", run.id, thread_id)
                    await asyncio.to_thread(
                        openaiclient.beta.threads.runs.submit_tool_outputs,
                        thread_id=thread_id,
                        run_id=run.id,
                        tool_outputs=[] # Submit empty tool outputs
                    )
                else:
                    return "Sorry, the AI requires an action I can't perform yet."
            else:
                logging.error("Unknown run status for run %s: %s for thread_id: %s", run.id, run_status.status, thread_id)
                return "Sorry, an unexpected error occurred with the AI."

        # Retrieve messages
        logging.info("Retrieving messages for thread_id: %s", thread_id)
        messages = await asyncio.to_thread(
            openaiclient.beta.threads.messages.list,
            thread_id=thread_id
        )

        # Find the assistant's response
        assistant_response = "Sorry, I couldn't get a response from the assistant."
        # Messages are returned in descending order (newest first)
        for msg in messages.data:
            if msg.run_id == run.id and msg.role == "assistant":
                if msg.content and len(msg.content) > 0:
                    content_item = msg.content[0]
                    if hasattr(content_item, 'text') and hasattr(content_item.text, 'value'):
                        assistant_response = content_item.text.value
                        # Log a snippet of the response if it's too long
                        response_snippet = (assistant_response[:75] + '...') if len(assistant_response) > 75 else assistant_response
                        logging.info("Assistant response found in thread %s for run %s: '%s'", thread_id, run.id, response_snippet)
                        break 
        
        if AI_LEARNING_FEATURE: # Retain existing feedback mechanism
             add_feedback_user_id(user_id)
        
        # Call to append to new long-term memory
        await append_to_long_term_memory(user_id, prompt, assistant_response)

        return assistant_response

    except openai.APIError as e:
        logging.error(f"OpenAI API Error for user {user_id}, prompt '{prompt}': {e}")
        return f"Sorry, there was an API error: {e}"
    except Exception as e:
        logging.error(f"Unexpected error in query_openai_assistant for user {user_id}, prompt '{prompt}': {e}")
        # It's good practice to log the traceback for unexpected errors
        # import traceback
        # logging.error(traceback.format_exc())
        return "Sorry, an unexpected error occurred while talking to the AI."


# Removed query_gemini_with_STT function


"""
This is the experimental STT function.
"""

if AI_STT_FEATURE:
    import threading
    from google.cloud import speech
    import numpy as np
    import keyboard

    is_muted = False

    def toggle_mute():
        global is_muted
        is_muted = not is_muted
        print(f"Audio input is {'muted' if is_muted else 'active'}.")

    keyboard.add_hotkey(MUTE_KEY, toggle_mute)

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    p = pyaudio.PyAudio()

    frames = []
    recording = False
    dynamic_threshold = STT_INITIAL_THRESHOLD
    noise_buffer = []
    last_audio_time = time.time()

    @time_it
    def calculate_noise_level():
        if not noise_buffer:
            return STT_INITIAL_THRESHOLD
        return np.mean(noise_buffer)

    @time_it
    async def transcribe_audio(audio_buffer):
        try:
            client = speech.SpeechClient()
            audio_content = audio_buffer.getvalue()

            audio = speech.RecognitionAudio(content=audio_content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=RATE,
                language_code="en-US",
            )

            response = client.recognize(config=config, audio=audio)

            for result in response.results:
                logging.info("Transcript sent to OpenAI Assistant API\n" # MODIFIED
                             f"{result.alternatives[0].transcript}")
                response_text = await query_openai_assistant(OWNER, result.alternatives[0].transcript) # MODIFIED
                logging.info("Response from OpenAI Assistant API: " + response_text) # MODIFIED
                clean_response = emoji.replace_emoji(response_text, replace='')
                clean_response = clean_response.replace('"', ' ')
                clean_response = clean_response.replace('*', ' ')
                audio_file_buffer = synthesize_speech(clean_response)
                logging.info("Generated speech audio buffer.")

                tts_queue.append(audio_file_buffer)

                if not is_playing:
                    await play_next_in_queue()
        except Exception as e:
            logging.error(f"Error processing audio: {e}")

    @time_it
    def process_audio(frames, channels, rate):
        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))
        audio_buffer.seek(0)

        def run_transcribe():
            asyncio.run(transcribe_audio(audio_buffer))

        threading.Thread(target=run_transcribe).start()

    @time_it
    def callback(in_data, frame_count, time_info, status):
        global recording, frames, last_audio_time

        audio_data = np.frombuffer(in_data, dtype=np.int16)
        audio_level = np.max(np.abs(audio_data))

        noise_buffer.append(audio_level)
        if len(noise_buffer) > STT_NOISE_BUFFER_SIZE:
            noise_buffer.pop(0)

        dynamic_threshold = calculate_noise_level() * 1.5

        current_time = time.time()

        if not is_muted:
            if audio_level > dynamic_threshold:
                last_audio_time = current_time
                if not recording:
                    recording = True
                    logging.debug("Input detected. Recording...")

            if recording:
                frames.append(in_data)

                if current_time - last_audio_time > STT_SILENCE_DURATION:
                    recording = False
                    logging.info("Silence detected. Finished recording.")
                    threading.Thread(target=process_audio, args=(frames, CHANNELS, RATE)).start()
                    frames = []

        return (in_data, pyaudio.paContinue)

    def start_stt():
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK,
                        input_device_index=INPUT_STT_DEVICE_INDEX,
                        stream_callback=callback)
        logging.info("Speech to text API started.")

        try:
            stream.start_stream()
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            logging.info("Speech to text API stopped.")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    stt_thread = threading.Thread(target=start_stt)
    stt_thread.start()


"""
This code block establishes the parameters for the bots emotional states
States are the emotional value, which corrospond to a description of the mood
These values can be adjusted by a slider,
by a random number, or chosen specifically
"""

# Removed AI_MOODS_FEATURE block, including:
# emotional_states list, emotional_states.txt loading, current_emotion_index,
# get_emotional_state function, mood_instructions, adjustment_counter,
# adjust_emotional_state function, and adjust_emotional_state_analysis function.

"""
This block handles the reinforcement learning
It takes user feedback values
And adjusts the parameters of the model
"""

feedback_memory = []


def update_parameters_based_on_feedback():
    global feedback_memory

    # This function previously handled GenAI-specific parameter tuning
    # (temperature, top_k, top_p) and mood adjustments.
    # Since GenAI components and related features (moods) have been removed,
    # this function now only serves to clear the feedback_memory.
    # If OpenAI requires similar feedback-based tuning, it would need a new implementation.

    logging.info("Processed feedback and cleared feedback_memory.")
    feedback_memory = []


"""
This function saves the feedback from the user
Simple positive or negative feedback will be used to adjust
the parameters of the model
"""


feedback_list_max_size = 5
feedback_list = []


def add_feedback_user_id(user_id):
    if len(feedback_list) >= feedback_list_max_size:
        feedback_list.pop(0)
    feedback_list.append(user_id)


def can_give_feedback(user_id):
    if user_id in feedback_list:
        feedback_list.remove(user_id)
        return True
    return False


"""
Load the instructions for the bot personality if they exist
"""

# Removed chatbot_instructions.txt loading, as OpenAI Assistant instructions are set in its configuration.


"""
Model settings and paramters
"""
# Removed Gemini model initialization (genai.GenerativeModel)

"""
Load and save the persistent memory
Additionally, cache the memory for faster loading
"""

# Removed the entire AI_MEMORY_FEATURE block, including:
# - import sqlite3
# - conn, cursor initialization
# - user_memory table creation
# - save_memory function
# - load_memory function
# - user_memory_cache dictionary
# - load_cached_memory function
# - save_cached_memory function

"""
Download keyword files if necessary
"""

# Removed download_nltk_data function
# Removed AI_WIKIPEDIA_FEATURE block that called download_nltk_data

"""
This function extracts keywords for the wikipedia and duckduckgo APIs
"""

# Removed extract_keywords function

"""
This code block searches wikipedia and duckduckgo APIs for relevant information
Based on keywords extracted from prompt
"""

# Removed fetch_information function

"""
This function formats the prompt to be sent to the API
It gathers historic user data if available
It then sends the generated response to the chat
And saves the user prompt and response to the memory
"""
message_count = 0


# Removed query_gemini_with_memory function

"""
--------------------------------------------------------------------------------
BOT CONFIGURATION - ADVANCED USERS
--- BOT EVENTS ---

These are the bot events
--------------------------------------------------------------------------------
"""


"""
This function checks if the message is from a user
If so, it formats the prompt to be sent to the API
It also sends the generated response to the chat
And only after FEEDBACK_TIME_THRESHOLD sends a message asking the user to use !feedback

This function also redeems a TTS Message channel point reward to send
The generated response through the Google TTS API to the chat
"""
last_feedback_message_time = 0


@ bot.event()
async def event_message(message):
    global message_count, last_feedback_message_time

    current_time = time.time()

    if bot.nick.lower() not in str(message.author).lower():
        message_count += 1

    if (AI_TTS_FEATURE and 'custom-reward-id=16051547-8f57-4832-acb5-56df48b6e761'
            in message.raw_data):

        user_id = str(message.author.id)
        prompt = message.content.strip()
        logging.debug(f"Processed prompt: {prompt}")

        try:
            response = await query_openai_assistant(user_id, prompt) # MODIFIED
            logging.info(f"Generated response from OpenAI Assistant: {response}") # MODIFIED
        except Exception as e:
            logging.error(f"Error processing message from OpenAI Assistant: {e}") # MODIFIED

        clean_response = emoji.replace_emoji(response, replace='')
        await message.channel.send(response)
        await handle_tts_request(clean_response)

        if (current_time - last_feedback_message_time >= FEEDBACK_TIME_THRESHOLD
                and AI_LEARNING_FEATURE):
            await message.channel.send(
                'Be sure to use !feedback <good/bad> '
                'to let me know if I did a good job!'
            )

            last_feedback_message_time = current_time

        logging.info(f"Sent response: {response}")

    elif (
        message.content.lower().startswith(BOT_NICKNAME.lower())
        or message.content.lower().startswith(f"@{BOT_TWITCH_NAME.lower()}")
    ):
        user_id = str(message.author.id)
        prompt = message.content.strip()

        logging.debug(f"Processed prompt: {prompt}")

        try:
            response = await query_openai_assistant(user_id, prompt) # MODIFIED
            logging.info(f"Generated response from OpenAI Assistant: {response}") # MODIFIED

            await message.channel.send(response)

            if (current_time - last_feedback_message_time >= FEEDBACK_TIME_THRESHOLD
                    and AI_LEARNING_FEATURE):
                await message.channel.send(
                    'Be sure to use !feedback <good/bad> '
                    'to let me know if I did a good job!'
                )

                last_feedback_message_time = current_time

            logging.info(f"Sent response: {response}")
        except Exception as e:
            logging.error(f"Error processing message from OpenAI Assistant: {e}") # MODIFIED
    else:
        logging.debug(f"Ignoring message: {message.content}")


"""
This function is used to send a message to the chat when the bot is ready.
Debug connectivity to Twitch
"""


@ bot.event()
async def event_ready():
    logging.info(f'Logged in as | {bot.nick}')
    logging.info(f'Connected to channel | {TWITCH_CHANNEL_NAME}')

    try:
        channel = bot.get_channel(TWITCH_CHANNEL_NAME)
        if channel:
            await channel.send(BOT_ONLINE_MESSAGE)
            logging.info('Sent online confirmation message to chat.')
        else:
            logging.error(f"Channel {TWITCH_CHANNEL_NAME} not found.")
    except Exception as e:
        logging.error(f"Error sending confirmation message: {e}")

    bot.loop.create_task(automated_response())
    bot.loop.create_task(upload_long_term_memory_periodically()) # Added
    logging.info("Started periodic long-term memory upload task.") # Added


"""
This is a loop that will send a message to chat
after at least 10 user messages have been received
and AUTOMATED_RESPONSE_TIME_RANGE minutes have passed

This block also updates the parameters of the model
based off the feedback received
"""

session_cleanup_time = time.time()


def cleanup_memory():
    global session_cleanup_time
    if time.time() - session_cleanup_time > 3600:
        gc.collect()
        session_cleanup_time = time.time()


async def automated_response():
    global message_count # current_emotion_index removed

    while True:
        wait_time = random.randint(*AUTOMATED_RESPONSE_TIME_RANGE)
        await asyncio.sleep(wait_time)
        cleanup_memory()
        # Removed AI_MOODS_FEATURE block related to random emotion changes
        if message_count >= 10:
            try:
                channel = bot.get_channel(TWITCH_CHANNEL_NAME)
                if channel:
                    await channel.send(AUTOMATED_MESSAGE)
                    logging.info(f"Sent automated message: {AUTOMATED_MESSAGE}")
                    message_count = 0
                else:
                    logging.error(f"Channel {TWITCH_CHANNEL_NAME} not found.")
            except Exception as e:
                logging.error(f"Error sending automated message: {e}")
        else:
            logging.debug(f"Not enough messages received yet: {message_count}")


"""
--------------------------------------------------------------------------------
BOT CONFIGURATION - ADVANCED USERS
--- BOT COMMANDS ---

These are the bot commands
Those with if.ctx.author.name in AUTHORIZED_USERS_LIST are authorized user only commands

--------------------------------------------------------------------------------
"""


# Command to describe the AI to the user
@ bot.command(name='AI')
async def ai(ctx):
    await ctx.send("I'm a bot created by @thejoshinatah! ^.^ "
                   "I make use of multiple APIs "
                   "and models to generate responses! "
                   "If you'd like to know more, check out our "
                   "github https://github.com/exosever/chatbot/"
                   )

# Command to submit user feedback to reinforcement learning
if AI_LEARNING_FEATURE:
    @ bot.command(name='feedback')
    async def feedback(ctx, feedback_type):
        global feedback_memory
        user_id = str(ctx.author.id)
        if can_give_feedback(user_id):
            if feedback_type.lower() == 'good':
                feedback_memory.append({'positive': 1})
                await ctx.send("Thank's for letting me know! ^.^")
            elif feedback_type.lower() == 'bad':
                feedback_memory.append({'negative': 1})
                await ctx.send("I'm sorry about that! Thanks for "
                               "helping me do better next time!")
            update_parameters_based_on_feedback()


# Removed !Wikipedia command

@ bot.command(name='TTS')
async def TTS_flag(ctx):
    global AI_TTS_FEATURE
    if ctx.author.name in AUTHORIZED_USERS_LIST:
        AI_TTS_FEATURE = not AI_TTS_FEATURE


@ bot.command(name='STT')
async def STT_flag(ctx):
    global AI_STT_FEATURE
    if ctx.author.name in AUTHORIZED_USERS_LIST:
        AI_STT_FEATURE = not AI_STT_FEATURE


# Removed !Memory command


# Removed !Moods command
# Removed !Detection command

@ bot.command(name='Learning')
async def learning_flag(ctx):
    global AI_LEARNING_FEATURE
    if ctx.author.name in AUTHORIZED_USERS_LIST:
        AI_LEARNING_FEATURE = not AI_LEARNING_FEATURE

try:
    bot.run()
except AttributeError:
    logging.error("Error running bot:\n"
                  "Please check your Twitch CLIENT ID and OAUTH Keys and try again."
                  )
input("Press ENTER to exit")

# Slow response
# Sometimes cutting the spoken prompt in half, and responding to each one independantly
# Not catching most of the spoken prompts
