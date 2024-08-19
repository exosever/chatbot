import logging
import asyncio
import random
import json
import os
import time
import gc

import google.generativeai as genai
from twitchio.ext import commands
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv


"""
Logging Configuration
Change from INFO to DEBUG for more detailed logging
"""
logging.basicConfig(level=logging.
                    INFO, format='%(asctime)s - %(levelname)s - %(message)s')

"""
--------------------------------------------------------------------------------
BOT CONFIGURATION - STANDARD USERS

Below are the main configuration settings for the bot.
Please adjust these variables to match your preferences and setup requirements.

Ensure that you review and modify the values according to your needs before running the bot.

--------------------------------------------------------------------------------
"""


"""
AUTHORIZED_USERS_LIST is a list of USER IDs you want to have access
to run admin commands from within twitch chat
I recommend using your own user ID, or someone you trust
"""

AUTHORIZED_USERS_LIST = ['123456789', '987654321']

"""
BOT_TWITCH_NAME is the name of the "BOT" twitch account.
BOT_NICKNAME is the name the bot will respond to.
"""
BOT_TWITCH_NAME = 'chesterbot9000'
BOT_NICKNAME = 'chester'


"""
FILTER_THRESHOLD adjusts the bot's response to harmful content.
BLOCK_ONLY_HIGH only blocks content with a HIGH risk.
BLOCK_ONLY_LOW only blocks content with a LOW risk.
"""
# BLOCK_ONLY_HIGH, BLOCK_ONLY_MEDIUM, BLOCK_ONLY_LOW, BLOCK_HIGH_AND_MEDIUM,
# BLOCK_HIGH_AND_MEDIUM_AND_LOW, BLOCK_HIGH_AND_MEDIUM_AND_LOW_AND_NONE
FILTER_THRESHOLD = HarmBlockThreshold.BLOCK_ONLY_HIGH


"""
BOT_ONLINE_MESSAGE is the message that the bot will send when it comes onine.
"""
BOT_ONLINE_MESSAGE = 'Hello everyone! How are you all doing?'


"""
ADJUSTMENT_WEIGHT defines the amount of change required for the bot
to make a single adjustment to its emotional state.
This is done by adding or subtracting the positive or negative
stimuli until the positive or negative WEIGHT is met.
Then a single emotional adjustment is made.
STIMULI are either positive or negative user feedback, or positive or
negative emotional analysis from user input.
"""
ADJUSTMENT_WEIGHT = 3


"""
FEEDBACK_TIME_THRESHOLD adjusts how long the bot will wait before sending it's feedback message.
"""
FEEDBACK_TIME_THRESHOLD = 120


"""
AUTOMATED_RESPONSE_TIME_RANGE adjusts how long the bot will
wait before sending it's automated response.
It takes a RANGE in seconds (min, max)
This time also affects how often the bot has the chance to become bored, curious,
or the low chance of a random emotion.
"""
AUTOMATED_RESPONSE_TIME_RANGE = (600, 1200)


"""
AUTOMATED_MESSAGE is the message the bot will send after the provided TIME_RANGE
"""
AUTOMATED_MESSAGE = (
    f"Hey There! I'm {BOT_NICKNAME}, "
    "your friendly neighborhood racoon! "
    "Feel free to chat with me "
    "by calling my name first ^.^ "
    f"ie: {BOT_NICKNAME}, why is Josh such a great name?"
)


"""
TTS CONFIGURATION
Using https://cloud.google.com/text-to-speech?hl=en to find your settings

"""

TTS_MODEL = "en-US-Wavenet-I"
TTS_LANGUAGE = "en-US"
TTS_PITCH = 6.0
TTS_SPEAKING_RATE = 1.0


"""
--------------------------------------------------------------------------------
FEATURE FLAGS - STANDARD USERS

The following flags are used to enable or disable certain features of the bot.
Use this to tailor the bot to your needs, to free up system resources,
or to minimize network usage.
These flags can also assist with DEBUGGING problems, along with
setting the LOGGING level to DEBUG.

--------------------------------------------------------------------------------
"""

AI_WIKIPEDIA_FEATURE = True  # Wikipedia API keyword search
AI_EMOTION_DETECTION_FEATURE = True  # AI analysis of user emotions
AI_MOODS_FEATURE = True  # AI moods based on interactions
AI_MEMORY_FEATURE = True  # Database storage of AI memory
AI_LEARNING_FEATURE = True  # AI learning from user feedback
AI_TTS_FEATURE = True  # TTS generation of AI responses
AI_STT_FEATURE = True  # SST API for speech recognition. TTS API is required.

if AI_STT_FEATURE and not AI_TTS_FEATURE:
    AI_TTS_FEATURE = True

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
These are the environmental variables for the API keys.
All of these variables should be stored in your .env file
"""

if not load_dotenv('chatbot_variables.env'):
    with open('chatbot_variables.env', 'w') as file:
        file.write('TWITCH_OAUTH_TOKEN=\n'
                   'TWITCH_CLIENT_ID=\n'
                   'GENAI_API_KEY=\n'
                   'TWITCH_CHANNEL_NAME=\n'
                   'GOOGLE_APPLICATION_CREDENTIALS=\n')
    print("No .env detected or file is empty. "
          "chatbot_variables.env created. "
          "Please add your API keys to this file and run again.")
    exit()


TWITCH_OAUTH_TOKEN = os.getenv('TWITCH_OAUTH_TOKEN')
TWITCH_CLIENT_ID = os.getenv('TWITCH_CLIENT_ID')
TWITCH_CHANNEL_NAME = os.getenv('TWITCH_CHANNEL_NAME')
GENAI_API_KEY = os.getenv('GENAI_API_KEY')

if AI_TTS_FEATURE:
    google_credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

    if google_credentials_path:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(
            os.getcwd(), google_credentials_path)
    else:
        print("No Google credentials found in chatbot_variables.env.")
        print("Please add your Google credentials to chatbot_variables.env and run again.")
        exit()

if not all([TWITCH_OAUTH_TOKEN, TWITCH_CLIENT_ID, TWITCH_CHANNEL_NAME, GENAI_API_KEY]):
    print("Please verify all API keys are present in chatbot_variables.env and run again.")
    exit()

genai.configure(api_key=GENAI_API_KEY)

"""
load the generation config from a JSON file
If a config is not present, a default one will be created
"""


try:
    with open("generation_config.json", "r") as config_file:
        generation_config = json.load(config_file)
        logging.info("generation_config.json loaded successfully.")
except FileNotFoundError:
    generation_config = {
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    print("No generation_config.json detected. Using default values.")
    with open("generation_config.json", "w") as config_file:
        json.dump(generation_config, config_file, indent=4)


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

if AI_EMOTION_DETECTION_FEATURE:
    from transformers import pipeline
    try:
        emotion_classifier = pipeline(
            'sentiment-analysis', model='j-hartmann/emotion-english-distilroberta-base')
        logging.info("Emotion classifier created successfully.")
    except Exception as e:
        logging.error("Failed to create emotion classifier, error:", f"{e}")

if AI_WIKIPEDIA_FEATURE:
    import nltk
    import wikipediaapi
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    try:
        wiki_wiki = wikipediaapi.Wikipedia(
            language='en',
            user_agent=f'{BOT_TWITCH_NAME} ; Python/3.x'
        )
        logging.info("Wikipedia instance created successfully.")
    except Exception as e:
        logging.error("Failed to create wikipedia instance, error:", f"{e}")

"""
This block handles the Google TTS API
Initializes model, language, pitch, speed, etc.
It also handles the audio buffer and queue
"""

if AI_TTS_FEATURE:
    import emoji
    from collections import deque
    from google.cloud import texttospeech
    from pydub import AudioSegment
    import io

    tts_queue = deque()
    is_playing = False

    client = texttospeech.TextToSpeechClient()

    logging.info("Google TTS API initialized successfully.")

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

        response = client.synthesize_speech(
            input=input_text, voice=voice, audio_config=audio_config
        )

        audio_buffer = io.BytesIO(response.audio_content)
        logging.info("Audio content written to in-memory buffer")

        return audio_buffer

    async def handle_tts_request(text):
        audio_buffer = synthesize_speech(text)
        tts_queue.append(audio_buffer)

        if not is_playing:
            await play_next_in_queue()

    async def play_next_in_queue():
        global is_playing
        while tts_queue:
            audio_buffer = tts_queue.popleft()
            is_playing = True
            play_audio_from_buffer(audio_buffer)
            audio_buffer.close()  # Close the buffer after playback

        is_playing = False

    def play_audio_from_buffer(audio_buffer):
        audio_buffer.seek(0)  # Ensure buffer is at the start

        # Convert MP3 to PCM
        audio_segment = AudioSegment.from_mp3(audio_buffer)
        pcm_buffer = io.BytesIO()
        audio_segment.export(pcm_buffer, format="wav")
        pcm_buffer.seek(0)

        with wave.open(pcm_buffer, "rb") as wf:
            p = pyaudio.PyAudio()

            stream = p.open(format=pyaudio.paInt16,
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True,)

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


async def query_gemini_with_STT(user_id, prompt):
    global message_count
    chat_session = model.start_chat(history=[])

    full_prompt = (
        "This is the user's current prompt:\n"
        f"{prompt}\n\n"
    )
    if AI_MEMORY_FEATURE:
        user_memory = await load_cached_memory(user_id)
        previous_data = "\n".join(
            ['User prompt: ' + interaction['prompt']
             + " Generated Response:" + interaction['response']
                for interaction in user_memory])
        full_prompt += ("Here is some previous data from the user to keep in mind:\n"
                        f"{previous_data}\n\n")

    if AI_EMOTION_DETECTION_FEATURE:
        emotion_analysis = emotion_classifier(prompt)
        detected_emotion = emotion_analysis[0]['label']
        emotion_confidence = emotion_analysis[0]['score']
        full_prompt += (f"Here is the current emotional state of the bot: \n{
                        mood_instructions}\n\n")

    if AI_MOODS_FEATURE:
        if AI_EMOTION_DETECTION_FEATURE:
            adjust_emotional_state_analysis(detected_emotion)
        print(emotional_states[current_emotion_index])
        print(mood_instructions)
        full_prompt += ("The user seems to be feeling "
                        f"{detected_emotion} with a confidence of {emotion_confidence:.2f}. \n"
                        "Please respond in a way that reflects this mood.\n\n")

    if AI_WIKIPEDIA_FEATURE:
        try:
            if prompt.lower().startswith(f"@{BOT_TWITCH_NAME.lower()}"):
                prompt = prompt[len(BOT_TWITCH_NAME) + 1:].strip()
            elif prompt.lower().startswith(BOT_NICKNAME.lower()):
                prompt = prompt[len(BOT_NICKNAME):].strip()

            wiki_summary = await fetch_information(prompt)

            if wiki_summary:
                full_prompt += (
                    "Additionally, here is some related factual "
                    "information from Wikipedia to consider in your response:\n"
                    f"{wiki_summary}\n\n"
                )

        except Exception as e:
            logging.error(f"Error in query processing: {e}")
            return "Sorry, I'm having trouble with the AI service right now."

        logging.info("Full prompt: " + full_prompt)

    response = chat_session.send_message(full_prompt)
    generated_text = response.text.strip()

    if AI_MEMORY_FEATURE:
        user_memory.append({'prompt': prompt, 'response': generated_text})
        await save_cached_memory(user_id, user_memory)

    return generated_text


"""
This is the experimental STT function.
"""

if AI_STT_FEATURE:
    import pyaudio
    import wave
    import threading
    from google.cloud import speech
    import numpy as np

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    INITIAL_THRESHOLD = 600
    SILENCE_DURATION = 1
    DEVICE_INDEX = 1
    NOISE_BUFFER_SIZE = 30

    p = pyaudio.PyAudio()

    if DEVICE_INDEX is None or DEVICE_INDEX >= p.get_device_count():
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            print(f"Device ID: {i}, Device Name: {info['name']}")
            print(f"Max Input Channels: {info['maxInputChannels']}")
        logging.info("Please add your microphone's device ID to DEVICE_INDEX and run again.")
        p.terminate()
        exit(1)

    frames = []
    recording = False
    dynamic_threshold = INITIAL_THRESHOLD
    noise_buffer = []
    last_audio_time = time.time()

    def calculate_noise_level():
        if not noise_buffer:
            return INITIAL_THRESHOLD
        return np.mean(noise_buffer)

    async def transcribe_audio(audio_buffer):
        try:
            client = speech.SpeechClient()
            audio_content = audio_buffer.getvalue()  # Read buffer content

            audio = speech.RecognitionAudio(content=audio_content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=RATE,
                language_code="en-US",
            )

            response = client.recognize(config=config, audio=audio)

            for result in response.results:
                logging.info("Transcript sent to Gemini API\n"
                             f"{result.alternatives[0].transcript}")
                response_text = await query_gemini_with_STT('owner', result.alternatives[0].transcript)
                logging.info("Response from Gemini API: " + response_text)
                clean_response = emoji.replace_emoji(response_text, replace='')
                audio_file_buffer = synthesize_speech(clean_response)
                logging.info("Generated speech audio buffer.")

                tts_queue.append(audio_file_buffer)

                if not is_playing:
                    await play_next_in_queue()
        except Exception as e:
            logging.error(f"Error processing audio: {e}")

    def process_audio(frames, channels, rate):
        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))
        audio_buffer.seek(0)  # Rewind buffer for reading

        def run_transcribe():
            asyncio.run(transcribe_audio(audio_buffer))

        threading.Thread(target=run_transcribe).start()

    def callback(in_data, frame_count, time_info, status):
        global recording, frames, last_audio_time

        audio_data = np.frombuffer(in_data, dtype=np.int16)
        audio_level = np.max(np.abs(audio_data))

        noise_buffer.append(audio_level)
        if len(noise_buffer) > NOISE_BUFFER_SIZE:
            noise_buffer.pop(0)

        dynamic_threshold = calculate_noise_level() * 1.5

        current_time = time.time()

        if audio_level > dynamic_threshold:
            last_audio_time = current_time
            if not recording:
                recording = True
                logging.debug("Input detected. Recording...")

        if recording:
            frames.append(in_data)

            if current_time - last_audio_time > SILENCE_DURATION:
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
                        input_device_index=DEVICE_INDEX,
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

if AI_MOODS_FEATURE:
    emotional_states = [
        "Angry",       # 0
        "Sad",        # 1
        "Nervous",    # 2
        "Confused",   # 3
        "Calm",        # 4
        "Happy",       # 5
        "Motivated",  # 6
        "Excited",     # 7
        "Curious",     # 8
        "Bored",       # 9

    ]

    try:
        with open("emotional_states.txt", "r") as file:
            emotional_state_descriptions = json.load(file)
        logging.info("Loaded Emotional States instructions")
    except FileNotFoundError:
        default_states = {
            "Happy": "The bot is cheerful and friendly, using positive and uplifting language.",
            "Sad": "The bot is empathetic and soothing, using comforting and gentle language.",
            "Angry": "The bot is assertive and forceful, using strong and direct language.",
            "Excited": "The bot is enthusiastic and energetic, using lively and engaging language.",
            "Confused": "The bot is uncertain and questioning, using exploratory and clarifying language.",
            "Bored": "The bot is indifferent and minimal, using straightforward and brief language.",
            "Curious": "The bot is inquisitive and interested, using probing and detailed language.",
            "Calm": "The bot is relaxed and composed, using calm and steady language.",
            "Nervous": "The bot is anxious and hesitant, using cautious and tentative language.",
            "Motivated": "The bot is encouraging and inspiring, using motivational and supportive language."
        }

        with open('emotional_states.txt', 'w') as file:
            json.dump(default_states, file, indent=4)

        print("No emotional_states.txt detected. A default one was created for you. "
              "You can customize your bot's moods by editing this file.")

        with open('emotional_states.txt', 'r') as file:
            emotional_state_descriptions = json.load(file)
        logging.info("Loaded Emotional States instructions")

    current_emotion_index = 5


def get_emotional_state(index):

    state = emotional_states[index]
    logging.info("Current emotional state: " + str(emotional_state_descriptions[state]))
    return emotional_state_descriptions[state]


"""
This code block handles the user interaction portion
Ranging from 0 to 7 (Angry to Excited)
"""
if AI_MOODS_FEATURE:
    MIN_EMOTIONAL_INDEX = 0
    MAX_EMOTIONAL_INDEX = 7

    mood_instructions = (
        f"{get_emotional_state(current_emotion_index)}. "
        "The bot's responses should reflect this mood. "
        "Please respond accordingly."
    )

    adjustment_counter = 0


def adjust_emotional_state(current_index, change):
    global adjustment_counter
    adjustment_counter += change
    if adjustment_counter >= ADJUSTMENT_WEIGHT or adjustment_counter <= -(ADJUSTMENT_WEIGHT):
        if current_index > MAX_EMOTIONAL_INDEX:
            new_index = 4
            adjustment_counter = 0
        else:
            new_index = max(MIN_EMOTIONAL_INDEX, min(
                MAX_EMOTIONAL_INDEX, current_index + change))
            adjustment_counter = 0
            logging.info("new emotional index: " + str(new_index))
            logging.info("new emotional state: " + str(get_emotional_state(new_index)))
        return new_index
    else:
        return current_index


def adjust_emotional_state_analysis(detected_emotion):
    global current_emotion_index

    if detected_emotion in ['anger', 'sadness', 'fear', 'disgust']:
        current_emotion_index = adjust_emotional_state(
            current_emotion_index, -1)
    elif detected_emotion in ['joy', 'surprise']:
        current_emotion_index = adjust_emotional_state(
            current_emotion_index, 1)
    elif detected_emotion in ['neutral']:
        current_emotion_index = adjust_emotional_state(
            current_emotion_index, 0)


"""
This block handles the reinforcement learning
It takes user feedback values
And adjusts the parameters of the model
"""

feedback_memory = []


def update_parameters_based_on_feedback():
    global generation_config, current_emotion_index, feedback_memory

    for feedback in feedback_memory:
        if 'positive' in feedback:
            generation_config['temperature'] += 0.1
            if AI_MOODS_FEATURE:
                current_emotion_index = adjust_emotional_state(
                    current_emotion_index, 1)
        else:
            generation_config['temperature'] -= 0.1
            if AI_MOODS_FEATURE:
                current_emotion_index = adjust_emotional_state(
                    current_emotion_index, -1)

        generation_config['temperature'] = max(
            0.1, min(1.5, generation_config['temperature']))

        if 'positive' in feedback:
            generation_config['top_k'] = min(
                100, generation_config.get('top_k', 50) + 1)
        else:
            generation_config['top_k'] = max(
                1, generation_config.get('top_k', 50) - 1)

        generation_config['top_k'] = max(
            1, min(100, generation_config['top_k']))

        if 'positive' in feedback:
            generation_config['top_p'] = min(
                1.0, generation_config.get('top_p', 0.9) + 0.01)
        else:
            generation_config['top_p'] = max(
                0.0, generation_config.get('top_p', 0.9) - 0.01)

        generation_config['top_p'] = max(
            0.0, min(1.0, generation_config['top_p']))

    if AI_MEMORY_FEATURE:
        with open("generation_config.json", "w") as config_file:
            json.dump(generation_config, config_file)

    logging.info("Processed feedback")

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

try:
    with open("chatbot_instructions.txt", "r") as instructions:
        chatbot_instructions = instructions.read().strip()
    logging.info("Loaded LLM instructions")
except FileNotFoundError:
    with open('chatbot_instruction.txt', 'w') as file:
        file.write('default instructions')
    print("No chatbot_instructions.txt detected. "
          "A default set was created for you. "
          "If you wish to customize your bots personality "
          "and instructions, edit this file.\n"
          "A prompt of 100-300 words is recommended if you want an AI with an in-depth pesonality.")
    with open('chatbot_instruction.txt', 'r') as file:
        chatbot_instructions = file.read().strip()
        logging.info("Loaded LLM instructions")


"""
Model settings and paramters
"""
try:
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        system_instruction=str(chatbot_instructions),
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH:
            FILTER_THRESHOLD,
            HarmCategory.HARM_CATEGORY_HARASSMENT:
            FILTER_THRESHOLD,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT:
            FILTER_THRESHOLD,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:
            FILTER_THRESHOLD,
        }
    )
    logging.info("Loaded LLM model")

except Exception as e:
    logging.error(f"Error loading model: {e}")
    print("Error loading model, please check logs for details.")


"""
Load and save the persistent memory
Additionally, cache the memory for faster loading
"""

if AI_MEMORY_FEATURE:
    import sqlite3
    conn = sqlite3.connect('chatbot_memory.db', check_same_thread=False)
    cursor = conn.cursor()
    logging.info("Loaded persistent memory")

    cursor.execute('''CREATE TABLE IF NOT EXISTS user_memory (
        user_id TEXT PRIMARY KEY,
        interactions TEXT
    )''')

    def save_memory(user_id, interactions):
        cursor.execute('''INSERT OR REPLACE INTO user_memory
                        (user_id, interactions)
                        VALUES (?, ?)''', (user_id, json.dumps(interactions)))
        conn.commit()
        logging.info("Saved to persistent memory")

    def load_memory(user_id):
        cursor.execute(
            'SELECT interactions FROM user_memory WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()
        logging.info("Loaded from persistent memory")
        return json.loads(row[0]) if row else []

    user_memory_cache = {}

    async def load_cached_memory(user_id):
        if user_id in user_memory_cache:
            return user_memory_cache[user_id]

        user_memory = load_memory(user_id)
        user_memory_cache[user_id] = user_memory
        return user_memory

    async def save_cached_memory(user_id, memory_data):

        save_memory(user_id, memory_data)
        user_memory_cache[user_id] = memory_data


"""
Download keyword files if necessary
"""


def download_nltk_data(resource_name, resource_url):
    try:
        nltk.data.find(resource_name)
        print(f"{resource_name} is already downloaded.")
    except LookupError:
        print(f"{resource_name} not found. Downloading...")
        nltk.download(resource_url)


if AI_WIKIPEDIA_FEATURE:
    download_nltk_data('corpora/stopwords.zip', 'stopwords')
    download_nltk_data('tokenizers/punkt.zip', 'punkt')


"""
This function extracts keywords for the wikipedia and duckduckgo APIs
"""


def extract_keywords(query):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(query)
    keywords = [word for word in word_tokens if word.isalnum()
                and word.lower() not in stop_words]

    if BOT_NICKNAME or BOT_TWITCH_NAME in keywords:
        try:
            keywords.remove(BOT_TWITCH_NAME)
        except ValueError:
            pass
        try:
            keywords.remove(BOT_NICKNAME)
        except ValueError:
            pass

    logging.info(("Extracted keywords" + str(keywords)))
    return keywords


"""
This code block searches wikipedia and duckduckgo APIs for relevant information
Based on keywords extracted from prompt
"""


async def fetch_information(query):
    keywords = extract_keywords(query)
    wikipedia_summary = None
    try:
        for keyword in keywords:
            page = wiki_wiki.page(keyword)
            if page.exists():
                wikipedia_summary = page.summary[:1000]
                break
    except Exception as e:
        logging.error(f"Error during Wikipedia search: {e}")
    logging.info("Wikipedia summary: " + str(wikipedia_summary))
    return wikipedia_summary


"""
This function formats the prompt to be sent to the API
It gathers historic user data if available
It then sends the generated response to the chat
And saves the user prompt and response to the memory
"""
message_count = 0


async def query_gemini_with_memory(user_id, prompt):
    global message_count

    chat_session = model.start_chat(history=[])

    full_prompt = (
        "This is the user's current prompt:\n"
        f"{prompt}\n\n"
    )

    if AI_MEMORY_FEATURE:
        user_memory = await load_cached_memory(user_id)
        previous_data = "\n".join(
            ['User prompt: ' + interaction['prompt']
             + " Generated Response:" + interaction['response']
                for interaction in user_memory])
        full_prompt += ("Here is some previous data from the user to keep in mind:\n"
                        f"{previous_data}\n\n")

    if AI_EMOTION_DETECTION_FEATURE:
        emotion_analysis = emotion_classifier(prompt)
        detected_emotion = emotion_analysis[0]['label']
        emotion_confidence = emotion_analysis[0]['score']
        full_prompt += (f"Here is the current emotional state of the bot: \n{
                        mood_instructions}\n\n")

    if AI_MOODS_FEATURE:
        if AI_EMOTION_DETECTION_FEATURE:
            adjust_emotional_state_analysis(detected_emotion)
        print(emotional_states[current_emotion_index])
        print(mood_instructions)
        full_prompt += ("The user seems to be feeling "
                        f"{detected_emotion} with a confidence of {emotion_confidence:.2f}. \n"
                        "Please respond in a way that reflects this mood.\n\n")

    if AI_WIKIPEDIA_FEATURE:
        try:
            if prompt.lower().startswith(f"@{BOT_TWITCH_NAME.lower()}"):
                prompt = prompt[len(BOT_TWITCH_NAME) + 1:].strip()
            elif prompt.lower().startswith(BOT_NICKNAME.lower()):
                prompt = prompt[len(BOT_NICKNAME):].strip()

            wiki_summary = await fetch_information(prompt)

            if wiki_summary:
                full_prompt += (
                    "Additionally, here is some related factual "
                    "information from Wikipedia to consider in your response:\n"
                    f"{wiki_summary}\n\n"
                )

        except Exception as e:
            logging.error(f"Error in query processing: {e}")
            return "Sorry, I'm having trouble with the AI service right now."

        logging.info("Full prompt: " + full_prompt)

    response = chat_session.send_message(full_prompt)
    generated_text = response.text.strip()

    if AI_LEARNING_FEATURE:
        add_feedback_user_id(user_id)

    if AI_MEMORY_FEATURE:
        user_memory.append({'prompt': prompt, 'response': generated_text})
        await save_cached_memory(user_id, user_memory)

    message_count = 0

    return generated_text


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
            response = await query_gemini_with_memory(user_id, prompt)
            logging.info(f"Generated response from Gemini: {response}")
        except Exception as e:
            logging.error(f"Error processing message from Gemini: {e}")

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
            response = await query_gemini_with_memory(user_id, prompt)
            logging.info(f"Generated response from Gemini: {response}")

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
            logging.error(f"Error processing message from Gemini: {e}")
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
    global message_count, current_emotion_index

    while True:
        wait_time = random.randint(*AUTOMATED_RESPONSE_TIME_RANGE)
        await asyncio.sleep(wait_time)
        cleanup_memory()
        if AI_MOODS_FEATURE:
            if random.randint(0, 2) == 0:
                current_emotion_index = 9
                logging.info(f"Emotion changed to {get_emotional_state(current_emotion_index)}")
            elif random.randint(0, 2) == 0:
                current_emotion_index = 8
                logging.info(f"Emotion changed to {get_emotional_state(current_emotion_index)}")
            elif random.randint(0, 2) == 0:
                random_emotion = random.randint(0, 7)
                current_emotion_index = random_emotion
                logging.info(f"Emotion changed to {get_emotional_state(current_emotion_index)}")
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
Those with if.ctx.author.ud in AUTHORIZED_USERS_LIST are authorized user only commands

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


@ bot.command(name='Wikipedia')
async def wikipedia_flag(ctx):
    global AI_WIKIPEDIA_FEATURE
    if ctx.author.ud in AUTHORIZED_USERS_LIST:
        AI_WIKIPEDIA_FEATURE = not AI_WIKIPEDIA_FEATURE


@ bot.command(name='TTS')
async def TTS_flag(ctx):
    global AI_TTS_FEATURE
    if ctx.author.ud in AUTHORIZED_USERS_LIST:
        AI_TTS_FEATURE = not AI_TTS_FEATURE


@ bot.command(name='Memory')
async def memory_flag(ctx):
    global AI_MEMORY_FEATURE
    if ctx.author.ud in AUTHORIZED_USERS_LIST:
        AI_MEMORY_FEATURE = not AI_MEMORY_FEATURE


@ bot.command(name='Moods')
async def moods_flag(ctx):
    global AI_MOODS_FEATURE
    if ctx.author.ud in AUTHORIZED_USERS_LIST:
        AI_MOODS_FEATURE = not AI_MOODS_FEATURE


@ bot.command(name='Detection')
async def emotion_detection_flag(ctx):
    global AI_EMOTION_DETECTION_FEATURE
    if ctx.author.ud in AUTHORIZED_USERS_LIST:
        AI_EMOTION_DETECTION_FEATURE = not AI_EMOTION_DETECTION_FEATURE


@ bot.command(name='Learning')
async def learning_flag(ctx):
    global AI_LEARNING_FEATURE
    if ctx.author.ud in AUTHORIZED_USERS_LIST:
        AI_LEARNING_FEATURE = not AI_LEARNING_FEATURE

try:
    bot.run()
except AttributeError:
    logging.error("Error running bot:\n"
                  "Please check your Twitch CLIENT ID and OAUTH Keys and try again."
                  )
