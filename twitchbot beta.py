import logging
import asyncio
import random
import json
import os
import time

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
TTS_SPEAKING_RATE = 1.15


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
AI_TTS_FEATURE = False  # TTS generation of AI responses

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
This block initializes the Google TTS API
Adjusts model, language, pitch, speed, etc
"""

if AI_TTS_FEATURE:
    from google.cloud import texttospeech
    import pygame
    import emoji
    from collections import deque

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

        audio_file = f"AI_tts_response_{int(time.time())}.mp3"

        with open(audio_file, "wb") as out:
            out.write(response.audio_content)
            logging.info(f"Audio content written to file: {audio_file}")

        return audio_file


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

    emotional_state_descriptions = {
        "Happy": "The bot is cheerful and friendly, using positive and uplifting language.",
        "Sad": "The bot is empathetic and soothing, using comforting and gentle language.",
        "Angry": (
            "The bot is assertive and forceful, using strong and direct language."
        ),
        "Excited": (
            "The bot is enthusiastic and energetic, using lively and engaging language."
        ),
        "Confused": (
            "The bot is uncertain and questioning, using exploratory and clarifying language."
        ),
        "Bored": (
            "The bot is indifferent and minimal, using straightforward and brief language."
        ),
        "Curious": (
            "The bot is inquisitive and interested, using probing and detailed language."
        ),
        "Calm": (
            "The bot is relaxed and composed, using calm and steady language."
        ),
        "Nervous": (
            "The bot is anxious and hesitant, using cautious and tentative language."
        ),
        "Motivated": (
            "The bot is encouraging and inspiring, using motivational and supportive language."
        )
    }

    current_emotion_index = 5


def get_emotional_state(index):

    state = emotional_states[index]
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
          "One was created for you, "
          "if you wish to customize your bots personality, "
          "and instructions.")
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
"""

if AI_MEMORY_FEATURE:
    import sqlite3
    conn = sqlite3.connect('chatbot_memory.db')
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
        user_memory = load_memory(user_id)
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
        save_memory(user_id, user_memory)

    message_count = 0

    return generated_text

"""
Command to describe the AI to the user
"""


@ bot.command(name='AI')
async def ai(ctx):
    await ctx.send("I'm a bot created by @thejoshinatah! ^.^ "
                   "I make use of multiple APIs "
                   "and models to generate responses! "
                   "If you'd like to know more, check out our "
                   "github https://github.com/exosever/chatbot/"
                   )

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


async def play_next_in_queue():
    global is_playing
    while tts_queue:

        audio_file = tts_queue.popleft()
        is_playing = True
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()

        logging.info(f"Playing audio file: {audio_file}")

        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.1)

        logging.info("Audio playback finished.")

        is_playing = False

        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
            os.remove(audio_file)
            logging.info(f"Removed audio file: {audio_file}")
        except Exception as e:
            logging.error(f"Error removing file {audio_file}: {e}")


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
        audio_file = synthesize_speech(clean_response)
        logging.info(f"Generated speech audio file: {audio_file}")

        tts_queue.append(audio_file)

        if not is_playing:
            await message.channel.send(response)
            await play_next_in_queue()

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


async def automated_response():
    global message_count, current_emotion_index

    while True:
        wait_time = random.randint(*AUTOMATED_RESPONSE_TIME_RANGE)
        await asyncio.sleep(wait_time)
        if AI_MOODS_FEATURE:
            if random.randint(0, 1) == 0:
                current_emotion_index = 9
                logging.info(f"Emotion changed to {get_emotional_state(current_emotion_index)}")
            elif random.randint(0, 1) == 0:
                current_emotion_index = 8
                logging.info(f"Emotion changed to {get_emotional_state(current_emotion_index)}")
            elif random.randint(0, 1) == 0:
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

try:
    bot.run()
except Exception as e:
    logging.error("Error running bot:\n"
                  "Please check your Twitch CLIENT ID and OAUTH Keys and try again."
                  )
