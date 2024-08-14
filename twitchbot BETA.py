import logging
import asyncio
import random
import json
import os
import time
import sqlite3
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline
import wikipediaapi
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

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.getcwd(),
#                                      'google.json')

"""
--------------------------------------------------------------------------------
BOT CONFIGURATION

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
--------------------------------------------------------------------------------
FEATURE FLAGS

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


"""
--------------------------------------------------------------------------------
CORE FUNCTIONALITY BELOW

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

try:
    load_dotenv('chatbot_variables.env')
except FileNotFoundError:
    with open('chatbot_variables.env', 'w') as file:
        file.write('TWITCH_OAUTH_TOKEN=AMOURANTH'
                   'TWITCH_CLIENT_ID=DRDISRESPECT'
                   'GENAI_API_KEY=GOOGLEISGREAT'
                   'TWITCH_CHANNEL_NAME=bobross')
    print("No .env detected. Chatbot_variables.env created. "
          "Please add your API keys to this file and run again.")
    exit()


TWITCH_OAUTH_TOKEN = os.getenv('TWITCH_OAUTH_TOKEN')
TWITCH_CLIENT_ID = os.getenv('TWITCH_CLIENT_ID')
TWITCH_CHANNEL_NAME = os.getenv('TWITCH_CHANNEL_NAME')
GENAI_API_KEY = os.getenv('GENAI_API_KEY')

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
except Exception as e:
    logging.error("Failed to create bot instance, error:", f"{e}")
    print("An error occurred while creating the bot instance. Check the log for details.")

try:
    emotion_classifier = pipeline(
        'sentiment-analysis', model='j-hartmann/emotion-english-distilroberta-base')
except Exception as e:
    logging.error("Failed to create emotion classifier, error:", f"{e}")

try:
    wiki_wiki = wikipediaapi.Wikipedia(
        language='en',
        user_agent=f'{BOT_TWITCH_NAME} ; Python/3.x'
    )
except Exception as e:
    logging.error("Failed to create wikipedia instance, error:", f"{e}")

"""
This block initializes the Google TTS API
Adjusts model, language, pitch, speed, etc
"""

# client = texttospeech.TextToSpeechClient()


# def synthesize_speech(text, pitch=13.60, speaking_rate=1.19):

#    input_text = texttospeech.SynthesisInput(text=text)

#    voice = texttospeech.VoiceSelectionParams(
#        language_code="en-US",
#        name="en-US-Neural2-I",
#    )

#    audio_config = texttospeech.AudioConfig(
#        audio_encoding=texttospeech.AudioEncoding.MP3,
#        pitch=pitch,
#        speaking_rate=speaking_rate,
#    )

#    response = client.synthesize_speech(
#        input=input_text, voice=voice, audio_config=audio_config
#    )

#    audio_file = "output.mp3"

#    with open(audio_file, "wb") as out:
#        out.write(response.audio_content)

#    return audio_file

"""
This code block establishes the parameters for the bots emotional states
States are the emotional value, which corrospond to a description of the mood
These values can be adjusted by a slider,
by a random number, or chosen specifically
"""

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
    "Angry": "The bot is assertive and forceful, using strong and direct language.",
    "Excited": "The bot is enthusiastic and energetic, using lively and engaging language.",
    "Confused": "The bot is uncertain and questioning, using exploratory and clarifying language.",
    "Bored": "The bot is indifferent and minimal, using straightforward and brief language.",
    "Curious": "The bot is inquisitive and interested, using probing and detailed language.",
    "Calm": "The bot is relaxed and composed, using calm and steady language.",
    "Nervous": "The bot is anxious and hesitant, using cautious and tentative language.",
    "Motivated": "The bot is encouraging and inspiring, using motivational and supportive language."
}

current_emotion_index = 5
print("Starting emotional index: " + str(current_emotion_index))
print("Starting emotional state: " + str(emotional_states[current_emotion_index]))


def get_emotional_state(index):

    state = emotional_states[index]
    return emotional_state_descriptions[state]


"""
This code block handles the user interaction portion
Ranging from 0 to 7 (Angry to Excited)
"""

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
        print("new emotional index: " + str(new_index))
        print("new emotional state: " + str(get_emotional_state(new_index)))
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
            current_emotion_index = adjust_emotional_state(
                current_emotion_index, 1)
        else:
            generation_config['temperature'] -= 0.1
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

    with open("generation_config.json", "w") as config_file:
        json.dump(generation_config, config_file)

    feedback_memory = []

    print("Feedback submitted to reainforcement learning")


"""
Load the instructions for the bot personality if they exist
"""

try:
    with open("chatbot_instructions.txt", "r") as instructions:
        chatbot_instructions = instructions.read().strip()
except FileNotFoundError:
    with open('chatbot_instruction.txt', 'w') as file:
        file.write('')
    print("No chatbot_instructions.txt detected. "
          "One was created for you, "
          "if you wish to customize your bots personality, "
          "and instructions.")


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
except Exception as e:
    logging.error(f"Error loading model: {e}")
    print("Error loading model, please check logs for details.")


"""
Load and save the persistent memory
"""

conn = sqlite3.connect('chatbot_memory.db')
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS user_memory (
    user_id TEXT PRIMARY KEY,
    interactions TEXT
)''')


def save_memory(user_id, interactions):
    cursor.execute('''INSERT OR REPLACE INTO user_memory
                      (user_id, interactions)
                      VALUES (?, ?)''', (user_id, json.dumps(interactions)))
    conn.commit()


def load_memory(user_id):
    cursor.execute(
        'SELECT interactions FROM user_memory WHERE user_id = ?', (user_id,))
    row = cursor.fetchone()
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
    user_memory = load_memory(user_id)
    previous_data = "\n".join(
        [interaction['prompt'] + " " + interaction['response']
            for interaction in user_memory])

    emotion_analysis = emotion_classifier(prompt)
    detected_emotion = emotion_analysis[0]['label']
    emotion_confidence = emotion_analysis[0]['score']
    adjust_emotional_state_analysis(detected_emotion)

    full_prompt = (
        f"Here is the current emotional state of the bot: {
            mood_instructions}. "
        "\n\nThe user seems to be feeling "
        f"{detected_emotion} with a confidence of {emotion_confidence:.2f}. "
        "Please respond in a way that reflects this mood.\n\n"
        "Here is some previous data from the user to keep in mind:\n"
        f"{previous_data}\n\n"
        "This is the user's current prompt:\n"
        f"{prompt}"
    )
    print(emotional_states[current_emotion_index])
    print(mood_instructions)
    print(detected_emotion)
    print(emotion_confidence)
    print(prompt)

    try:
        wiki_summary = await fetch_information(prompt)

        if wiki_summary:
            full_prompt += (
                "\n\nAdditionally, here is some related factual"
                "information from Wikipedia to consider in your response:\n"
                f"{wiki_summary}"
            )

    except Exception as e:
        logging.error(f"Error in query processing: {e}")
        return "Sorry, I'm having trouble with the AI service right now."

    response = chat_session.send_message(full_prompt)
    generated_text = response.text.strip()

    add_feedback_user_id(user_id)

    user_memory.append({'prompt': prompt, 'response': generated_text})
    save_memory(user_id, user_memory)

    message_count = 0

    return generated_text

"""
Command to describe the AI to the user
"""


@bot.command(name='AI')
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

# Need to add system to only allow user who sent a prompt to give feedback, and only once

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


@bot.command(name='feedback')
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

"""
This function redeems a TTS Message channel point reward to send
The generated response through the Google TTS API to the chat
"""


# @bot.event()
# async def on_channel_points_redeem(redemption):
#    logging.info(f"Channel point redemption detected: {
#                 redemption.reward.title}")
#    print(f"Channel point redemption detected: {redemption.reward.title}")

#    if redemption.reward.title == "TTS Message":
#        logging.info(f"TTS Message reward redeemed by user {
#                     redemption.user.name} ({redemption.user.id})")
#        user_message = redemption.user_input
#        logging.info(f"User message: {user_message}")

#        try:
#            response = await query_gemini_with_memory(str(redemption.user.id),
#                                                      user_message)
#            logging.info(f"Generated response from Gemini: {response}")

#            audio_file = synthesize_speech(response)
#            logging.info(f"Generated speech audio file: {audio_file}")

#            pygame.mixer.init()
#            pygame.mixer.music.load(audio_file)
#            pygame.mixer.music.play()

#            logging.info("Playing audio file...")

#            while pygame.mixer.music.get_busy():
#                pygame.time.Clock().tick(10)

#            logging.info("Audio playback finished.")

#        except Exception as e:
#            logging.error(f"Error in TTS redemption processing: {e}")


"""
This function is used to send a message to the chat when the bot is ready.
Debug connectivity to Twitch
"""


@bot.event()
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
This function checks if the message is from a user
If so, it formats the prompt to be sent to the API
It also sends the generated response to the chat
And only after 60 seconds sends a message asking the user to use !feedback
"""
last_feedback_message_time = 0


@bot.event()
async def event_message(message):
    global message_count, last_feedback_message_time

    current_time = time.time()

    if bot.nick.lower() not in str(message.author).lower():
        message_count += 1

    if (
        message.content.lower().startswith(BOT_NICKNAME.lower())
        or message.content.lower().startswith(f"@{BOT_TWITCH_NAME.lower()}")
    ):
        user_id = str(message.author.id)
        prompt = message.content.strip()

        logging.debug(f"Processed prompt: {prompt}")
        response = await query_gemini_with_memory(user_id, prompt)
        try:
            await message.channel.send(response)

            if current_time - last_feedback_message_time >= FEEDBACK_TIME_THRESHOLD:
                await message.channel.send(
                    'Be sure to use !feedback <good/bad> '
                    'to let me know if I did a good job!'
                )

                last_feedback_message_time = current_time

            logging.info(f"Sent response: {response}")
        except Exception as e:
            logging.error(f"Error sending message: {e}")
    else:
        logging.debug(f"Ignoring message: {message.content}")

"""
This is a loop that will send a message to chat
after at least 10 user messages have been received
and 10-20 minutes have passed

This block also updates the parameters of the model
based off the feedback received
"""


async def automated_response():
    global message_count, current_emotion_index

    while True:
        wait_time = random.randint(*AUTOMATED_RESPONSE_TIME_RANGE)
        await asyncio.sleep(wait_time)
        if random.randint(0, 1) == 0:
            current_emotion_index = 9
        elif random.randint(0, 1) == 0:
            current_emotion_index = 8
        elif random.randint(0, 1) == 0:
            random_emotion = random.randint(0, 7)
            current_emotion_index = random_emotion
        if message_count >= 10:
            try:
                channel = bot.get_channel(TWITCH_CHANNEL_NAME)
                if channel:
                    await channel.send(AUTOMATED_MESSAGE)

                    message_count = 0
                else:
                    logging.error(f"Channel {TWITCH_CHANNEL_NAME} not found.")
            except Exception as e:
                logging.error(f"Error sending automated message: {e}")
        else:
            logging.debug(f"Not enough messages received yet: {message_count}")

bot.run()
