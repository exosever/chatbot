import logging
import asyncio
import random
import json
import os
import time
import requests
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

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

"""
These are the environmental variables for the API keys
as well as the initial setup variables
"""

load_dotenv('chatbot_variables.env')

TWITCH_OAUTH_TOKEN = os.getenv('TWITCH_OAUTH_TOKEN')
TWITCH_CLIENT_ID = os.getenv('TWITCH_CLIENT_ID')
TWITCH_CHANNEL_NAME = os.getenv('TWITCH_CHANNEL_NAME')
genai.configure(api_key=os.getenv('GENAI_API_KEY'))
BOT_NAME = os.getenv('BOT_NAME')
BOT_NICKNAME = os.getenv('BOT_NICKNAME')
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.getcwd(),
#                                                            'google.json')


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
    with open("generation_config.json", "w") as config_file:
        json.dump(generation_config, config_file, indent=4)

"""
This block handle the Google TTS API
Adjust model, language, pitch, speed, etc
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

current_emotional_index = 5


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
    f"{get_emotional_state(current_emotional_index)}. "
    "The bot's responses should reflect this mood. "
    "Please respond accordingly."
)

adjustment_counter = 0


def adjust_emotional_state(current_index, change):
    global adjustment_counter
    print(adjustment_counter)
    adjustment_counter += change
    print(change)
    print(adjustment_counter)
    if adjustment_counter >= 3 or adjustment_counter <= -3:
        if current_index > MAX_EMOTIONAL_INDEX:
            new_index = 4
        new_index = max(MIN_EMOTIONAL_INDEX, min(
            MAX_EMOTIONAL_INDEX, current_index + change))
        return new_index
    else:
        return current_index


def adjust_emotional_state_analysis(detected_emotion):
    global current_emotional_index

    if detected_emotion in ['Angry', 'Sad', 'Fearful', 'Disgusted']:
        current_emotional_index = adjust_emotional_state(
            current_emotional_index, -1)
    elif detected_emotion in ['Happy', 'Excited', 'Surprised']:
        current_emotional_index = adjust_emotional_state(
            current_emotional_index, 1)
    elif detected_emotion in ['Neutral']:
        current_emotional_index = adjust_emotional_state(
            current_emotional_index, 0)


"""
This block handles the reinforcement learning
It takes user feedback values
And adjusts the parameters of the model
"""
feedback_counter = 0
feedback_memory = []


def update_parameters_based_on_feedback():
    global generation_config, feedback_counter, current_emotional_index

# This accesses a tuple, User ID with a dictionary : positive and negative feedback
# Submits the feedback as one number
#    if feedback['positive']:

# KeyError: 'positive'

    for feedback in feedback_memory:
        print(feedback)
        if feedback['positive']:
            generation_config['temperature'] += 0.1
            current_emotional_index = adjust_emotional_state(
                current_emotional_index, 1)
        else:
            generation_config['temperature'] -= 0.1
            current_emotional_index = adjust_emotional_state(
                current_emotional_index, -1)

        generation_config['temperature'] = max(
            0.1, min(1.5, generation_config['temperature']))

        if feedback['positive']:
            generation_config['top_k'] = min(
                100, generation_config.get('top_k', 50) + 1)  # Increase top_k
        else:
            generation_config['top_k'] = max(
                1, generation_config.get('top_k', 50) - 1)  # Decrease top_k

        generation_config['top_k'] = max(
            1, min(100, generation_config['top_k']))

        if feedback['positive'] > 0.8:
            generation_config['top_p'] = min(
                1.0, generation_config.get('top_p', 0.9) + 0.01)
        else:
            generation_config['top_p'] = max(
                0.0, generation_config.get('top_p', 0.9) - 0.01)

        generation_config['top_p'] = max(
            0.0, min(1.0, generation_config['top_p']))

    with open("generation_config.json", "w") as config_file:
        json.dump(generation_config, config_file)

    feedback_counter = 0


"""
Load the instructions for the bot if they exist
"""

try:
    with open("chatbot_instructions.txt", "r") as instructions:
        chatbot_instructions = instructions.read().strip()
except FileNotFoundError:
    pass


"""
Model settings and paramters

BLOCK_ONLY_HIGH
BLOCK_ONLY_MEDIUM
BLOCK_ONLY_LOW
BLOCK_HIGH_AND_MEDIUM
BLOCK_HIGH_AND_MEDIUM_AND_LOW
BLOCK_HIGH_AND_MEDIUM_AND_LOW_AND_NONE
"""
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction=str(chatbot_instructions),
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH:
        HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HARASSMENT:
        HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT:
        HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:
        HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }
)

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
Initialize the bot
"""

bot = commands.Bot(
    token=TWITCH_OAUTH_TOKEN,
    client_id=TWITCH_CLIENT_ID,
    nick=BOT_NAME,
    prefix='!',
    initial_channels=[TWITCH_CHANNEL_NAME]
)

"""
Initialize the emotion detection pipeline
"""
emotion_classifier = pipeline(
    'sentiment-analysis', model='bhadresh-savani/distilbert-'
                                'base-uncased-emotion')


"""
Counters to keep chatbot, commands, and automated messages from spamming
"""

message_count = 0
last_message_time = None

"""
Function to query the Wikipedia API
Using keywords extrapulated from prompt
"""

wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent=f'{BOT_NAME} ; Python/3.x'
)

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

"""
This function extracts keywords for the prompt
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
    search_query = " ".join(keywords)

    wikipedia_summary = None
    try:
        for keyword in keywords:
            page = wiki_wiki.page(keyword)
            if page.exists():
                wikipedia_summary = page.summary[:1000]
                break
    except Exception as e:
        logging.error(f"Error during Wikipedia search: {e}")

    duckduckgo_result = None
    try:
        response = requests.get('https://api.duckduckgo.com/', params={
            'q': search_query,
            'format': 'json',
            'no_html': 1,
            'no_redirect': 1,
            'skip_disambig': 1
        })
        data = response.json()
        if 'AbstractText' in data and data['AbstractText']:
            duckduckgo_result = data['AbstractText']
        elif 'RelatedTopics' in data and data['RelatedTopics']:
            duckduckgo_result = data['RelatedTopics'][0].get(
                'Text', 'No information found.')
        else:
            pass

    except Exception as e:
        logging.error(f"Error during DuckDuckGo search: {e}")

    return wikipedia_summary, duckduckgo_result


"""
This function formats the prompt to be sent to the API
It gathers historic user data if available
It then sends the generated response to the chat
And saves the user prompt and response to the memory
"""


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
    print(mood_instructions)
    print(detected_emotion)
    print(prompt)

    try:
        wiki_summary, duckduckgo_result = await fetch_information(prompt)

        if wiki_summary:
            full_prompt += (
                "\n\nAdditionally, here is some related factual"
                "information from Wikipedia to consider in your response:\n"
                f"{wiki_summary}"
            )

        if duckduckgo_result:
            print(duckduckgo_result)
            full_prompt += (
                "\n\nHere is some additional information from the"
                "web to consider in your response:\n"
                f"{duckduckgo_result}"
            )

    except Exception as e:
        logging.error(f"Error in query processing: {e}")
        return "Sorry, I'm having trouble with the AI service right now."

    response = chat_session.send_message(full_prompt)
    generated_text = response.text.strip()

    # Update the user's memory with the new interaction
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


@bot.command(name='feedback')
async def feedback(ctx, feedback_type):
    global feedback_counter

    if feedback_type.lower() == 'good':
        feedback_memory.append({'positive': 1})
        await ctx.send("Thank's for letting me know! ^.^")
    elif feedback_type.lower() == 'bad':
        feedback_memory.append({'negative': 1})
        await ctx.send("I'm sorry about that! Thanks for "
                       "helping me do better next time!")
    feedback_counter += 1

    if feedback_counter > 10:
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
            await channel.send(
                'Hello everyone! How are you all doing?'
            )
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
        message.content.lower().startswith(BOT_NICKNAME)
        or message.content.lower().startswith(f"@{BOT_NAME}")
    ):
        user_id = str(message.author.id)
        prompt = message.content[len(BOT_NICKNAME):].strip()
        logging.debug(f"Processed prompt: {prompt}")
        response = await query_gemini_with_memory(user_id, prompt)
        try:
            await message.channel.send(response)

            if current_time - last_feedback_message_time >= 120:
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
    global message_count, current_emotional_index

    while True:
        wait_time = random.randint(600, 1200)
        await asyncio.sleep(wait_time)
        if random.randint(0, 1) == 0:
            current_emotional_index = 9
        elif random.randint(0, 1) == 0:
            current_emotional_index = 8
        elif random.randint(0, 1) == 0:
            random_emotion = random.randint(0, 7)
            current_emotional_index = random_emotion
        if message_count >= 10:
            try:
                channel = bot.get_channel(TWITCH_CHANNEL_NAME)
                if channel:
                    await channel.send(
                        f"Hey There! I'm {BOT_NICKNAME}, "
                        "your friendly neighborhood racoon! "
                        "Feel free to chat with me "
                        "by calling my name first ^.^ "
                        f"ie: {BOT_NICKNAME}, why is Josh such a great name?"
                    )

                    logging.info('Sent automated response to chat.')
                    message_count = 0
                else:
                    logging.error(f"Channel {TWITCH_CHANNEL_NAME} not found.")
            except Exception as e:
                logging.error(f"Error sending automated message: {e}")
        else:
            logging.debug(f"Not enough messages received yet: {message_count}")

bot.run()
