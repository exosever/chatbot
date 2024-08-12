import logging
import asyncio
import random
import json
import pygame
import wikipediaapi
import os
import time
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import google.generativeai as genai
from twitchio.ext import commands
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv
from google.cloud import texttospeech

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

"""
These are the environmental variables for the API keys
"""

load_dotenv('chatbot_variables.env')

TWITCH_OAUTH_TOKEN = os.getenv('TWITCH_OAUTH_TOKEN')
TWITCH_CLIENT_ID = os.getenv('TWITCH_CLIENT_ID')
TWITCH_CHANNEL_NAME = os.getenv('TWITCH_CHANNEL_NAME')
genai.configure(api_key=os.getenv('GENAI_API_KEY'))
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.getcwd(),
                                                            'google.json')


"""
This block handle the Google TTS API
Adjust model, language, pitch, speed, etc
"""

client = texttospeech.TextToSpeechClient()


def synthesize_speech(text, pitch=13.60, speaking_rate=1.19):

    input_text = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Neural2-I",
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        pitch=pitch,
        speaking_rate=speaking_rate,
    )

    response = client.synthesize_speech(
        input=input_text, voice=voice, audio_config=audio_config
    )

    audio_file = "output.mp3"

    with open(audio_file, "wb") as out:
        out.write(response.audio_content)

    return audio_file


"""
Set the bot name(twitch account) and nickname
Also load the generation config from a JSON file
If a config is not present, a default one will be created
"""

BOT_NAME = 'chesterbot9000'
BOT_NICKNAME = 'chester'

try:
    with open("generation_config.json", "r") as config_file:
        generation_config = json.load(config_file)
except FileNotFoundError:
    # If the file doesn't exist, use default settings
    generation_config = {
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

"""
This block handles the reinforcement learning
It takes user feedback values
And adjusts the parameters of the model
"""
feedback_counter = 0
feedback_memory = {}


def update_parameters_based_on_feedback():
    global generation_config, feedback_counter

    for user_id, feedback in feedback_memory.items():

        feedback_ratio = feedback['positive'] / \
            (feedback['positive'] + feedback['negative'] + 1)

        if feedback_ratio > 0.8:
            generation_config['temperature'] += 0.1
        else:
            generation_config['temperature'] -= 0.1

        generation_config['temperature'] = max(
            0.1, min(1.5, generation_config['temperature']))

        if feedback_ratio > 0.8:
            generation_config['top_k'] = min(
                100, generation_config.get('top_k', 50) + 1)  # Increase top_k
        else:
            generation_config['top_k'] = max(
                1, generation_config.get('top_k', 50) - 1)  # Decrease top_k

        generation_config['top_k'] = max(
            1, min(100, generation_config['top_k']))

        if feedback_ratio > 0.8:
            generation_config['top_p'] = min(
                1.0, generation_config.get('top_p', 0.9) + 0.01)
        else:
            generation_config['top_p'] = max(
                0.0, generation_config.get('top_p', 0.9) - 0.01)

        generation_config['top_p'] = max(
            0.0, min(1.0, generation_config['top_p']))

    with open("generation_config.json", "w") as config_file:
        json.dump(generation_config, config_file)

    save_memory()

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

try:
    with open("chatbot_memory.json", "r") as memory_file:
        chatbot_memory = json.load(memory_file)
except FileNotFoundError:
    chatbot_memory = {}


def save_memory():
    with open("chatbot_memory.json", "w") as memory_file:
        json.dump(chatbot_memory, memory_file)


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
Counters to keep chatbot, commands, and automated messages from spamming
"""

message_count = 0
last_message_time = None

"""
Function to query the Wikipedia API
"""

wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent=f'{BOT_NAME} ; Python/3.x'
)

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


def extract_keywords(query):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(query)
    keywords = [word for word in word_tokens if word.isalnum()
                and word.lower() not in stop_words]
    return keywords


async def fetch_wikipedia_summary(query):
    keywords = extract_keywords(query)
    for keyword in keywords:
        page = wiki_wiki.page(keyword)
        if page.exists():
            return page.summary[:1000]  # Return a limited summary
    return "Sorry, I couldn't find relevant information."


"""
This function formats the prompt to be sent to the API
It gathers historic user data if available
It then sends the generated response to the chat
And saves the user prompt and response to the memory
"""


async def query_gemini_with_memory(user_id, prompt):
    global message_count

    chat_session = model.start_chat(history=[])

    user_memory = chatbot_memory.get(user_id, {}).get('prompts', [])
    previous_data = "\n".join(user_memory)

    full_prompt = (
        "Here is some previous data from the user to keep in mind:\n"
        f"{previous_data}\n\n"
        "This is the user's current prompt:\n"
        f"{prompt}"
    )

    try:
        wiki_summary_task = asyncio.create_task(
            fetch_wikipedia_summary(prompt))

        wiki_summary = await wiki_summary_task

        if wiki_summary:
            full_prompt += (
                "\n\nAdditionally, here is some related factual"
                "information from Wikipedia:\n"
                f"{wiki_summary}"
            )

        response = chat_session.send_message(full_prompt)
        generated_text = response.text.strip()

        if user_id not in chatbot_memory:
            chatbot_memory[user_id] = {'prompts': []}
        chatbot_memory[user_id]['prompts'].append(prompt)

        save_memory()

        message_count = 0

        return generated_text

    except Exception as e:
        logging.error(f"Error in query processing: {e}")
        return "Sorry, I'm having trouble with the AI service right now."

"""
Command to describe the AI to the user
"""


@bot.command(name='AI')
async def ai(ctx):
    await ctx.send("I'm a bot created by @thejoshinatah! ^.^"
                   "I am powered by Google Gemini and have the ability"
                   "to learn and remember from our conversations!"
                   )

"""
This function saves the feedback from the user
Simple positive or negative feedback will be used to adjust
the parameters of the model
"""


@bot.command(name='feedback')
async def feedback(ctx, feedback_type):
    global feedback_counter
    user_id = str(ctx.author.id)
    if user_id not in feedback_memory:
        feedback_memory[user_id] = {'positive': 0, 'negative': 0}

    if feedback_type.lower() == 'good':
        feedback_memory[user_id]['positive'] += 1
        await ctx.send("Thank's for letting me know! ^.^")
    elif feedback_type.lower() == 'bad':
        feedback_memory[user_id]['negative'] += 1
        await ctx.send("I'm sorry about that! Thanks for"
                       "helping me do better next time!")
    else:
        feedback_counter += 1

"""
This function redeems a TTS Message channel point reward to send
The generated response through the Google TTS API to the chat
"""


@bot.event()
async def on_channel_points_redeem(redemption):
    logging.info(f"Channel point redemption detected: {
                 redemption.reward.title}")
    print(f"Channel point redemption detected: {redemption.reward.title}")

    if redemption.reward.title == "TTS Message":
        logging.info(f"TTS Message reward redeemed by user {
                     redemption.user.name} ({redemption.user.id})")
        user_message = redemption.user_input
        logging.info(f"User message: {user_message}")

        try:
            # Query the Gemini model with memory
            response = await query_gemini_with_memory(str(redemption.user.id),
                                                      user_message)
            logging.info(f"Generated response from Gemini: {response}")

            # Synthesize speech with the response
            audio_file = synthesize_speech(response)
            logging.info(f"Generated speech audio file: {audio_file}")

            # Initialize pygame mixer
            pygame.mixer.init()
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()

            logging.info("Playing audio file...")

            # Wait until the playback is finished
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            logging.info("Audio playback finished.")

        except Exception as e:
            logging.error(f"Error in TTS redemption processing: {e}")


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
                'Hello everyone! I am now online and ready to chat!'
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

            if current_time - last_feedback_message_time >= 60:
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
    global message_count

    while True:
        wait_time = random.randint(600, 1200)
        await asyncio.sleep(wait_time)
        if feedback_counter > 10:
            update_parameters_based_on_feedback()
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
