import logging
import asyncio
import random
import json
import google.generativeai as genai
from twitchio.ext import commands
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv('chatbot_variables.env')

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Access ENV variables
TWITCH_OAUTH_TOKEN = os.getenv('TWITCH_OAUTH_TOKEN')
TWITCH_CLIENT_ID = os.getenv('TWITCH_CLIENT_ID')
TWITCH_CHANNEL_NAME = os.getenv('TWITCH_CHANNEL_NAME')
genai.configure(api_key=os.getenv('GENAI_API_KEY'))

# BOT_NAME is the twitch account, BOT_NICKNAME is what you will call the bot
BOT_NAME = 'chesterbot9000'
BOT_NICKNAME = 'Chester'

# Create the LLM model
generation_config = {
    "temperature": 0.9,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Load textfile with prompt instructions for AI
try:
    with open("chatbot_instructions.txt", "r") as instructions:
        chatbot_instructions = instructions.read().strip()
except FileNotFoundError:
    pass

# Model settings
"""
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

# Load persistent memory
try:
    with open("chatbot_memory.json", "r") as memory_file:
        chatbot_memory = json.load(memory_file)
except FileNotFoundError:
    chatbot_memory = {}

# Save the current memory to the file


def save_memory():
    with open("chatbot_memory.json", "w") as memory_file:
        json.dump(chatbot_memory, memory_file)


# Initialize the bot
bot = commands.Bot(
    token=TWITCH_OAUTH_TOKEN,
    client_id=TWITCH_CLIENT_ID,
    nick=BOT_NAME,
    prefix='!',
    initial_channels=[TWITCH_CHANNEL_NAME]
)

# Counter for message tracking
message_count = 0
last_message_time = None

# Function to call the Google Gemini API


async def query_gemini_with_memory(user_id, prompt):
    global message_count

    chat_session = model.start_chat(history=[])

    # Retrieve user-specific memory if available
    user_memory = chatbot_memory.get(user_id, {}).get('prompts', [])
    previous_data = "\n".join(user_memory)

    # Formulate the full prompt with clear distinctions
    full_prompt = (
        "Here is some previous data from the user to keep in mind:\n"
        f"{previous_data}\n\n"
        "This is the user's current prompt:\n"
        f"{prompt}"
    )

    try:
        response = chat_session.send_message(full_prompt)
        logging.debug(f"Raw response: {response}")
        generated_text = response.text.strip()
        logging.info(f"Gemini response: {generated_text}")

        # Store the prompt for future reference
        if user_id not in chatbot_memory:
            chatbot_memory[user_id] = {'prompts': []}
        chatbot_memory[user_id]['prompts'].append(prompt)

        save_memory()

        message_count = 0

        return generated_text

    except Exception as e:
        logging.error(f"Error calling Gemini API: {e}")
        return (
            "Sorry, I'm having trouble connecting to the AI service "
            "right now."
        )


@bot.event()
async def event_ready():
    logging.info(f'Logged in as | {bot.nick}')
    logging.info(f'Connected to channel | {TWITCH_CHANNEL_NAME}')

    # Send a message to the chat confirming the bot is online
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

    # Start the automated response task
    bot.loop.create_task(automated_response())

# Function to handle messages and trigger responses


@bot.event()
async def event_message(message):
    global message_count

    # Do not include bot messages in message_count
    if bot.nick.lower() not in str(message.author).lower():
        message_count += 1

    # Process messages that start with bot_nickname or "@bot_nickname"
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
            logging.info(f"Sent response: {response}")
        except Exception as e:
            logging.error(f"Error sending message: {e}")
    elif message.content.lower().startswith("!ai"):
        logging.info(f"Received !ai command from user: {message.author}")
        await message.channel.send("I am an AI")
    else:
        logging.debug(f"Ignoring message: {message.content}")

# Function to send automated responses


async def automated_response():
    global message_count

    while True:
        wait_time = random.randint(300, 600)
        await asyncio.sleep(wait_time)

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
