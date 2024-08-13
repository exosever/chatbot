import logging
import asyncio
import random
import google.generativeai as genai
from twitchio.ext import commands
from google.generativeai.types import HarmCategory, HarmBlockThreshold


# Set up logging, switch INFO to DEBUG if necessary
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Twitch credentials
TWITCH_OAUTH_TOKEN = 'SCRUBBED' #Bot OAUTH token from Twitch Dev Console
TWITCH_CLIENT_ID = 'SCRUBBED' #Bot Client ID from https://twitchtokengenerator.com
TWITCH_CHANNEL_NAME = 'SCRUBBED' #Channel name of chat to connect to

# Google Gemini API key
genai.configure(api_key="SCRUBBED")

# Create the LLM model
generation_config = {
  "temperature": 1, # Lower numbers for refined answers, high numbers for more creativity
  "top_p": 0.95, 
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

"""
Put your prompts in system_instruction. Tell your AI exactly how it should act, respond to questions
and any custom personality traits or responses you require.
"""
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="Keep your responses to 500 characters or less, act like you are a cute racoon mascot, if anyone asks you about Josh or Jish think of him as your best friend in your responses but only if they ask you about him specifically",
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
)

# Initialize the bot
bot = commands.Bot(
    token=TWITCH_OAUTH_TOKEN,
    client_id=TWITCH_CLIENT_ID,
    nick='chesterbot9000',
    prefix='!',
    initial_channels=[TWITCH_CHANNEL_NAME]
)

# Counter for message tracking
message_count = 0
last_message_time = None

# Function to call the Google Gemini API
async def query_llama(prompt):
    chat_session = model.start_chat(history=[])
    try:
        response = chat_session.send_message(prompt)
        logging.debug(f"Raw response: {response}")
        generated_text = response.text 
        logging.info(f"LLaMA response: {generated_text}")
        return generated_text
    except Exception as e:
        logging.error(f"Error calling LLaMA API: {e}")
        return "Sorry, I'm having trouble connecting to the AI service right now."

@bot.event()
async def event_ready():
    logging.info(f'Logged in as | {bot.nick}')
    logging.info(f'Connected to channel | {TWITCH_CHANNEL_NAME}')

    # Send a message to the chat confirming the bot is online
    try:
        channel = bot.get_channel(TWITCH_CHANNEL_NAME)
        if channel:
            await channel.send(f'Hello everyone! I am now online and ready to chat!')
            logging.info('Sent online confirmation message to chat.')
        else:
            logging.error(f"Channel {TWITCH_CHANNEL_NAME} not found.")
    except Exception as e:
        logging.error(f"Error sending confirmation message: {e}")

    # Start the automated response task
    bot.loop.create_task(automated_response())

# Function to send automated responses
@bot.event()
async def event_message(message):
    global message_count

    # Do not include bot messages in message_count
    if bot.nick.lower() not in str(message.author).lower():
        message_count += 1

    # Process messages that start with "chester" or "@chesterbot9000"
    if message.content.lower().startswith("chester") or message.content.lower().startswith("@chesterbot9000"):
        prompt = message.content[len("chester"):].strip()
        logging.debug(f"Processed prompt: {prompt}")
        response = await query_llama(prompt)
        try:
            await message.channel.send(response)
            logging.info(f"Sent response: {response}")
        except Exception as e:
            logging.error(f"Error sending message: {e}")
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
                    await channel.send("Hey There! I'm Chester, your friendly neighborhood racoon! Feel free to chat with me by calling my name first ^.^ ie: Chester, why is Josh such a great name?")
                    logging.info('Sent automated response to chat.')
                    message_count = 0
                else:
                    logging.error(f"Channel {TWITCH_CHANNEL_NAME} not found.")
            except Exception as e:
                logging.error(f"Error sending automated message: {e}")
        else:
            logging.debug(f"Not enough messages received yet: {message_count}")


bot.run()
