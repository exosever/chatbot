import requests
from twitchio.ext import commands
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG,  # Set to DEBUG to capture all log levels
                    format='%(asctime)s - %(levelname)s - %(message)s')  # Add timestamp and log level

# Directly include your Hugging Face API key and Twitch credentials
HUGGING_FACE_API_KEY = "SCRUBBED"
TWITCH_OAUTH_TOKEN = 'SCRUBBED'
TWITCH_CLIENT_ID = 'SCRUBBED'
TWITCH_CHANNEL_NAME = 'SCRUBBED'

# Set up the API URL for GPT-Neo
API_URL = "https://api-inference.huggingface.co/models/gpt2"

# Initialize the bot
bot = commands.Bot(
    token=TWITCH_OAUTH_TOKEN,
    client_id=TWITCH_CLIENT_ID,
    nick='chesterbot9000',
    prefix='!',
    initial_channels=[TWITCH_CHANNEL_NAME]
)

# Function to call the Hugging Face API
def query_gpt_neo(prompt):
    headers = {
        "Authorization": f"Bearer {HUGGING_FACE_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "inputs": prompt,
        "max_length": 100,
        "temperature": 0.3
    }
    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        response_json = response.json()
        # Extracting only the generated text
        if isinstance(response_json, list) and len(response_json) > 0:
            generated_text = response_json[0].get('generated_text', '')
            logging.info(f"GPT-Neo response: {generated_text}")
            # Optionally, remove the prompt from the response if needed
            return generated_text.replace(prompt, '').strip()
        else:
            logging.error("Invalid response format from Hugging Face API.")
            return "Sorry, I received an invalid response from the AI service."
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling Hugging Face API: {e}")
        return "Sorry, I'm having trouble connecting to the AI service right now."

@bot.event()
async def event_ready():
    print("BOT READY")
    logging.info(f'Logged in as | {bot.nick}')
    logging.info(f'Connected to channel | {TWITCH_CHANNEL_NAME}')

    # Send a message to the chat confirming the bot is online
    try:
        channel = bot.get_channel(TWITCH_CHANNEL_NAME)
        if channel:
            await channel.send(f'Hello, {TWITCH_CHANNEL_NAME}! I am now online and ready to chat!')
            logging.info('Sent online confirmation message to chat.')
        else:
            logging.error(f"Channel {TWITCH_CHANNEL_NAME} not found.")
    except Exception as e:
        logging.error(f"Error sending confirmation message: {e}")

@bot.event()
async def event_message(message):
    # Process messages that start with "chester"
    if message.content.lower().startswith("chester"):
        prompt = message.content[len("chester"):].strip()
        logging.debug(f"Processed prompt: {prompt}")
        response = query_gpt_neo(prompt)
        try:
            await message.channel.send(response)
            logging.info(f"Sent response: {response}")
        except Exception as e:
            logging.error(f"Error sending message: {e}")
    else:
        logging.debug(f"Ignoring message: {message.content}")

bot.run()
