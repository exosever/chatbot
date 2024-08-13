<body>
    <h1>Twitch AI Chatbot</h1>
    <h2>Overview</h2>
    <p>This project involves a customizable Twitch AI chatbot that adapts and learns from chat history on a user-by-user basis. It leverages the Gemini 1.5 Flash model to generate text responses and uses JSON for storing user-submitted prompts. The stored prompts are utilized in future interactions to enhance the quality and relevance of the chatbot's responses based on learned data.</p>
    <h2>Features</h2>
    <ul>
        <li><strong>Generative AI:</strong> Utilizes the Gemini 1.5 Flash model for generating context-aware text responses.</li>
        <li><strong>User-Specific Learning:</strong> The bot stores and references historical user prompts to refine responses over time.</li>
        <li><strong>Customizable Instructions:</strong> Instructions for the AI are defined in the <code>chatbot_instructions.txt</code> file, allowing for tailored behavior and specific responses.</li>
        <li><strong>Robust Logging:</strong> Multiple levels of logging are implemented, including DEBUG mode for tracing and troubleshooting.</li>
        <li><strong>API Key Management:</strong> API keys are securely stored in a local <code>.env</code> file.</li>
        <li><strong>Safety Controls:</strong> Configurable safety settings to ensure appropriate content moderation.</li>
    </ul>
    <h2>Setup Instructions</h2>
    <ol>
        <li><strong>Twitch Account Setup:</strong> Create a Twitch account for your chatbot and obtain the Client ID and OAuth token with read/write access:
            <ul>
                <li>Client ID: <a href="https://twitchtokengenerator.com" target="_blank">Twitch Token Generator</a></li>
                <li>OAuth Token: <a href="https://dev.twitch.tv/console" target="_blank">Twitch Developer Console</a></li>
            </ul>
            Ensure these credentials are for your bot account.</li>
        <li><strong>Gemini API Key:</strong> Get an API key for the Gemini LLM from <a href="https://aistudio.google.com/app/apikey" target="_blank">Google AI Studio</a>.</li>
        <li><strong>Environment Configuration:</strong> Store your API keys in a <code>.env</code> file. Ensure the file is correctly loaded using <code>load_dotenv()</code> in your script.</li>
        <li><strong>Channel Configuration:</strong> Specify the Twitch channel where the bot will operate by setting the <code>TWITCH_CHANNEL_NAME</code> variable.</li>
        <li><strong>AI Instructions:</strong> Write the AI’s behavioral guidelines and specific response rules in the <code>chatbot_instructions.txt</code> file.</li>
        <li><strong>Launch the Bot:</strong> Run the script to start your chatbot. If you encounter any issues, set the logging level to DEBUG and check the logs for detailed error information.</li>
    </ol>
    <h2>Summary</h2>
    <p>This Twitch AI chatbot offers a powerful and flexible platform for engaging with users in a personalized way. With its ability to learn from past interactions and adapt its responses, it provides a dynamic and interactive experience for Twitch communities.</p>
    <br>
    <br>
    <h1>Upcoming Updates and Features</h1>
    <ul>
        <li>Upgrade from JSON database to SQLite</li>
        <li>Implement a TTS feature - Currently on HOLD</li>
        <li>Implement STT feature for streamer</li>
        <li>Integrate Wikipedia API for more knowledgable responses - DONE</li>
        <li>Integrate web search for real-time data - Currently in BETA</li>
        <li>Add reinforcement learning to fine-tune responses - DONE</li>
        <li>Add moods</li>
        <li>Adjust database to include full conversations - Currently in BETA</li>
        <li>Add <s>Sentiment Analysis</s> Emotion Detection - Currently in BETA</li>
        <li>Add !AI command to describe the AI and it's features - DONE</li>
    </ul>
    <br>
    <br>
<body>
    <h1>Change Log</h1>
    <br>
    <div class="version">
        <h2>Version 2.0</h2>
        <div class="section">
            <div class="section-title">New Features:</div>
            <ul>
                <li>Integrated Google Gemini API for advanced AI responses.</li>
                <li>Added automated response functionality to engage viewers after a set number of messages.</li>
            </ul>
        </div>
        <div class="section">
            <div class="section-title">Improvements:</div>
            <ul>
                <li>Updated logging configuration to include timestamps and log levels for better debugging.</li>
                <li>Replaced Hugging Face GPT-2 model with Google Gemini for more dynamic and creative responses.</li>
                <li>Enhanced safety settings to block harmful content categories from the Google Gemini API.</li>
                <li>Implemented automated responses that trigger when a certain number of messages are received.</li>
            </ul>
        </div>
        <div class="section">
            <div class="section-title">Bug Fixes:</div>
            <ul>
                <li>Fixed issue where invalid responses from Hugging Face API were not properly handled.</li>
                <li>Corrected handling of message prompts to improve accuracy of AI responses.</li>
                <li>Resolved issue with bot message filtering to ensure accurate message counting.</li>
            </ul>
        </div>
        <div class="section">
            <div class="section-title">Code Enhancements:</div>
            <ul>
                <li>Added detailed logging for API interactions and message processing.</li>
                <li>Improved error handling for API request failures and message sending issues.</li>
                <li>Updated prompt processing to handle variations in message content more effectively.</li>
            </ul>
        </div>
        <div class="section">
            <div class="section-title">Dependencies Updated:</div>
            <ul>
                <li>Switched from using the Hugging Face API to Google Gemini API for natural language generation.</li>
            </ul>
        </div>
    </div>
<div class="version">
        <h2>Version 3.0</h2>
        <div class="section">
            <div class="section-title">New Features:</div>
            <ul>
                <li>Switched to environment variables for configuration using a `.env` file.</li>
                <li>Added support for persistent memory storage in `chatbot_memory.json` for user-specific interactions.</li>
                <li>Implemented user-specific memory in AI responses to retain context across messages.</li>
                <li>Integrated `dotenv` for managing environment variables securely.</li>
            </ul>
        </div>
        <div class="section">
            <div class="section-title">Improvements:</div>
            <ul>
                <li>Updated the AI model's system instruction and safety settings for better performance and content moderation.</li>
                <li>Revised message handling to include user-specific context in responses and handle bot commands.</li>
                <li>Improved logging to include detailed information about memory interactions and API calls.</li>
                <li>Adjusted AI response temperature for a balance between creativity and coherence.</li>
                <li>Refined automated response logic to use updated bot name and nickname.</li>
            </ul>
        </div>
        <div class="section">
            <div class="section-title">Bug Fixes:</div>
            <ul>
                <li>Resolved issues with handling environment variables and file loading errors.</li>
                <li>Fixed problems with saving and loading persistent memory correctly.</li>
                <li>Addressed issues with message content filtering and response accuracy.</li>
            </ul>
        </div>
        <div class="section">
            <div class="section-title">Code Enhancements:</div>
            <ul>
                <li>Added support for external configuration files and environment variables for improved security and flexibility.</li>
                <li>Introduced a more robust system for managing and utilizing persistent memory in AI interactions.</li>
                <li>Improved the automated response system to ensure more engaging interactions with viewers.</li>
            </ul>
        </div>
        <div class="section">
            <div class="section-title">Dependencies Updated:</div>
            <ul>
                <li>Added `dotenv` for environment variable management.</li>
                <li>Revised dependencies related to AI model configuration and memory handling.</li>
            </ul>
        </div>
    </div>
    <div class="version">
        <h2>Version 4.0</h2>
<p>Coming Soon</p>
</body>
</html>
