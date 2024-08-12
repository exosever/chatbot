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
        <li>Implement a TTS feature - Currently in BETA</li>
        <li>Implement STT feature for streamer</li>
        <li>Integrate Wikipedia API for more knowledgable responses</li>
        <li>Integrate web search for real-time data</li>
        <li>Add reinforcement learning to fine-tune responses</li>
        <li>Add moods</li>
        <li>Adjust databse to include full conversations</li>
        <li>Add Sentiment Analysis</li>
    </ul>
    <br>
    <br>
    <h1>Project Changelog</h1>
    <div class="version">
        <h2>Version 1.0</h2>
        <ul>
            <li><strong>Integration:</strong> Utilized GPT-Neo model from the Hugging Face library.</li>
            <li><strong>Configuration:</strong> Employed plain text API keys for authentication and service access.</li>
        </ul>
    </div>
    <div class="version">
        <h2>Version 2.0</h2>
        <ul>
            <li><strong>Model Upgrade:</strong> Transitioned to Gemini Flash 1.5 model for enhanced performance and capabilities.</li>
            <li><strong>System Instructions:</strong> Implemented system instructions and safety settings to refine response quality and ensure compliance with content policies.</li>
            <li><strong>Automated Messaging:</strong> Introduced a periodic reminder message feature, triggering after 300-600 seconds and upon receiving more than 10 messages, to engage users and encourage interaction.</li>
        </ul>
    </div>
    <div class="version">
        <h2>Version 3.0</h2>
        <ul>
            <li><strong>Configuration Management:</strong> Migrated API keys to a <code>.env</code> file for improved security and management of sensitive information.</li>
            <li><strong>System Instructions:</strong> Moved system instructions to an external readable file, facilitating easier updates and management.</li>
            <li><strong>User Interaction Memory:</strong> Implemented a JSON-based database to store and manage user interactions, allowing the AI to learn from and remember user input for enhanced personalization and context-aware responses.</li>
        </ul>
    </div>
</body>
</html>
