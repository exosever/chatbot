<body>
    <h1>Project Overview</h1>
    <p>This project delivers a highly customizable Twitch AI chatbot that evolves and learns from individual user interactions. Powered by the Google Gemini 1.5 Flash model, the chatbot generates contextually aware text responses. It leverages SQLite for storing user-submitted prompts, allowing the chatbot to enhance its performance and relevance over time by referencing historical data.</p>
    <h2>Features</h2>
    <ul>
        <li><strong>Generative AI:</strong> Utilizes the Gemini 1.5 Flash model for generating context-aware text responses.</li>
        <li><strong>User-Specific Learning:</strong> The bot stores and references historical user prompts to refine responses over time.</li>
        <li><strong>Customizable Instructions:</strong> Instructions for the AI are defined in the chatbot_instructions.txt file, allowing for tailored behavior and specific responses.</li>
        <li><strong>Robust Logging:</strong> Multiple levels of logging are implemented, including DEBUG mode for tracing and troubleshooting.</li>
        <li><strong>API Key Management:</strong> API keys are securely stored in a local .env file.</li>
        <li><strong>Safety Controls:</strong> Configurable safety settings to ensure appropriate content moderation.</li>
        <li><strong>Wikipedia Integration:</strong> Accesses Wikipedia API for more knowledgeable responses.</li>
        <li><strong>Reinforcement Learning:</strong> Utilizes reinforcement learning to fine-tune responses based on interactions.</li>
        <li><strong>Emotion Detection:</strong> Detects and adjusts responses based on user sentiment analysis.</li>
        <li><strong>Mood-Based Emotional Range:</strong> Expresses a wide array of emotions based on user interactions, enhancing dynamic engagement.</li>
    </ul>
    <h2>Setup Instructions</h2>
    <ul>
        <li><strong>Twitch Account Setup:</strong> Create a Twitch account for your chatbot and obtain the Client ID and OAuth token with read/write access:
            <ul>
                <li>Client ID: Twitch Token Generator</li>
                <li>OAuth Token: Twitch Developer Console</li>
            </ul>
            Ensure these credentials are for your bot account.</li>
        <li><strong>Gemini API Key:</strong> Get an API key for the Google Gemini LLM from Google AI Studio.</li>
        <li><strong>Environment Configuration:</strong> Store your API keys and setup variables in a <code>.env</code> file. Ensure the file is correctly loaded using <code>load_dotenv()</code> in your script.</li>
        <li><strong>AI Instructions:</strong> Write the AI’s behavioral guidelines and specific response rules in the <code>chatbot_instructions.txt</code> file.</li>
        <li><strong>Launch the Bot:</strong> Run the script to start your chatbot. If you encounter any issues, set the logging level to DEBUG and check the logs for detailed error information.</li>
    </ul>
    <h2>Summary</h2>
    <p>This Twitch AI chatbot offers a powerful and flexible platform for engaging with users in a personalized way. With its ability to learn from past interactions, adapt its responses, and manage emotional states, it provides a dynamic and interactive experience for Twitch communities.</p>
    <br>
    <br>
    <h1>Upcoming Updates and Features</h1>
    <ul>
        <li>Implement a TTS feature - Currently in BETA</li>
        <li>Implement STT feature for streamer - Currently in PROGRESS</li>
        <li>Add buffer/queue for TTS responses</li>
    </ul>
    <br>
    <br>
<body>
    <h1>Change Log</h1>
    <details open>
        <summary><h2>Version 4.1</h2></summary>
        <p>---------------------------------------------------------------------------------------------------------------------------------</p>
        <h3>New Features:</h3>
        <ul>
            <li>Implemented a feedback spam filter to ensure only the user who submitted a prompt can provide feedback once per prompt.</li>
            <li>Introduced a feature flag section, allowing users to easily enable or disable specific chatbot functions.</li>
        </ul>
        <h3>Improvements:</h3>
        <ul>
            <li>Enhanced the feedback tracker to utilize a list-based approach for storing user IDs, ensuring feedback is processed correctly and efficiently.</li>
        </ul>
        <h3>Bug Fixes:</h3>
        <ul>
            <li>Fixed emotion detection by changing the model, so the bot no longer constantly detects the user as angry or in fear.</li>
            <li>Resolved an issue where the bot's emotional state could not be adjusted.</li>
        </ul>
        <h3>Code Enhancements:</h3>
        <ul>
            <li>Refactored code for improved maintainability and readability.</li>
            <li>Created an easy-to-use setup section for users unfamiliar with Python or the APIs.</li>
            <li>Updated the <code>adjust_emotional_state</code> function to handle edge cases where the emotional state could exceed predefined limits, ensuring consistent behavior.</li>
            <li>Added more logging and error checking to improve debugging and stability.</li>
        </ul>
        <h3>Dependencies Updated:</h3>
        <ul>
            <li>Removed DuckDuckGo API for websearch</li>
        </ul>
        <p>---------------------------------------------------------------------------------------------------------------------------------</p>
    </details>
    <details>
        <summary><h2>Version 4.0</h2></summary>
        <p>---------------------------------------------------------------------------------------------------------------------------------</p>
        <h3>New Features:</h3>
        <ul>
            <li>Integrated DuckDuckGo Instant Answer API for quick and relevant search results in chatbot responses.</li>
            <li>Implemented a mood-based system allowing the chatbot to exhibit a range of emotional states: Happy, Sad, Angry, Excited, Confused, Bored, Curious, Calm, Nervous, and Motivated.</li>
            <li>Developed slider functionality for gradual changes in emotional state, enabling smooth transitions based on user interactions.</li>
            <li>Integrated Wikipedia API to query keywords in user prompts to increase accuracy and depth of responses.</li>
            <li>Developed an Emotion Detection model to enhance the understanding of user prompts.</li>
        </ul>
        <h3>Improvements:</h3>
        <ul>
            <li>Enhanced emotional state management by integrating mood variables into the <code>chatbox_instructional</code> prompt for more nuanced interactions.</li>
            <li>Replaced <code>chatbot_memory.json</code> with SQLite for persistent memory storage.</li>
            <li>Optimized memory handling to prioritize current conversations over historical data for improved relevance and accuracy.</li>
        </ul>
        <h3>Bug Fixes:</h3>
        <ul>
            <li>Resolved issues with emotional state transitions for appropriate mood adjustments.</li>
        </ul>
        <h3>Code Enhancements:</h3>
        <ul>
            <li>Improved handling of mood-based responses with updated <code>chatbox_instructional</code> prompt structure.</li>
            <li>Enhanced error handling and logging for better debugging and monitoring of emotional state changes and memory interactions.</li>
        </ul>
        <h3>Dependencies Updated:</h3>
        <ul>
            <li>Integrated DuckDuckGo Instant Answer API for improved search result integration.</li>
            <li>Revised SQLite library usage to support updated database management features.</li>
        </ul>
        <p>---------------------------------------------------------------------------------------------------------------------------------</p>
    </details>
    <details>
        <summary><h2>Version 3.0</h2></summary>
        <p>---------------------------------------------------------------------------------------------------------------------------------</p>
        <h3>New Features:</h3>
        <ul>
            <li>Switched to environment variables for configuration using a <code>.env</code> file.</li>
            <li>Added support for persistent memory storage in <code>chatbot_memory.json</code> for user-specific interactions.</li>
            <li>Implemented user-specific memory in AI responses to retain context across messages.</li>
            <li>Integrated <code>dotenv</code> for managing environment variables securely.</li>
        </ul>
        <h3>Improvements:</h3>
        <ul>
            <li>Updated AI model's system instruction and safety settings for better performance and content moderation.</li>
            <li>Revised message handling to include user-specific context and handle bot commands.</li>
            <li>Improved logging to include detailed information about memory interactions and API calls.</li>
            <li>Adjusted AI response temperature for a balance between creativity and coherence.</li>
            <li>Refined automated response logic to use the updated bot name and nickname.</li>
        </ul>
        <h3>Bug Fixes:</h3>
        <ul>
            <li>Resolved issues with handling environment variables and file loading errors.</li>
            <li>Fixed problems with saving and loading persistent memory.</li>
            <li>Addressed issues with message content filtering and response accuracy.</li>
        </ul>
        <h3>Code Enhancements:</h3>
        <ul>
            <li>Added support for external configuration files and environment variables for improved security and flexibility.</li>
            <li>Introduced a more robust system for managing and utilizing persistent memory in AI interactions.</li>
            <li>Enhanced the automated response system for more engaging interactions with viewers.</li>
        </ul>
        <h3>Dependencies Updated:</h3>
        <ul>
            <li>Added <code>dotenv</code> for environment variable management.</li>
            <li>Revised dependencies related to AI model configuration and memory handling.</li>
        </ul>
        <p>---------------------------------------------------------------------------------------------------------------------------------</p>
    </details>
    <details>
        <summary><h2>Version 2.0</h2></summary>
        <p>---------------------------------------------------------------------------------------------------------------------------------</p>
        <h3>New Features:</h3>
        <ul>
            <li>Integrated Google Gemini API for advanced AI responses.</li>
            <li>Added automated response functionality to engage viewers after a set number of messages.</li>
        </ul>
        <h3>Improvements:</h3>
        <ul>
            <li>Updated logging configuration to include timestamps and log levels for better debugging.</li>
            <li>Replaced Hugging Face GPT-2 model with Google Gemini for more dynamic and creative responses.</li>
            <li>Enhanced safety settings to block harmful content categories from the Google Gemini API.</li>
            <li>Implemented automated responses that trigger after a specific number of messages.</li>
        </ul>
        <h3>Bug Fixes:</h3>
        <ul>
            <li>Fixed handling of invalid responses from Hugging Face API.</li>
            <li>Improved accuracy of AI responses by correcting message prompt handling.</li>
            <li>Resolved issues with bot message filtering and message counting.</li>
        </ul>
        <h3>Code Enhancements:</h3>
        <ul>
            <li>Added detailed logging for API interactions and message processing.</li>
            <li>Improved error handling for API request failures and message sending issues.</li>
            <li>Updated prompt processing to handle message content variations more effectively.</li>
        </ul>
        <h3>Dependencies Updated:</h3>
        <ul>
            <li>Switched from Hugging Face API to Google Gemini API for natural language generation.</li>
        </ul>
        <p>---------------------------------------------------------------------------------------------------------------------------------</p>
    </details>
</body>
