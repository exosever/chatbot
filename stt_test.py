import io
import wave
from google.cloud import speech_v1 as speech
import pyaudio
import numpy as np

# Configuration
CHUNK = 1024  # Number of audio frames per buffer
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1  # Number of audio channels (1 for mono, 2 for stereo)
RATE = 44100  # Sample rate (samples per second)
THRESHOLD = 500  # Audio level threshold for starting recording
RECORD_SECONDS = 5  # Duration to record once the threshold is exceeded
OUTPUT_FILE = "output.wav"

p = pyaudio.PyAudio()

# Buffer for recording
frames = []


def callback(in_data, frame_count, time_info, status):
    audio_data = np.frombuffer(in_data, dtype=np.int16)
    audio_level = np.max(np.abs(audio_data))

    if audio_level > THRESHOLD:
        # Start recording
        print("Recording...")
        frames.extend(in_data)
    else:
        # Not recording
        print("No significant audio detected.")

    return (in_data, pyaudio.paContinue)


stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=callback)

print("Stream started.")
try:
    stream.start_stream()
    while True:
        # Keep the stream open
        pass
except KeyboardInterrupt:
    print("Stream stopped.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded audio to a file
    with wave.open(OUTPUT_FILE, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    # Transcribe the recorded audio
    def transcribe_audio(filename):
        client = speech.SpeechClient()

        with io.open(filename, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code="en-US",
        )

        response = client.recognize(config=config, audio=audio)

        # Print the transcriptions
        for result in response.results:
            print("Transcript: {}".format(result.alternatives[0].transcript))

    transcribe_audio(OUTPUT_FILE)
