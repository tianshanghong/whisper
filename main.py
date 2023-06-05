import sounddevice as sd
import scipy.io.wavfile as wavfile
import openai
import os
import sys
import select
import numpy as np
import requests

# Check OPENAI_API_KEY environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise ValueError("Please set your OpenAI API key in the environment variable 'OPENAI_API_KEY'")

SAMPLE_RATE = 44100  # Hertz
channels = 1
recording = []
is_recording = False

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    if is_recording:
        recording.extend(indata.tolist())

# Record audio
try:
    stream = sd.InputStream(callback=callback, channels=channels, samplerate=SAMPLE_RATE)
    with stream:
        print("Recording started. Press 'Enter' to stop recording...")
        is_recording = True
        while True:
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                line = input()
                break
except sd.PortAudioError as e:
    print(f"Error while recording: {str(e)}")
    sys.exit(1)
except KeyboardInterrupt:
    pass  # Handle Ctrl+C gracefully

is_recording = False
print("Recording stopped.")

# Save as .wav file
try:
    scaled_recording = np.int16(np.array(recording) * 32767)  # convert to int16
    wavfile.write('output.wav', SAMPLE_RATE, scaled_recording)
except (ValueError, IOError) as e:
    print(f"Error while saving the audio file: {str(e)}")
    sys.exit(1)

# Send to API
try:
    openai.api_key = OPENAI_API_KEY
    with open("output.wav", "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        original_transcription = transcript["text"]
        print(f"Original transcription: \n\n{original_transcription}\n")
except requests.exceptions.RequestException as e:
    print(f"Error while transcribing the audio file: {str(e)}")
    sys.exit(1)

# Proofread the transcription using ChatGPT
try:
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f'Please proofread and enhance the following text: "{original_transcription}"'}
        ]
    )
    enhanced_transcription = completion.choices[0].message["content"].strip("\"")
    print(f"Enhanced transcription: \n\n{enhanced_transcription}\n")
except requests.exceptions.RequestException as e:
    print(f"Error while enhancing the transcription: {str(e)}")
    sys.exit(1)

# Delete the recorded audio file
try:
    os.remove("output.wav")
    # print("Recorded audio file has been deleted.")
except OSError as e:
    print(f"Error while deleting the audio file: {str(e)}")
    sys.exit(1)
