import os
import wave
import numpy as np
import pyaudio
from faster_whisper import WhisperModel

# 🎙️ Settings
CHUNK_SIZE = 1024  
FORMAT = pyaudio.paInt16
CHANNELS = 1  
RATE = 16000  

model_size = "medium.en"  # You can change this to "tiny", "base", "small", etc.
model = WhisperModel(model_size, device="cpu", compute_type="int8")

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE * 4)

def record_chunk(file_path, chunk_length=1):
    """ Records short chunks of audio and saves as .wav """
    frames = []
    for _ in range(int(RATE / CHUNK_SIZE * chunk_length)):  
        frames.append(stream.read(CHUNK_SIZE))

    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

def transcribe_chunk(file_path):
    """ Transcribes the recorded chunk using Faster-Whisper """
    segments, _ = model.transcribe(file_path)
    return " ".join(segment.text for segment in segments)

def main():
    print("🎤 Listening... Press Ctrl+C to stop.")

    accumulated_transcription = ""  # Store full conversation

    try:
        while True:
            chunk_file = "temp_chunk.wav"
            record_chunk(chunk_file, chunk_length=1)  # Record 1-second chunk
            transcription = transcribe_chunk(chunk_file)
            os.remove(chunk_file)  # Delete the chunk file after processing

            if transcription.strip():
                print("🗣", transcription)
                accumulated_transcription += transcription + " "

    except KeyboardInterrupt:
        print("⏹ Stopping... Saving transcription.")
        with open("log.txt", "w") as log_file:
            log_file.write(accumulated_transcription)

if __name__ == "__main__":
    main()
