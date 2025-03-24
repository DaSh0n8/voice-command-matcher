import pvporcupine  
import pyaudio
import whisper
import numpy as np
import sounddevice as sd
import queue

ACCESS_KEY = "Qvzec8n0OXf26c/lJCoyxdIEUEinZW64C0MercOCf2R1p5QY5+9qOQ=="

WAKE_WORD_PATH = "Hey-Igloo_en_mac_v3_0_0/Hey-Igloo_en_mac_v3_0_0.ppn"  

model = whisper.load_model("base")

# Initialize Porcupine with custom wake word
porcupine = pvporcupine.create(
    access_key=ACCESS_KEY,  
    keyword_paths=[WAKE_WORD_PATH]  
)  

pa = pyaudio.PyAudio()

# Open an audio stream for wake word detection
audio_stream = pa.open(
    rate=porcupine.sample_rate,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=porcupine.frame_length
)

def listen_for_wake_word():
    """
    Continuously listens for the wake word using Porcupine.
    """
    print("Listening for wake word 'Hey Igloo'...")

    while True:
        pcm = np.frombuffer(audio_stream.read(porcupine.frame_length), dtype=np.int16)
        keyword_index = porcupine.process(pcm)

        if keyword_index >= 0:
            print("✅ Wake word detected! Processing command...")
            #transcribe_command()


listen_for_wake_word()
