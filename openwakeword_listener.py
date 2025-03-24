import torch
import sounddevice as sd
import numpy as np
from transformers import pipeline
import torchaudio

# Load Whisper model
device = "cpu"  # Force CPU to avoid MPS issues
whisper = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=device)

# Set your custom wake word
WAKE_WORD = "hey igloo"

def callback(indata, frames, time, status):
    if status:
        print(status)
    
    # Convert audio to NumPy array
    audio_tensor = torch.tensor(audio_data, dtype=torch.float32)

    # If audio has multiple channels, take only the first channel (convert to mono)
    if audio_tensor.ndim > 1:
        audio_tensor = audio_tensor.mean(dim=0)

    # Ensure correct sample rate
    audio_tensor = torchaudio.transforms.Resample(orig_freq=16000, new_freq=16000)(audio_tensor)

    # Convert from PyTorch tensor to NumPy array
    audio_numpy = audio_tensor.numpy()

    # Extract features for Whisper model
    input_features = whisper.feature_extractor(audio_numpy, sampling_rate=16000, return_tensors="np").input_features
    # Perform inference
    result = whisper(input_features)
    text = result['text'].lower()

    if WAKE_WORD in text:
        print("✅ Wake word detected!")
        raise sd.CallbackStop  # Stop listening after detection

# Start listening
with sd.InputStream(callback=callback, samplerate=16000, channels=1, dtype='float32'):
    print("🎤 Listening for wake word...")
    sd.sleep(1000000)
