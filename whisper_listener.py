import whisper
import numpy as np
import sounddevice as sd
import queue

WAKE_WORD = "igloo"
SAMPLE_RATE = 16000
BLOCK_SIZE = 1024  # Smaller buffer for lower latency
LISTEN_DURATION = 3  # Minimum time to collect speech before transcribing

# Load Whisper model
model = whisper.load_model("base")

audio_queue = queue.Queue()
wake_word_detected = False  # Flag to stop listening when the wake word is heard

def callback(indata, frames, time, status):
    """ Continuously records audio and pushes to queue """
    if status:
        print(f"⚠️ Audio error: {status}")
    if not wake_word_detected:  # Stop collecting if wake word is detected
        audio_queue.put(indata.copy())

def transcribe_audio():
    """ Listens for wake word and stops immediately once detected """
    global wake_word_detected  # Access the flag to stop listening

    print("Listening for wake word...")

    with sd.InputStream(callback=callback, samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK_SIZE, dtype='float32'):
        while not wake_word_detected:  # Keep listening until wake word is detected
            try:
                # Collect audio data (1-2 seconds of speech before transcribing)
                audio_frames = []
                for _ in range(int(SAMPLE_RATE / BLOCK_SIZE * LISTEN_DURATION)):  
                    audio_frames.append(audio_queue.get())

                # Convert audio to NumPy array & ensure float32 format
                audio_data = np.concatenate(audio_frames, axis=0).flatten()
                audio_data = audio_data.astype(np.float32)  

                # Transcribe with Whisper
                result = model.transcribe(audio_data, fp16=False)
                text = result["text"].strip().lower()

                print(f"Heard: {text}")

                # Check for wake word
                if WAKE_WORD in text:
                    print("✅ Wake word detected! Stopping listening...")
                    wake_word_detected = True  # Set flag to stop collecting audio
                    break  # Exit the loop immediately

            except queue.Empty:
                continue  # Avoid errors if queue is empty

    print("Ready to process the next command.")

transcribe_audio()
