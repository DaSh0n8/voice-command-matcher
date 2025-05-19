import socket
import json
import pvporcupine  
import pyaudio
import time
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import spacy
import string
import threading
import re
import whisper
from spacy.matcher import Matcher
from faster_whisper import WhisperModel
from rapidfuzz import fuzz, process
import torch
import torchaudio
import torch.nn.functional as F
import os, sys
from contextlib import redirect_stderr, redirect_stdout


# Load configuration from config.json
def load_config():
    with open("config.json", "r") as file:
        return json.load(file)

# Initialize config variables
config = load_config()
IGLOO_SERVER_IP = config["IGLOO_SERVER_IP"]
IGLOO_SERVER_PORT = config["IGLOO_SERVER_PORT"]
API_KEY = config["API_KEY"]
ACCESS_KEY = config["ACCESS_KEY"]
SERVER_ADDRESS = (IGLOO_SERVER_IP, IGLOO_SERVER_PORT)

# Path to wake word model
WAKE_WORD_PATH = "Hey-Igloo_en_windows_v3_0_0/Hey-Igloo_en_windows_v3_0_0.ppn" 

CURRENT_LAYER = None

LAST_FN = None

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
    # Retrieving session id and name
    global current_session_id, current_session_name

    sessions = get_session_list()
    if sessions:
        current_session_id, current_session_name = next(iter(sessions.items()))
    else:
        current_session_id, current_session_name = None, None

    print("Listening for wake word 'Hey Igloo'...")

    while True:
        pcm = np.frombuffer(audio_stream.read(porcupine.frame_length, exception_on_overflow=False), dtype=np.int16)
        keyword_index = porcupine.process(pcm)

        if keyword_index >= 0:
            print("Wake word detected!")
            
            command_loop() 
            # Hide processing icon
            if "processing" in layer_dict: set_layer_visibility(layer_dict["processing"], False)
            if "tooltip" in layer_dict: set_layer_visibility(layer_dict["tooltip"], False)

            print("Returning to sleep mode...")

current_session_id = None
current_session_name = None
layer_dict = {}
layer_volume_dict = {} 

with open(os.devnull, 'w') as fnull:
    with redirect_stdout(fnull), redirect_stderr(fnull):
        model = WhisperModel("medium", device="cuda", compute_type="float16")

with open(os.devnull, 'w') as fnull:
    with redirect_stdout(fnull), redirect_stderr(fnull):
        vad_model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", force_reload=False)

def start_background_services():
    """
    Start 3 background threads - listener for layer changes, layer volume changes and wake word detector.
    """
    layer_thread = threading.Thread(target=listen_for_all_layer_changes, daemon=True)
    
    wake_thread = threading.Thread(target=listen_for_wake_word, daemon=True)

    volume_thread = threading.Thread(target=listen_for_layer_volumes, daemon=True)

    layer_thread.start()
    wake_thread.start()
    time.sleep(1)

    volume_thread.start()

    system_icons = {
        "processing": r"C:\Users\igloo\Desktop\processing.jpg",
        "listening": r"C:\Users\igloo\Desktop\listening.jpg",
        "tooltip": r"C:\Users\igloo\Desktop\tooltip.png"
    }

    time.sleep(10)
    for name, path in system_icons.items():
        if name not in layer_dict:
            add_layer("image")
            time.sleep(0.5)
            id = get_layer_id("image")
            set_layer_image_path(id, path)
            rename_layer(id, name)    
            if name == "tooltip":
                set_layer_scale(id ,0.75)
                set_layer_position(id, 0.055, 0.46)
            else:
                set_layer_scale(id, 0.3)
                set_layer_position(id, 0.136, 0.15)
            set_layer_pin(id, True)
            set_layer_visibility(id, False) 
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")

def command_loop():
    """
    Continuously listens for commands.
    Exits loop if no valid command is detected.
    """
    print("Entering command mode (say a command)...")

    while True:
        total_start = time.time()
        speech_string = speech_to_text()
        if not speech_string.strip():
            print("No speech detected. Returning to sleep mode.")
            break

        t3 = time.time()
        parsed = parse_command(speech_string)
        t4 = time.time()
        print(parsed)
        print(f"[Timing] Parsing took: {t4 - t3:.2f} seconds")

        if not parsed["layer"] and not parsed["action"]:
            print("Command not understood. Returning to sleep mode.")
            break
        t5 = time.time()
        command_to_function(parsed)
        t6 = time.time()

        print(f"[Timing] Command execution took: {t6 - t5:.2f} seconds")
        print("Command executed. Listening for more... (or go silent to exit)")
        time.sleep(1)


def record_until_silence(filename="live_audio.wav", threshold=0.80, silence_duration=2.0, max_duration=15):
    """
    Records audio in chunks, using Silero VAD to detect silence.
    The first 3 seconds (initial_grace_chunks) are a grace period where silence is ignored.
    After that, if speech probability (averaged over the chunk) is below `threshold` for 
    a duration equal to `silence_duration`, recording stops.
    Also prints elapsed listening time every second.
    """
    sample_rate = 44100
    buffer_size = 1024

    # Calculate how many consecutive chunks of silence make up the silence duration
    silence_limit = int(silence_duration * sample_rate / buffer_size)
    silence_counter = 0

    # Do the same for initial grace period, int we multiply sample_rate by is the number of seconds
    initial_grace_chunks = int(2 * sample_rate / buffer_size)

    recording = []

    original_volumes = {layer_id: layer_volume_dict.get(layer_id, 0.5) for layer_id in layer_dict.values()}

    for layer in layer_dict.values():
        if original_volumes[layer] and original_volumes[layer] > 0.05:
            set_layer_volume(layer, 0.05)

    print("Listening...")

    # Display listening icon
    if "listening" in layer_dict: 
        set_layer_visibility(layer_dict["listening"], True)
        move_layer_front(layer_dict["listening"])
    if "tooltip" in layer_dict: 
        set_layer_visibility(layer_dict["tooltip"], True)  
        move_layer_front(layer_dict["tooltip"])  
    
    vad_model.eval()
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)

    start_time = time.time()
    last_printed_second = 0

    stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16')
    with stream:
        total_chunks = int(max_duration * sample_rate / buffer_size)
        for i in range(total_chunks):
            current_time = time.time()
            elapsed_time = current_time - start_time
            if int(elapsed_time) > last_printed_second:
                print(f"Listening for {int(elapsed_time)} seconds")
                last_printed_second = int(elapsed_time)
            
            audio_chunk, _ = stream.read(buffer_size)

            audio_tensor = torch.tensor(audio_chunk.astype(np.float32)).transpose(0, 1) / 32768.0
            audio_tensor = resampler(audio_tensor)
            
            min_samples = 512
            if audio_tensor.shape[1] < min_samples:
                pad_length = min_samples - audio_tensor.shape[1]
                audio_tensor = F.pad(audio_tensor, (0, pad_length))

            with torch.no_grad():
                speech_prob = vad_model(audio_tensor, 16000)
            speech_prob_value = speech_prob.mean().item()

            recording.append(audio_chunk)

            if i >= initial_grace_chunks:
                if speech_prob_value < threshold:
                    silence_counter += 1
                else:
                    silence_counter = 0

                if silence_counter >= silence_limit:
                    print("Silence detected, stopping recording.")
                    # Hide Listening icon
                    if "listening" in layer_dict: set_layer_visibility(layer_dict["listening"], False)
                    # Display Processing icon
                    if "processing" in layer_dict: 
                        set_layer_visibility(layer_dict["processing"], True)
                        move_layer_front(layer_dict["processing"])
                    break

    full_audio = np.concatenate(recording, axis=0)
    wav.write(filename, sample_rate, full_audio)

    for layer_id in list(layer_dict.values()):
        original_volume = original_volumes.get(layer_id, 0.5)
        set_layer_volume(layer_id, original_volume)

    return filename

def old_record(filename="live_audio.wav", threshold=200, silence_duration=2.0, max_duration=8):
    """
    Records audio in chunks, and calculates rms to detect silence
    """
    sample_rate = 44100
    buffer_size = 1024

    silence_limit = int(silence_duration * sample_rate / buffer_size)
    silence_counter = 0

    initial_grace_chunks = int(3 * sample_rate / buffer_size)

    recording = []
    print("Listening...")

    start_time = time.time()
    last_printed_second = 0

    stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16')
    with stream:
        total_chunks = int(max_duration * sample_rate / buffer_size)
        for i in range(total_chunks):
            current_time = time.time()
            elapsed_time = current_time - start_time
            if int(elapsed_time) > last_printed_second:
                print(f"Listening for {int(elapsed_time)} seconds")
                last_printed_second = int(elapsed_time)
            
            audio_chunk, _ = stream.read(buffer_size)
            rms = np.sqrt(np.mean(audio_chunk**2))
            
            recording.append(audio_chunk)

            if i >= initial_grace_chunks:
                if rms < threshold:
                    silence_counter += 1
                else:
                    silence_counter = 0

                if silence_counter >= silence_limit:
                    print("Silence detected, stopping recording.")
                    break

    full_audio = np.concatenate(recording, axis=0)
    wav.write(filename, sample_rate, full_audio)
    time.sleep(0.5) 

    return filename

def speech_to_text():
    filename = record_until_silence()
    t1 = time.time()

    segments, _ = model.transcribe(filename, language="en")

    full_text = " ".join(segment.text for segment in segments).strip()
    print("You said:", full_text)

    t2 = time.time()
    print(f"[Timing] Transcription took: {t2 - t1:.2f} seconds")
    return full_text.lower()

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Action patterns to be recognized 
action_patterns = [
    [{"LOWER": "hide"}], [{"LOWER": "unhide"}], [{"LOWER": "duplicate"}],
    [{"LOWER": "clone"}], [{"LOWER": "reset"}], [{"LOWER": "home"}],
    [{"LOWER": "remove"}], [{"LOWER": "delete"}], [{"LOWER": "start"}], [{"LOWER": "save"}], 
    [{"LOWER": "select"}], [{"LOWER": "pin"}], [{"LOWER": "unpin"}], [{"LOWER": "snap"}], 
    [{"LOWER": "lock"}], [{"LOWER": "enable"}], [{"LOWER": "disable"}], [{"LOWER": "add"}],
    [{"LOWER": "play"}], [{"LOWER": "please"}], [{"LOWER": "pause"}], [{"LOWER": "stop"}],
    [{"LOWER": "add"}], [{"LOWER": "create"}], [{"LOWER": "shift"}], [{"LOWER": "align"}],
    [{"LOWER": "bring"}], [{"LOWER": "put"}], [{"LOWER": "position"}], [{"LOWER": "reposition"}],
    [{"LOWER": "place"}], [{"LOWER": "front"}], [{"LOWER": "back"}], [{"LOWER": "forward"}],
    [{"LOWER": "backward"}], [{"LOWER": "backwards"}], [{"LOWER": "region"}],
    [{"LOWER": "load"}], [{"LOWER": "loathe"}], [{"LOWER": "lode"}], [{"LOWER": "lord"}],
    
    # For Scale
    [{"LOWER": "minimize"}], [{"LOWER": "shrink"}], [{"LOWER": "minimized"}],
    [{"LOWER": "zoom"}], [{"LOWER": "enlarge"}],
    [{"LOWER": "set"}, {"LOWER": "scale"}],
    [{"LOWER": "increase"}, {"LOWER": "scale"}],
    [{"LOWER": "decrease"}, {"LOWER": "scale"}],
    [{"LOWER": "increase"}, {"LOWER": "the"}, {"LOWER": "scale"}],
    [{"LOWER": "decrease"}, {"LOWER": "the"}, {"LOWER": "scale"}],
    [{"LOWER": "lower"}, {"LOWER": "the"}, {"LOWER": "scale"}],
    [{"LOWER": "set"}, {"LOWER": "the"}, {"LOWER": "scale"}],
    [{"LOWER": "set"}, {"LOWER": "size"}],
    [{"LOWER": "increase"}, {"LOWER": "size"}],
    [{"LOWER": "decrease"}, {"LOWER": "size"}],
    [{"LOWER": "increase"}, {"LOWER": "the"}, {"LOWER": "size"}],
    [{"LOWER": "decrease"}, {"LOWER": "the"}, {"LOWER": "size"}],
    [{"LOWER": "lower"}, {"LOWER": "the"}, {"LOWER": "size"}],
    [{"LOWER": "set"}, {"LOWER": "the"}, {"LOWER": "size"}],

    # For volume
    [{"LOWER": "set"}, {"LOWER": "volume"}],
    [{"LOWER": "increase"}, {"LOWER": "volume"}],
    [{"LOWER": "decrease"}, {"LOWER": "volume"}],
    [{"LOWER": "lower"}, {"LOWER": "volume"}],
    [{"LOWER": "increase"}, {"LOWER": "the"}, {"LOWER": "volume"}],
    [{"LOWER": "decrease"}, {"LOWER": "the"}, {"LOWER": "volume"}],
    [{"LOWER": "lower"}, {"LOWER": "the"}, {"LOWER": "volume"}],
    [{"LOWER": "set"}, {"LOWER": "the"}, {"LOWER": "volume"}],
    [{"LOWER": "mute"}], [{"LOWER": "unmute"}],
]
matcher.add("ACTION", action_patterns)

# Value patterns
value_patterns = [
    [{"LOWER": "by"}, {"TEXT": {"REGEX": r"^\d*\.?\d+$"}}, {"TEXT": {"REGEX": "%|px|pixels"}}],

    [{"LOWER": "zero"}], 
    
    [{"TEXT": {"REGEX": r"^\d*\.?\d+$"}}, {"TEXT": {"REGEX": "%|px|pixels"}}],
    
    [{"LOWER": "by"}, {"TEXT": {"REGEX": r"^\d*\.?\d+$"}}],
    [{"LOWER": "to"}, {"TEXT": {"REGEX": r"^\d*\.?\d+$"}}],

    [{"TEXT": {"REGEX": r"^\d*\.?\d+$"}}],  
    
    [{"LOWER": "half"}, {"LOWER": "size"}],
    [{"LOWER": "session"}],
    [{"LOWER": "region"}, {"LIKE_NUM": True}],
    [{"LOWER": "one"}], [{"LOWER": "two"}], [{"LOWER": "three"}], [{"LOWER": "four"}], [{"LOWER": "five"}],
    [{"LOWER": "six"}], [{"LOWER": "seven"}], [{"LOWER": "eight"}], [{"LOWER": "nine"}], [{"LOWER": "ten"}],
]
matcher.add("VALUE", value_patterns)

# Type patterns for adding layers
type_patterns = [
    [{"LOWER": "image"}], [{"LOWER": "video"}], [{"LOWER": "pdf"}],
    [{"LOWER": "webview"}], [{"LOWER": "youtube"}], [{"LOWER": "ndi"}],
    [{"LOWER": "spout"}], [{"LOWER": "datapath"}], [{"LOWER": "contentbank"}],
    [{"LOWER": "loopback"}], [{"LOWER": "appview"}],
]
matcher.add("TYPE", type_patterns)

# Direction patterns for moving layers
direction_patterns = [
    [{"LOWER": "top"}, {"LOWER": "left"}],
    [{"LOWER": "top"}, {"LOWER": "right"}],
    [{"LOWER": "bottom"}, {"LOWER": "left"}],
    [{"LOWER": "bottom"}, {"LOWER": "right"}],

    [{"LOWER": "left"}], [{"LOWER": "right"}], [{"LOWER": "up"}], [{"LOWER": "down"}],
    [{"LOWER": "upward"}], [{"LOWER": "downward"}],
    [{"LOWER": "bottom"}], [{"LOWER": "center"}], [{"LOWER": "middle"}], [{"LOWER": "top"}],
    [{"LOWER": "horizontal"}], [{"LOWER": "vertical"}], [{"LOWER": "x"}], [{"LOWER": "y"}],
]
matcher.add("DIRECTION", direction_patterns)

SCALE_ACTIONS = (
    "set scale", "increase scale", "decrease scale", "lower scale", "set the scale",
    "increase the scale", "decrease the scale", "lower the scale", "set size",
    "increase size", "decrease size", "lower size", "set the size", "increase the size",
    "decrease the size", "lower the size"
)

VOLUME_ACTIONS = (
    "set volume", "increase volume", "decrease volume", "set the volume", 
    "increase the volume", "decrease the volume", "lower volume", "lower the volume"
)

def clean_text(text):
    """
    Cleans text by removing punctuation (except spaces) and converting to lowercase.
    """
    return re.sub(r"[^\w\s]", "", text.lower()).strip()

def remove_file_extension(name):
    """
    Removes common file extensions.
    """
    common_exts = [".jpg", ".jpeg", ".png", ".mp4", ".mov", ".avi", ".mkv", ".pdf"]
    name_lower = name.lower()
    for ext in common_exts:
        if name_lower.endswith(ext):
            return name[: -len(ext)]
    return name

def find_best_layer_match(doc_text, layer_dict):
    """
    Uses a sliding window approach to find match starting from highest word count (window size) in dictionary,
    because if we have both 'Deloitte Tech Summary' and 'Tech Summary' in the dictionary, and we're looking 
    for 'Deloitte Tech Summary', 'Tech Summary' might be found first if the search window size is from lowest
    to highest word count.

    doc_test (str): Command spoken by user.
    layer_dict (dict): A dictionary containing layer_name as key, and layer_id as value.
    """
    if "all layers" in doc_text.lower() or "all videos" in doc_text.lower():
        return "all"
    
    words = clean_text(doc_text).split()
    max_n = max(len(layer.split()) for layer in layer_dict) if layer_dict else 1  

    best_match = None
    best_score = 0

    for window_size in range(max_n, 0, -1):
        for i in range(len(words) - window_size + 1):
            candidate = " ".join(words[i : i + window_size]).strip()

            if candidate in layer_dict:
                return layer_dict[candidate]

            match, score, _ = process.extractOne(
                candidate, layer_dict.keys(), scorer=fuzz.token_sort_ratio
            )

            if score > best_score and score >= 65:
                best_match = match
                best_score = score
                
    if best_match:
        return layer_dict[best_match]

    return None

def process_numeric_value(value):
    """
    Cleans and parses numerical values including decimals and units like %, px, etc.

    value (str): Raw extracted value.
    """
    number_dict = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10
    }

    value = value.strip().replace("%", "").replace("px", "").replace("pixels", "")

    if value in number_dict:
        value = number_dict[value]
    
    try:
        return float(value)
    except ValueError:
        return None


def normalize_percentage(value):

    """
    Converts a value to 0–1 scale.

    value (str): Raw extracted value.
    """
    if value is None:
        return None
    return value if 0 < value <= 1 else value / 100


def parse_command(command):
    """
    Extracts the action, layer, and value from a speech command.
    command (str): User's command transcribed by Whisper.
    """
    doc = nlp(command)
    matches = matcher(doc)

    extracted = {"action": None, "layer": None, "value": None, "type": None, "direction": None}

    direction_matches = []
    for match_id, start, end in matches:
        label = nlp.vocab.strings[match_id]
        if label == "DIRECTION":
            direction_matches.append((start, end))
        else:
            entity_text = doc[start:end].text.strip().lower()
            if label == "ACTION":
                extracted["action"] = entity_text
            elif label == "VALUE":
                if entity_text == 'zero':
                    extracted["value"] = '0'
                else:
                    extracted["value"] = entity_text
            elif label == "TYPE":
                extracted["type"] = entity_text

    # Finding the longest match, because the system would pick 'left' rather than 'top left'
    if direction_matches:
        direction_matches.sort(key=lambda span: span[1]-span[0], reverse=True)
        best_start, best_end = direction_matches[0]
        extracted["direction"] = doc[best_start:best_end].text.strip().lower()

    if extracted["value"] and extracted["value"] != "session":
        extracted["value"] = process_numeric_value(extracted["value"])

    best_layer_match = find_best_layer_match(command, layer_dict)
    if best_layer_match:
        extracted["layer"] = best_layer_match 

    if extracted["action"] == "load" or extracted["action"] == "loathe" or extracted["action"] == "lode":
        session_dict = get_all_sessions()
        best_session_match = find_best_session_match(command, session_dict)

        if best_session_match:
            extracted["value"] = best_session_match
        else:
            print("No matching session found to load")
            #return extracted

    return extracted

def get_layer_id(layer_name):
    """
    Retrieves the layer's id.

    layer_name (string): Name of a layer, parsed from user speech.
    """
    return layer_dict.get(layer_name, "No such layer.")

def command_to_function(command):
    """
    Determines what function to call depending on the values of each command.

    command (dict): Command in the form of dict, storing action, layer, value and type.
    """
    global CURRENT_LAYER
    action = command.get("action")
    layer = command.get("layer")
    value = command.get("value")
    direction = command.get("direction")

    # If the user doesn't specify layer, use layer from previous command
    print("layer : " , layer , "CURRENT_LAYER: " , CURRENT_LAYER)
    if not layer and CURRENT_LAYER:
        layer = CURRENT_LAYER
        print(f"No layer specified — using CURRENT_LAYER: {layer}")

    if action == "save" and value == "session":
        print("saving as:" + current_session_name)
        return save_session(current_session_name)

    if action in ("add", "create"):
        layer_type = command.get("type")

        if layer_type:
            print(f"Adding new layer of type: {layer_type}")
            return add_layer(layer_type)
        else:
            print("No layer type specified for add/create command.")
            return
    
    action_map = {
        ("select", True): select_layer,
        ("remove", True): remove_layer,
        ("delete", True): remove_layer,
        ("clone", True): clone_layer,
        ("duplicate", True): clone_layer,
        ("reset", True): reset_layer,
        ("front", True): move_layer_front,
        ("back", True): move_layer_back,
        ("forward", True): lambda l: move_layer(l, value, "up"),
        ("backward", True): lambda l: move_layer(l, value, "down"),
        ("backwards", True): lambda l: move_layer(l, value, "down"),
        ("pin", True): lambda l: set_layer_pin(l, True),
        ("unpin", True): lambda l: set_layer_pin(l, False),
        ("lock", True): lambda l: set_layer_lock(l, True),
        ("unlock", True): lambda l: set_layer_lock(l, False),
        ("enable", True): lambda l: set_layer_visibility(l, True),
        ("unhide", True): lambda l: set_layer_visibility(l, True),
        ("disable", True): lambda l: set_layer_visibility(l, False),
        ("hide", True): lambda l: set_layer_visibility(l, False),
        ("mute", True): lambda l: set_video_mute(l, 1),
        ("unmute", True): lambda l: set_video_mute(l, 0),
        ("region", True): lambda l: move_to_region(l, value),
        ("play", True): play_video,
        ("pause", True): pause_video,
        ("please", True): play_video,
        ("stop", True): stop_video,
        ("load", False): lambda: load_session(value),
        ("loathe", False): lambda: load_session(value),
        ("lode", False): lambda: load_session(value),
        ("lord", False): lambda: load_session(value),
        ("home", False): lambda: remove_unpinned()
    }

    if layer == "all":
        func = action_map.get((action, True))
        if func:
            print(f"Applying '{action}' to all layers...")
            for layer_id in layer_dict.values():
                func(layer_id)
            return
        
        if action in (SCALE_ACTIONS):

            action_base = action.replace("the ", "").strip()

            for layer_id in layer_dict.values():
                if ("set scale" in action_base or "set size" in action_base) and value is not None:
                    new_scale = max(min(normalize_percentage(value), 1.0), 0.0)

                    set_layer_scale(layer_id, new_scale)
                    continue

                current_scale = get_layer_scale(layer_id)
                if current_scale is None:
                    print(f"Failed to get scale for layer {layer_id}. Skipping.")
                    continue

                if value is None:
                    adjustment = 0.1 * current_scale
                elif 0 < value <= 1:
                    adjustment = value * current_scale
                else:
                    adjustment = (value / 100) * current_scale

                if "increase scale" in action_base or "increase size" in action_base:
                    new_scale = min(current_scale + adjustment, 1.0)
                elif "decrease scale" in action_base or "lower scale" in action_base or "decrease size" in action_base:
                    new_scale = max(current_scale - adjustment, 0.0)
                else:
                    continue

                set_layer_scale(layer_id, new_scale)

            return

        if action in (VOLUME_ACTIONS):

            action_base = action.replace("the ", "").strip()

            for layer_id in layer_dict.values():
                if "set volume" in action_base and value is not None:
                    new_volume = max(min(normalize_percentage(value), 1.0), 0.0)

                    set_layer_volume(layer_id, new_volume)
                    continue

                current_volume = get_layer_volume(layer_id)
                if current_volume is None:
                    print(f"Failed to get volume for layer {layer_id}. Skipping.")
                    continue

                if value is None:
                    adjustment = 0.1 * current_volume
                elif 0 < value <= 1:
                    adjustment = value * current_volume
                else:
                    adjustment = (value / 100) * current_volume


                if "increase volume" in action_base:
                    new_volume = min(current_volume + adjustment, 1.0)
                elif "decrease volume" in action_base or "lower volume" in action_base:
                    new_volume = max(current_volume - adjustment, 0.0)
                else:
                    continue

                set_layer_volume(layer_id, new_volume)

            return

    func = action_map.get((action, True))
    if func and layer:
        if layer != "all":
            CURRENT_LAYER = layer
        return func(layer)
    
    func_layerless = action_map.get((action, False))
    if func_layerless:
        return func_layerless()

    if action == "zoom" and layer:
        zoom_value = value if value else 25
        return set_layer_scale(layer, zoom_value, True)

    if action in ("shrink", "minimize") and layer:
        shrink_value = value if value else 25
        return set_layer_scale(layer, shrink_value, False)
    
    if action in (SCALE_ACTIONS) and layer:
        action_base = action.replace("the ", "").strip()

        if layer and layer != "all":
            CURRENT_LAYER = layer

        if "set scale" in action_base or "set size" in action_base:
            if value is None:
                print(f"No scale value provided for layer {layer}.")
                return
            new_scale = max(min(normalize_percentage(value), 1.0), 0.0)

            print(f"Setting scale to {new_scale} for layer {layer}")
            return set_layer_scale(layer, new_scale)

        current_scale = get_layer_scale(layer)
        if current_scale is None:
            print(f"Failed to get scale for layer {layer}.")
            return

        if value is None:
            adjustment = 0.1 * current_scale
        elif 0 < value <= 1:
            adjustment = value * current_scale
        else:
            adjustment = (value / 100) * current_scale

        if "increase scale" in action_base or "increase size" in action_base:
            new_scale = min(current_scale + adjustment, 1.0)
        elif "decrease scale" in action_base or "lower scale" in action_base or "decrease size" in action_base:
            new_scale = max(current_scale - adjustment, 0.0)
        else:
            return
        
        return set_layer_scale(layer, new_scale)

    if action in (VOLUME_ACTIONS) and layer:
        action_base = action.replace("the ", "").strip()

        if layer and layer != "all":
            CURRENT_LAYER = layer
        
        if "set volume" in action_base:
            if value is None:
                print(f"No volume value provided for layer {layer}.")
                return
            new_volume = max(min(normalize_percentage(value), 1.0), 0.0)

            print(f"Setting volume to {new_volume} for layer {layer}")
            return set_layer_volume(layer, new_volume)

        current_volume = get_layer_volume(layer)
        if current_volume is None:
            print(f"Failed to get volume for layer {layer}.")
            return

        if value is None:
            adjustment = 0.1 * current_volume
        elif 0 < value <= 1:
            adjustment = value * current_volume
        else:
            adjustment = (value / 100) * current_volume

        if "increase volume" in action_base:
            new_volume = min(current_volume + adjustment, 1.0)
        elif "decrease volume" in action_base or "lower volume" in action_base:
            new_volume = max(current_volume - adjustment, 0.0)
        else:
            return
        
        return set_layer_volume(layer, new_volume)

    print("Don't understand command")


def get_session_list():
    """
    Retrieves all sessions' ID and name, stores them in session_data.
    """
    global session_data  
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_socket.settimeout(2)

    try:
        auth_command = f"apikey?value={API_KEY}"
        client_socket.sendto(auth_command.encode(), SERVER_ADDRESS)

        session_list_command = "sessionList/get"
        client_socket.sendto(session_list_command.encode(), SERVER_ADDRESS)
        print(f"Sent command: {session_list_command}")

        responses = []
        start_time = time.time()

        while time.time() - start_time < 5:
            try:
                response, _ = client_socket.recvfrom(4096)
                response_str = response.decode("utf-8", errors="ignore")
                print(f"Received Response: {response_str}")
                responses.append(response_str)

                if "sessionList/get" in response_str:
                    break

            except socket.timeout:
                print("No additional response, retrying...")
                client_socket.sendto(session_list_command.encode(), SERVER_ADDRESS)

        if "session_data" not in globals():
            session_data = {}

        # Store session list
        session_data.clear()
        for response_str in responses:
            if "sessionList/get" in response_str:
                parts = response_str.split("id=")[1:]  
                for part in parts:
                    session_info = part.strip().split("+")  
                    session_id = session_info[0]  

                    # Extract everything after "name=" as the session name
                    session_name = "Unnamed Session"
                    name_index = next((i for i, val in enumerate(session_info) if val.startswith("name=")), None)
                    if name_index is not None:
                        session_name = session_info[name_index].replace("name=", "")  # Remove "name="
                        session_name += " " + " ".join(session_info[name_index + 1:])  # Append the rest

                    session_data[session_id] = session_name.strip()  
                break  

        return session_data

    except socket.timeout:
        print("No response received from ICE. Check if sessions exist.")
        return {}

    finally:
        client_socket.close()

def get_all_sessions():
    """
    Retrieve all sessions, returns dict of names to ID
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_socket.settimeout(2)

    try:
        auth_command = f"apikey?value={API_KEY}"
        client_socket.sendto(auth_command.encode(), SERVER_ADDRESS)

        api_command = "content/sessionList/get"
        client_socket.sendto(api_command.encode(), SERVER_ADDRESS)
        print(f"Sent command: {api_command}")

        start_time = time.time()
        responses = []

        while time.time() - start_time < 5:
            try:
                response, _ = client_socket.recvfrom(65507)
                response_str = response.decode("utf-8", errors="ignore")

                if "content/sessionList/get?" in response_str:
                    responses.append(response_str)
                    break

            except socket.timeout:
                print("No additional response, retrying...")
                client_socket.sendto(api_command.encode(), SERVER_ADDRESS)

        session_dict = {}
        for response in responses:
            parts = response.split("session={")[1:]
            for part in parts:
                session_id = None
                session_name = "Unnamed Session"
                fields = part.strip("}").split("+")
                for field in fields:
                    if field.startswith("id="):
                        session_id = field.replace("id=", "").strip()
                    elif field.startswith("name="):
                        session_name = field.replace("name=", "").strip()
                if session_id:
                    session_dict[session_name] = session_id

        return session_dict
    
    except socket.timeout:
        print("No response received from ICE.")
        return {}
    
    finally:
        client_socket.close()

def find_best_session_match(doc_text, session_dict):
    """
    Uses a sliding window approach to find match starting from highest word count (window size) in dictionary.

    doc_test (str): Command spoken by user.
    session_dict (dict): A dictionary containing session_name as key, and session_id as value.
    """
    words = clean_text(doc_text).split()

    session_dict_cleaned = {
        clean_text(name): session_id for name, session_id in session_dict.items()
    }

    max_n = max(len(name.split()) for name in session_dict_cleaned) if session_dict_cleaned else 1

    best_match = None
    best_score = 0

    for window_size in range(max_n, 0, -1):
        for i in range(len(words) - window_size + 1):
            candidate = " ".join(words[i : i + window_size]).strip()

            if candidate in session_dict_cleaned:
                return session_dict_cleaned[candidate]
            
            match, score, _ = process.extractOne(
                candidate, session_dict_cleaned.keys(), scorer = fuzz.token_sort_ratio
            )

            if score > best_score and score >= 70:
                best_match = match
                best_score = score

    if best_match:
        return session_dict_cleaned[best_match]
    
    return None


def listen_for_all_layer_changes():
    """
    Authenticates, fetches initial layer list, subscribes to name updates,
    and continuously listens for all changes (add, remove, rename).
    Updates global `layer_dict` when an update is detected.
    """
    global layer_dict

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_socket.settimeout(5)

    try:
        client_socket.sendto(f"apikey?value={API_KEY}".encode(), SERVER_ADDRESS)

        client_socket.sendto("layerList/subscribe".encode(), SERVER_ADDRESS)
        print("Subscribed to layer list...")

        client_socket.sendto("layerList/get".encode(), SERVER_ADDRESS)

        known_layer_ids = set()
        id_to_name = {} 
        subscribed_to_name_ids = set()

        while True:
            try:
                response, _ = client_socket.recvfrom(8192)
                response_str = response.decode("utf-8", errors="ignore")

                # Handle layerList updates
                if "layerList/get" in response_str:
                    parts = response_str.split("id=")[1:]
                    current_layer_ids = set()

                    for part in parts:
                        layer_info = part.strip().split("+")
                        layer_id = layer_info[0]

                        # Get name and type
                        name_value = [x.replace("name=", "") for x in layer_info if x.startswith("name=")]
                        type_value = [x.replace("type=", "") for x in layer_info if x.startswith("type=")]

                        raw_name = name_value[0] if name_value else "Unnamed Layer"
                        layer_type = type_value[0] if type_value else "Unknown"

                        # If name is defaulted to "Layer1", use type instead
                        if raw_name == "Layer1":
                            raw_name = layer_type[0] + layer_type[1:].lower()

                        cleaned_name = clean_text(remove_file_extension(raw_name))

                        current_layer_ids.add(layer_id)

                        # Update global dict
                        old_name = id_to_name.get(layer_id)
                        if old_name and old_name in layer_dict:
                            del layer_dict[old_name]

                        id_to_name[layer_id] = cleaned_name
                        layer_dict[cleaned_name] = layer_id

                        if layer_id not in subscribed_to_name_ids:
                            sub_cmd = f"layer/general/name/subscribe?id={layer_id}"
                            client_socket.sendto(sub_cmd.encode(), SERVER_ADDRESS)
                            subscribed_to_name_ids.add(layer_id)
                            print(f"Subscribed to name changes for {layer_id}")

                    # Detect added/removed layers
                    added = current_layer_ids - known_layer_ids
                    removed = known_layer_ids - current_layer_ids

                    if added:
                        print("Layers Added:")
                        for layer_id in added:
                            name = id_to_name[layer_id]
                            print(f" - ID: {layer_id}, Name: {name}")

                    if removed:
                        print("Layers Removed:")
                        for layer_id in removed:
                            name = id_to_name.get(layer_id, "Unknown")
                            print(f" - ID: {layer_id}, Name: {name}")
                            if name in layer_dict:
                                del layer_dict[name]
                            if layer_id in id_to_name:
                                del id_to_name[layer_id]

                    known_layer_ids = current_layer_ids

                # Handle name changes
                elif "layer/general/name/get" in response_str:
                    parts = response_str.split("+")
                    layer_id = parts[0].split("id=")[-1]
                    name_part = [p for p in parts if p.startswith("value=")]
                    new_name = name_part[0].replace("value=", "") if name_part else "Unnamed"
                    cleaned_name = clean_text(new_name)

                    old_name = id_to_name.get(layer_id)

                    if old_name != cleaned_name:
                        print(f"Name changed for {layer_id}: '{old_name}' → '{cleaned_name}'")

                        # Update dicts
                        if old_name in layer_dict:
                            del layer_dict[old_name]
                        id_to_name[layer_id] = cleaned_name
                        layer_dict[cleaned_name] = layer_id
            except socket.timeout:
                continue

    finally:
        client_socket.close()

def listen_for_layer_volumes():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_socket.settimeout(3)

    client_socket.sendto(f"apikey?value={API_KEY}".encode(), SERVER_ADDRESS)

    try:
        for layer_name, layer_id in layer_dict.items():
            try:
                vol = get_layer_volume(layer_id)
                layer_volume_dict[layer_id] = vol

                sub_cmd = f"layer/playback/volume/subscribe?id={layer_id}"
                client_socket.sendto(sub_cmd.encode(), SERVER_ADDRESS)
                print(f"Subscribed to volume changes for {layer_id}")
            except Exception as e:
                print(f"[Volume] Skipping layer {layer_id} — doesn't support volume")

        # Continuously listen for volume updates
        while True:
            try:
                response, _ = client_socket.recvfrom(4096)
                response_str = response.decode("utf-8", errors="ignore")
                
                if "layer/playback/volume/get" in response_str:
                    parts = response_str.split("+")
                    layer_id = parts[0].split("id=")[-1]
                    value_part = [p for p in parts if p.startswith("value=")]
                    volume = float(value_part[0].replace("value=", "")) if value_part else None

                    if volume is not None:
                        layer_volume_dict[layer_id] = volume

            except socket.timeout:
                continue

    finally:
        client_socket.close()

def select_layer(layer_id):
    """
    Selects a layer.

    layer_id: Layer id for layer to be selected.
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        auth_command = f"apikey?value={API_KEY}"
        client_socket.sendto(auth_command.encode(), SERVER_ADDRESS)

        api_command = f"layer/select?id={layer_id}"
        client_socket.sendto(api_command.encode(), SERVER_ADDRESS)
        print(f"Sent command: {api_command}")

    finally:
        client_socket.close()

def set_layer_image_path(layer_id, path):
    """
    Helper function for setting image path for icons and tooltip
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        auth_command = f"apikey?value={API_KEY}"
        client_socket.sendto(auth_command.encode(), SERVER_ADDRESS)

        api_command = f"layer/contextual/filepath/set?id={layer_id}+path={path}"
        client_socket.sendto(api_command.encode(), SERVER_ADDRESS)
        print(f"Sent command: {api_command}")
    finally:
        client_socket.close()

def rename_layer(layer_id, new_name):
    """
    Helper function for renaming icon/tooltip layer
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        auth_command = f"apikey?value={API_KEY}"
        client_socket.sendto(auth_command.encode(), SERVER_ADDRESS)

        api_command = f"layer/general/name/set?id={layer_id}+value={new_name}"
        client_socket.sendto(api_command.encode(), SERVER_ADDRESS)
        print(f"Sent command: {api_command}")
    finally:
        client_socket.close()


def load_session(session_id):
    """
    Loads a session.

    session_id: ID of session to be loaded
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        auth_command = f"apikey?value={API_KEY}"
        client_socket.sendto(auth_command.encode(), SERVER_ADDRESS)

        api_command = f"session/load?id={session_id}"
        client_socket.sendto(api_command.encode(), SERVER_ADDRESS)
        print(f"Sent command: {api_command}")

    finally:
        client_socket.close()

def remove_layer(layer_id):
    """
    Removes a layer.

    layer_id: Layer id for layer to be removed.
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        auth_command = f"apikey?value={API_KEY}"
        client_socket.sendto(auth_command.encode(), SERVER_ADDRESS)

        api_command = f"layer/remove?id={layer_id}"
        client_socket.sendto(api_command.encode(), SERVER_ADDRESS)
        print(f"Sent command: {api_command}")

    finally:
        client_socket.close()

def remove_unpinned():
    """
    Removes all unpinned layers.
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        auth_command = f"apikey?value={API_KEY}"
        client_socket.sendto(auth_command.encode(), SERVER_ADDRESS)

        api_command = f"layerList/removeUnpinned"
        client_socket.sendto(api_command.encode(), SERVER_ADDRESS)
        print(f"Sent command: {api_command}")

    finally:
        client_socket.close()

def clone_layer(layer_id):
    """
    Clones a layer.

    layer_id: Layer id for layer to be clone.
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        auth_command = f"apikey?value={API_KEY}"
        client_socket.sendto(auth_command.encode(), SERVER_ADDRESS)

        api_command = f"layer/clone?id={layer_id}"
        client_socket.sendto(api_command.encode(), SERVER_ADDRESS)
        print(f"Sent command: {api_command}")

    finally:
        client_socket.close()

def reset_layer(layer_id):
    """
    Resets a layer's geometry.

    layer_id: Layer id for layer to be reset.
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        auth_command = f"apikey?value={API_KEY}"
        client_socket.sendto(auth_command.encode(), SERVER_ADDRESS)

        api_command = f"layer/geometry/reset?id={layer_id}"
        client_socket.sendto(api_command.encode(), SERVER_ADDRESS)
        print(f"Sent command: {api_command}")

    finally:
        client_socket.close()

def set_layer_pin(layer_id, pin):
    """
    Pins a layer.

    layer_id: Layer id for layer to be pinned.
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        auth_command = f"apikey?value={API_KEY}"
        client_socket.sendto(auth_command.encode(), SERVER_ADDRESS)

        if pin:
            api_command = f"layer/general/pin/set?id={layer_id}+value=1"
        else:
            api_command = f"layer/general/pin/set?id={layer_id}+value=0"

        client_socket.sendto(api_command.encode(), SERVER_ADDRESS)
        print(f"Sent command: {api_command}")

    finally:
        client_socket.close()

def set_layer_visibility(layer_id, enable):
    """
    Enables or disables a layer.

    layer_id: Layer id for layer to be enabled/disabled.
    enable (bool): True -> enable/unhide, False -> disable/hide.
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        auth_command = f"apikey?value={API_KEY}"
        client_socket.sendto(auth_command.encode(), SERVER_ADDRESS)

        if enable:
            api_command = f"layer/general/enabled/set?id={layer_id}+value=1"
        else:
            api_command = f"layer/general/enabled/set?id={layer_id}+value=0"

        client_socket.sendto(api_command.encode(), SERVER_ADDRESS)
        print(f"Sent command: {api_command}")

    finally:
        client_socket.close()

def layer_position(layer_id, position_mode):
    """
    Changes the position mode for a layer.

    layer_id: Layer id for layer to be moved.
    position_mode (int): 0 - Lock to Container, 1 - Lock to Region, 2 - Free.
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        auth_command = f"apikey?value={API_KEY}"
        client_socket.sendto(auth_command.encode(), SERVER_ADDRESS)

        api_command = f"layer/geometry/positionMode/set?id={layer_id}+value={position_mode}"
        client_socket.sendto(api_command.encode(), SERVER_ADDRESS)
        print(f"Sent command: {api_command}")

    finally:
        client_socket.close()

def move_layer_front(layer_id):
    """
    Moves a layer to the front

    layer_id: Layer id for layer to be moved.
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        auth_command = f"apikey?value={API_KEY}"
        client_socket.sendto(auth_command.encode(), SERVER_ADDRESS)

        api_command = f"layer/toFront?id={layer_id}"
        client_socket.sendto(api_command.encode(), SERVER_ADDRESS)
        print(f"Sent command: {api_command}")

    finally:
        client_socket.close()

def move_layer_back(layer_id):
    """
    Moves a layer to the back

    layer_id: Layer id for layer to be moved.
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        auth_command = f"apikey?value={API_KEY}"
        client_socket.sendto(auth_command.encode(), SERVER_ADDRESS)

        api_command = f"layer/toBack?id={layer_id}"
        client_socket.sendto(api_command.encode(), SERVER_ADDRESS)
        print(f"Sent command: {api_command}")

    finally:
        client_socket.close()

def move_to_region(layer_id, region_index):
    """
    Moves a layer to a specific region

    layer_id: Layer id for layer to be moved.
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        if region_index:
            auth_command = f"apikey?value={API_KEY}"
            client_socket.sendto(auth_command.encode(), SERVER_ADDRESS)

            api_command = f"layer/geometry/moveToRegion?id={layer_id}+layoutIndex=0+regionIndex={region_index}"
            client_socket.sendto(api_command.encode(), SERVER_ADDRESS)
            print(f"Sent command: {api_command}")

    finally:
        client_socket.close()

def set_layer_lock(layer_id, lock):
    """
    Locks or unlocks a layer.

    layer_id: Layer id for layer to be locked/unlocked.
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        auth_command = f"apikey?value={API_KEY}"
        client_socket.sendto(auth_command.encode(), SERVER_ADDRESS)

        if lock:
            api_command = f"layer/general/alwaysOnTop/set?id={layer_id}+value=1"
        else:
            api_command = f"layer/general/alwaysOnTop/set?id={layer_id}+value=0"
        client_socket.sendto(api_command.encode(), SERVER_ADDRESS)
        print(f"Sent command: {api_command}")

    finally:
        client_socket.close()

def set_layer_scale(layer_id, value):
    """
    Sets the scale of a layer to a given value (0-1).

    layer_id: layer_id for video layer.
    value (float): final volume for scale.
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        auth_command = f"apikey?value={API_KEY}"
        client_socket.sendto(auth_command.encode(), SERVER_ADDRESS)

        api_command = f"layer/geometry/scale/set?id={layer_id}+value={value}"
        client_socket.sendto(api_command.encode(), SERVER_ADDRESS)

    finally:
        client_socket.close()

def get_layer_scale(layer_id):
    return get_layer_property(layer_id, "layer/geometry/scale/get", "layer/geometry/scale/get")

def get_layer_volume(layer_id):
    return get_layer_property(layer_id, "layer/playback/volume/get", "layer/playback/volume/get")

def get_layer_property(layer_id, endpoint, response_prefix, timeout=1):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_socket.settimeout(1)

    try:
        auth_command = f"apikey?value={API_KEY}"
        client_socket.sendto(auth_command.encode(), SERVER_ADDRESS)

        api_command = f"{endpoint}?id={layer_id}"
        start_time = time.time()

        while time.time() - start_time < timeout:
            client_socket.sendto(api_command.encode(), SERVER_ADDRESS)
            try:
                response, _ = client_socket.recvfrom(8192)
                response_str = response.decode("utf-8", errors="ignore")

                if response_prefix in response_str and "value=" in response_str:
                    for line in response_str.splitlines():
                        if response_prefix in line and "value=" in line:
                            value_part = [part for part in line.split("+") if "value=" in part]
                            if value_part:
                                try:
                                    return float(value_part[0].replace("value=", ""))
                                except ValueError:
                                    print(f"Failed to convert value: {value_part[0]}")
                                    return None
            except socket.timeout:
                continue
    finally:
        client_socket.close()

    print(f"Failed to retrieve value for {endpoint} within timeout.")
    return None


def set_layer_volume(layer_id, value):
    """
    Sets the volume of a layer to a given value (0-1).

    layer_id: layer_id for video layer.
    value(float): final value for volume.
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        auth_command = f"apikey?value={API_KEY}"
        client_socket.sendto(auth_command.encode(), SERVER_ADDRESS)

        api_command = f"layer/playback/volume/set?id={layer_id}+value={value}"
        client_socket.sendto(api_command.encode(), SERVER_ADDRESS)

    finally:
        client_socket.close()

def move_layer(layer_id, times, direction):
    """
    Moves a layer up or down, repeat for a specified number of times.

    layer_id: Layer id for layer to be moved.
    times (int): How many times the layer should be moved.
    direction (string): Layer moving up or down.
    """
    if not times:
        times = 1
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        auth_command = f"apikey?value={API_KEY}"
        client_socket.sendto(auth_command.encode(), SERVER_ADDRESS)

        if direction == "up":
            api_command = f"layer/moveUp/?id={layer_id}"
        else:
            api_command = f"layer/moveDown/?id={layer_id}"

        for _ in range(int(times)):
            client_socket.sendto(api_command.encode(), SERVER_ADDRESS)
        print(f"Sent command: {api_command}")

    finally:
        client_socket.close()


def add_layer(type):
    """
    Adds a new layer.

    type (string): Layer type (IMAGE, VIDEO etc.)
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        auth_command = f"apikey?value={API_KEY}"
        client_socket.sendto(auth_command.encode(), SERVER_ADDRESS)

        api_command = f"layer/add?args=Image+type={type}"
        client_socket.sendto(api_command.encode(), SERVER_ADDRESS)
        print(f"Sent command: {api_command}")

    finally:
        client_socket.close()

def save_session(name):
    """
    Saves current session.

    layer_id: Layer id for layer to be locked.
    name (string): Name of session.
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        auth_command = f"apikey?value={API_KEY}"
        client_socket.sendto(auth_command.encode(), SERVER_ADDRESS)

        api_command = f"session/saveAs?name={name}"
        client_socket.sendto(api_command.encode(), SERVER_ADDRESS)
        print(f"Sent command: {api_command}")

    finally:
        client_socket.close()

def set_layer_position(layer_id,x,y):
    """
    Changes the position of a layer

    layer_id: Layer id for layer to be moved
    x (int)
    y (int)
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        auth_command = f"apikey?value={API_KEY}"
        client_socket.sendto(auth_command.encode(), SERVER_ADDRESS)

        api_command = f"layer/geometry/position/set?id={layer_id}+x={x}+y={y}"
        client_socket.sendto(api_command.encode(), SERVER_ADDRESS)
        print(f"Sent command: {api_command}")

    finally:
        client_socket.close()

def play_video(layer_id):
    """
    Plays video.

    layer_id: Layer id for video/ Youtube layer.
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        auth_command = f"apikey?value={API_KEY}"
        client_socket.sendto(auth_command.encode(), SERVER_ADDRESS)

        api_command = f"layer/playback/play?id={layer_id}"
        client_socket.sendto(api_command.encode(), SERVER_ADDRESS)
        print(f"Sent command: {api_command}")

    finally:
        client_socket.close()

def pause_video(layer_id):
    """
    Pauses video.

    layer_id: Layer id for video/ Youtube layer.
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        auth_command = f"apikey?value={API_KEY}"
        client_socket.sendto(auth_command.encode(), SERVER_ADDRESS)

        api_command = f"layer/playback/pause?id={layer_id}"
        client_socket.sendto(api_command.encode(), SERVER_ADDRESS)
        print(f"Sent command: {api_command}")

    finally:
        client_socket.close()

def stop_video(layer_id):
    """
    Pauses video and resets duration to 0.

    layer_id: Layer id for video/ Youtube layer.
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        auth_command = f"apikey?value={API_KEY}"
        client_socket.sendto(auth_command.encode(), SERVER_ADDRESS)

        api_command = f"layer/playback/stop?id={layer_id}"
        client_socket.sendto(api_command.encode(), SERVER_ADDRESS)
        print(f"Sent command: {api_command}")

    finally:
        client_socket.close()

def set_video_mute(layer_id, value):
    """
    Mutes or unmutes a video layer.

    layer_id: Layer id for video/ Youtube layer.
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        auth_command = f"apikey?value={API_KEY}"
        client_socket.sendto(auth_command.encode(), SERVER_ADDRESS)

        api_command = f"layer/playback/mute/set?id={layer_id}+value={value}"
        client_socket.sendto(api_command.encode(), SERVER_ADDRESS)
        print(f"Sent command: {api_command}")

    finally:
        client_socket.close()
        
def get_regions():
    """
    Retrieves and prints the list of available regions/layouts.
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_socket.settimeout(2)  

    try:
        auth_command = f"apikey?value={API_KEY}"
        client_socket.sendto(auth_command.encode(), SERVER_ADDRESS)

        layout_list_command = "app/layout/get?index=0"
        client_socket.sendto(layout_list_command.encode(), SERVER_ADDRESS)
        print(f"Sent command: {layout_list_command}")

        start_time = time.time()
        while time.time() - start_time < 5: 
            try:
                response, _ = client_socket.recvfrom(4096)  
                response_str = response.decode("utf-8", errors="ignore")
                print(f"Raw Response: {response_str}")

                if "app/layout/get?index=0" in response_str:
                    break 

            except socket.timeout:
                print("Still waiting for response...")
                client_socket.sendto(layout_list_command.encode(), SERVER_ADDRESS) 

    finally:
        client_socket.close()

if __name__ == "__main__":
    #listen_for_wake_word()
    #listen_for_all_layer_changes()    
    # print(sd.query_devices())
    
    start_background_services()
