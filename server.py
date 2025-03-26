import socket
import json
import pvporcupine  
import pyaudio
import time
import speech_recognition as sr
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import spacy
import string
import threading
import re
from spacy.matcher import Matcher
from faster_whisper import WhisperModel
from rapidfuzz import fuzz, process

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
WAKE_WORD_PATH = "Hey-Igloo_en_mac_v3_0_0/Hey-Igloo_en_mac_v3_0_0.ppn" 

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
            print("Returning to sleep mode...")

current_session_id = None
current_session_name = None
layer_dict = {}

SAMPLE_RATE = 44100  
DURATION = 4
OUTPUT_FILE = "live_audio.wav"  

def start_background_services():
    """
    Start 2 background threads - listener for layer changes and wake word detector.
    """
    layer_thread = threading.Thread(target=listen_for_all_layer_changes, daemon=True)
    
    wake_thread = threading.Thread(target=listen_for_wake_word, daemon=True)

    layer_thread.start()
    wake_thread.start()

    wake_thread.join()

def command_loop():
    """
    Continuously listens for commands for 4-second intervals.
    Exits loop if no valid command is detected.
    """
    print("Entering command mode (say a command)...")

    while True:
        speech_string = speech_to_text()

        if not speech_string.strip():
            print("No speech detected. Returning to sleep mode.")
            break

        parsed = parse_command(speech_string)
        print(parsed)

        if not parsed["layer"] and not parsed["action"]:
            print("Command not understood. Returning to sleep mode.")
            break
        
        command_to_function(parsed)
        print("Command executed. Listening for more... (or go silent to exit)")
        time.sleep(1)

def record_audio():
    """
    Records user speech and turns it into a .wav file.
    Uses default microphone, and records for a set duration.
    """   
    print("Say something:")
    
    audio_data = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()  

    wav.write(OUTPUT_FILE, SAMPLE_RATE, audio_data)

def speech_to_text(): 
    """
    Converts speech to text using specified model.
    """   
    record_audio()

    # Change model size here (tiny, base, small, medium, large, turbo)
    model = WhisperModel("small", compute_type="auto")

    segments, info = model.transcribe(OUTPUT_FILE, language="en")

    full_text = ""
    for segment in segments:
        full_text += segment.text + " "

    full_text = full_text.strip()
    print("You said:", full_text)

    return full_text.lower()

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Action patterns to be recognized 
action_patterns = [
    [{"LOWER": "hide"}], [{"LOWER": "unhide"}], [{"LOWER": "move"}], [{"LOWER": "duplicate"}],
    [{"LOWER": "remove"}], [{"LOWER": "delete"}], [{"LOWER": "start"}], [{"LOWER": "save"}], 
    [{"LOWER": "select"}], [{"LOWER": "pin"}], [{"LOWER": "unpin"}], [{"LOWER": "snap"}], 
    [{"LOWER": "lock"}], [{"LOWER": "enable"}], [{"LOWER": "disable"}], [{"LOWER": "add"}],
    [{"LOWER": "play"}], [{"LOWER": "please"}], [{"LOWER": "pause"}], [{"LOWER": "stop"}],
    [{"LOWER": "add"}], [{"LOWER": "create"}],
    
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

    # For volume
    [{"LOWER": "set"}, {"LOWER": "volume"}],
    [{"LOWER": "increase"}, {"LOWER": "volume"}],
    [{"LOWER": "decrease"}, {"LOWER": "volume"}],
    [{"LOWER": "lower"}, {"LOWER": "volume"}],
    [{"LOWER": "increase"}, {"LOWER": "the"}, {"LOWER": "volume"}],
    [{"LOWER": "decrease"}, {"LOWER": "the"}, {"LOWER": "volume"}],
    [{"LOWER": "lower"}, {"LOWER": "the"}, {"LOWER": "volume"}],
    [{"LOWER": "set"}, {"LOWER": "the"}, {"LOWER": "volume"}]
]
matcher.add("ACTION", action_patterns)

# Value patterns
value_patterns = [
    [{"LOWER": "by"}, {"LIKE_NUM": True}, {"TEXT": {"REGEX": "%|px|pixels"}}], 
    [{"LIKE_NUM": True}, {"TEXT": {"REGEX": "%|px"}}],  
    [{"LOWER": "half"}, {"LOWER": "size"}], 
    [{"LOWER": "session"}],  
    [{"LOWER": "region"}, {"LIKE_NUM": True}], 
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

    for window_size in range(max_n, 0, -1):
        for i in range(len(words) - window_size + 1):
            candidate = " ".join(words[i : i + window_size]).strip()

            if candidate in layer_dict:
                return layer_dict[candidate]

            match, score, _ = process.extractOne(
                candidate, layer_dict.keys(), scorer=fuzz.token_sort_ratio
            )
            if score >= 70:
                return layer_dict[match]

    return None

def process_numeric_value(value):
    """
    Converts extracted values like "50%" or "20px" into integers.

    value (str): Raw extracted value.
    """
    match = re.match(r"(\d+)", value) 
    return int(match.group(1)) if match else None

def parse_command(command):
    """
    Extracts the action, layer, and value from a speech command.

    command (str): User's command transcribed by Whisper.
    """
    doc = nlp(command)
    matches = matcher(doc)

    extracted = {"action": None, "layer": None, "value": None, "type": None}

    for match_id, start, end in matches:
        label = nlp.vocab.strings[match_id]  
        entity_text = doc[start:end].text.strip().lower()

        if label == "ACTION":
            extracted["action"] = entity_text
        elif label == "VALUE":
            extracted["value"] = entity_text
        elif label == "TYPE":
            extracted["type"] = entity_text
    
    if extracted["value"]:
        extracted["value"] = process_numeric_value(extracted["value"])

    best_layer_match = find_best_layer_match(command, layer_dict)
    if best_layer_match:
        extracted["layer"] = best_layer_match 

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
        ("pin", True): lambda l: set_layer_pin(l, True),
        ("unpin", True): lambda l: set_layer_pin(l, False),
        ("lock", True): lambda l: set_layer_lock(l, True),
        ("unlock", True): lambda l: set_layer_lock(l, False),
        ("enable", True): lambda l: set_layer_visibility(l, True),
        ("unhide", True): lambda l: set_layer_visibility(l, True),
        ("disable", True): lambda l: set_layer_visibility(l, False),
        ("hide", True): lambda l: set_layer_visibility(l, False),
        ("play", True): play_video,
        ("please", True): play_video,
        ("pause", True): pause_video,
        ("stop", True): stop_video
    }

    if layer == "all":
        func = action_map.get((action, True))
        if func:
            print(f"Applying '{action}' to all layers...")
            for layer_id in layer_dict.values():
                func(layer_id)
            return
        
        if action in ("set scale", "increase scale", "decrease scale", "lower scale", "set the scale", 
                      "increase the scale", "decrease the scale", "lower the scale"):
            action_base = action.replace("the ", "").strip()

            for layer_id in layer_dict.values():
                current_scale = get_layer_scale(layer_id)
                if current_scale is None:
                    print(f"Failed to get scale for layer {layer_id}. Skipping.")
                    continue

                if "set scale" in action_base and value is not None:
                    new_scale = max(value / 100, 0.01)
                else:
                    adjustment = (value / 100) * current_scale if value else 0.25 * current_scale

                    if "increase scale" in action_base:
                        new_scale = current_scale + adjustment
                    elif "decrease scale" in action_base or "lower scale" in action_base:
                        new_scale = current_scale - adjustment
                    else:
                        continue

                    new_scale = max(new_scale, 0.01)

                set_layer_scale_absolute(layer_id, new_scale)

            return

        if action in ("set volume", "increase volume", "decrease volume", "set the volume", "increase the volume", 
              "decrease the volume", "lower volume", "lower the volume"):

            action_base = action.replace("the ", "").strip()

            for layer_id in layer_dict.values():
                if "set volume" in action_base and value is not None:
                    new_volume = max(min(value / 100, 1.0), 0.0)
                    set_layer_volume(layer_id, new_volume)
                    continue

                current_volume = get_layer_volume(layer_id)
                if current_volume is None:
                    print(f"Failed to get volume for layer {layer_id}. Skipping.")
                    continue

                adjustment = (value / 100) * current_volume if value else 0.1 * current_volume

                if "increase volume" in action_base:
                    new_volume = min(current_volume + adjustment, 1.0)
                    print(new_volume)
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

    if action == "zoom" and layer:
        zoom_value = value if value else 25
        return set_layer_scale(layer, zoom_value, True)

    if action in ("shrink", "minimize") and layer:
        shrink_value = value if value else 25
        return set_layer_scale(layer, shrink_value, False)
    
    if action in ("set scale", "increase scale", "decrease scale", "lower scale",
        "set the scale", "increase the scale", "decrease the scale", "lower the scale") and layer:
        action_base = action.replace("the ", "").strip()

        current_scale = get_layer_scale(layer)
        if current_scale is None:
            print("Failed to get current scale.")
            return

        if "set scale" in action_base and value is not None:
            new_scale = max(value / 100, 0.01)
        else:
            adjustment = (value / 100) * current_scale if value else 0.25 * current_scale

            if "increase scale" in action_base:
                new_scale = current_scale + adjustment
            elif "decrease scale" in action_base or "lower scale" in action_base:
                new_scale = current_scale - adjustment
            else:
                new_scale = current_scale

            new_scale = max(new_scale, 0.01)

        if layer and layer != "all":
            CURRENT_LAYER = layer

        return set_layer_scale_absolute(layer, new_scale)

    if action in ( "set volume", "increase volume", "decrease volume", "set the volume", 
                  "increase the volume", "decrease the volume","lower volume", "lower the volume") and layer:
        print("ENTERED")
        action_base = action.replace("the ", "").strip()

        if "set volume" in action_base:
            if value is None:
                print(f"No volume value provided for layer {layer}.")
                return
            new_volume = max(min(value / 100, 1.0), 0.0)
            print(f"Setting volume to {new_volume} for layer {layer}")
            return set_layer_volume(layer, new_volume)

        current_volume = get_layer_volume(layer)
        if current_volume is None:
            print(f"Failed to get volume for layer {layer}.")
            return

        adjustment = (value / 100) * current_volume if value else 0.1 * current_volume

        if "increase volume" in action_base:
            new_volume = min(current_volume + adjustment, 1.0)
        elif "decrease volume" in action_base or "lower volume" in action_base:
            new_volume = max(current_volume - adjustment, 0.0)
        else:
            return
        
        if layer and layer != "all":
            CURRENT_LAYER = layer

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

def listen_for_all_layer_changes():
    """
    Authenticates, fetches initial layer list, subscribes to name updates,
    and continuously listens for all changes (add, remove, rename).
    Updates global `layer_dict` accordingly.
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
                        print(f"✏️ Name changed for {layer_id}: '{old_name}' → '{cleaned_name}'")

                        # Update dicts
                        if old_name in layer_dict:
                            del layer_dict[old_name]
                        id_to_name[layer_id] = cleaned_name
                        layer_dict[cleaned_name] = layer_id

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

def get_layer_scale(layer_id):
    """
    Retrieves the current scale of a layer.

    layer_id: Layer id of the layer whose scale is to be retrieved.
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_socket.settimeout(1)

    try:
        auth_command = f"apikey?value={API_KEY}"
        client_socket.sendto(auth_command.encode(), SERVER_ADDRESS)

        api_command = f"layer/geometry/scale/get?id={layer_id}"
        client_socket.sendto(api_command.encode(), SERVER_ADDRESS)
        print(f"Sent command: {api_command}")

        start_time = time.time()

        while time.time() - start_time < 5:  
            try:
                response, _ = client_socket.recvfrom(8192)
                response_str = response.decode("utf-8", errors="ignore")
                if "layer/geometry/scale/get" in response_str and "value=" in response_str:
                    lines = response_str.splitlines()
                    for line in lines:
                        if "layer/geometry/scale/get" in line and "value=" in line:
                            value_part = [part for part in line.split("+") if "value=" in part]
                            if value_part:
                                try:
                                    return float(value_part[0].replace("value=", ""))
                                except ValueError:
                                    print("Failed to convert scale value to float:", value_part[0])
                                    return None
            except socket.timeout:
                continue

        print("Scale retrieval timed out.")
        return None

    finally:
        client_socket.close()

def set_layer_scale(layer_id, value, scale):
    """
    Adjusts the scale of a layer.

    layer_id: Layer id for layer to be scaled.
    value (int): Percentage change (e.g., 50 for +50% or -50 for -50%).
    scale (bool): True -> Scale up, False -> Scale down.
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        current_scale = get_layer_scale(layer_id)
        if current_scale is None:
            print("Failed to retrieve current scale.")
            return

        adjustment = (value / 100) * current_scale
        new_scale = current_scale + adjustment if scale else current_scale - adjustment
        new_scale = max(new_scale, 0.01)

        auth_command = f"apikey?value={API_KEY}"
        client_socket.sendto(auth_command.encode(), SERVER_ADDRESS)

        api_command = f"layer/geometry/scale/set?id={layer_id}+value={new_scale}"
        client_socket.sendto(api_command.encode(), SERVER_ADDRESS)

    finally:
        client_socket.close()

def set_layer_scale_absolute(layer_id, new_scale):
    """
    Directly sets the scale of a layer to an absolute value.

    layer_id: Layer id for layer to be scaled.
    new_scale (int): Scale for layer to be set to.
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        new_scale = max(new_scale, 0.01)  
        auth_command = f"apikey?value={API_KEY}"
        client_socket.sendto(auth_command.encode(), SERVER_ADDRESS)

        api_command = f"layer/geometry/scale/set?id={layer_id}+value={new_scale}"
        client_socket.sendto(api_command.encode(), SERVER_ADDRESS)

    finally:
        client_socket.close()

def get_layer_volume(layer_id):
    """
    Gets the current volume of a layer.
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_socket.settimeout(1) 

    try:
        auth_command = f"apikey?value={API_KEY}"
        client_socket.sendto(auth_command.encode(), SERVER_ADDRESS)

        get_command = f"layer/playback/volume/get?id={layer_id}"
        start_time = time.time()

        while time.time() - start_time < 5:
            client_socket.sendto(get_command.encode(), SERVER_ADDRESS)

            try:
                response, _ = client_socket.recvfrom(8192)
                response_str = response.decode("utf-8", errors="ignore")

                if "layer/playback/volume/get" in response_str and "value=" in response_str:
                    value_part = [part for part in response_str.split("+") if "value=" in part]
                    if value_part:
                        return float(value_part[0].replace("value=", ""))
            except socket.timeout:
                continue 

    finally:
        client_socket.close()

    print("Failed to get volume within timeout.")
    return None


def set_layer_volume(layer_id, value):
    """
    Sets the volume of a layer to a given value (0-100).

    layer_id = layer_id for video layer.
    value = the adjustment value, not the final value.
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
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        auth_command = f"apikey?value={API_KEY}"
        client_socket.sendto(auth_command.encode(), SERVER_ADDRESS)

        if direction == "up":
            api_command = f"layer/moveUp/?id={layer_id}"
        else:
            api_command = f"layer/moveDown/?id={layer_id}"

        for _ in range(times):
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

        api_command = f"layer/add?args=Image+id=123123123+index=4+type={type}"
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

def position_layer(layer_id,x,y):
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
    #voice_command()
    #listen_for_all_layer_changes()    
    start_background_services()
