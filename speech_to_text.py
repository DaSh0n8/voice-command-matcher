import speech_recognition as sr
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import whisper
import spacy
from spacy.matcher import Matcher


# Using Google's speech_recognition library (Online, limited)
def speech_to_text():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print("Say something:")
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        print("You said:", text)
    except:
        print("Sorry, I didn't understand.")

# Remove later, change to button on frontend 
SAMPLE_RATE = 44100  
DURATION = 4  
OUTPUT_FILE = "live_audio.wav"  

def record_audio():
    print("Say something:")
    
    audio_data = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()  

    wav.write(OUTPUT_FILE, SAMPLE_RATE, audio_data)
    #print("Saved as:", OUTPUT_FILE)

# Using lightweight model
def speech_to_text_2():    
    record_audio()

    model = whisper.load_model("tiny")

    result = model.transcribe(OUTPUT_FILE)
    
    print("You said:", result["text"])

# Using base model
def speech_to_text_3():    
    record_audio()

    model = whisper.load_model("base")

    result = model.transcribe(OUTPUT_FILE)
    
    print("You said:", result["text"])

    return result["text"]

#result = speech_to_text_3()

#speech_to_text()

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Action patterns to be recognized 
action_patterns = [
    [{"LOWER": "shrink"}], [{"LOWER": "reduce"}], [{"LOWER": "hide"}], [{"LOWER": "move"}], 
    [{"LOWER": "duplicate"}], [{"LOWER": "scale"}, {"LOWER": "down"}], [{"LOWER": "remove"}],
    [{"LOWER": "delete"}], [{"LOWER": "start"}], [{"LOWER": "save"}], [{"LOWER": "select"}],
    [{"LOWER": "pin"}]
]
matcher.add("ACTION", action_patterns)

# Direction patterns for move command 
direction_patterns = [
    [{"LOWER": "left"}], [{"LOWER": "right"}], [{"LOWER": "up"}], [{"LOWER": "down"}],
    [{"LOWER": "to"}, {"LOWER": "the"}, {"LOWER": "left"}], 
    [{"LOWER": "to"}, {"LOWER": "the"}, {"LOWER": "right"}],
]
matcher.add("DIRECTION", direction_patterns)

# Idenitfying layer (id?)
layer_patterns = [
    [{"LOWER": "layer"}, {"IS_ALPHA": True, "OP": "+"}],  
    [{"LOWER": "the"}, {"IS_ALPHA": True, "OP": "+"}, {"LOWER": "layer"}],
    [{"LOWER": "layer"}, {"IS_DIGIT": True}], 
    [{"LOWER": "the"}, {"ENT_TYPE": "ORDINAL"}, {"LOWER": "layer"}],
    [{"LOWER": "layer"}, {"TEXT": {"REGEX": "^[A-Za-z0-9 ]+$"}, "OP": "+"}],  
    [{"LOWER": "the"}, {"TEXT": {"REGEX": "^[A-Za-z0-9 ]+$"}, "OP": "+"}, {"LOWER": "layer"}]
]
matcher.add("LAYER", layer_patterns)

# Values like % or pixels in command
parameter_patterns = [
    [{"LOWER": "by"}, {"LIKE_NUM": True}, {"TEXT": {"REGEX": "%|px|pixels"}}],
    [{"LIKE_NUM": True}, {"TEXT": {"REGEX": "%|px"}}],  
    [{"LOWER": "half"}, {"LOWER": "size"}],
    [{"LOWER": "session"}]  
]
matcher.add("PARAMETER", parameter_patterns)

def parse_command(command):
    doc = nlp(command)
    matches = matcher(doc)
    
    extracted = {"action": None, "layer": None, "value": None}
    
    for match_id, start, end in matches:
        label = nlp.vocab.strings[match_id]  
        entity_text = doc[start:end].text  
        
        if label == "ACTION":
            extracted["action"] = entity_text.lower()
        elif label == "LAYER":
            extracted["layer"] = entity_text.lower()
        elif label == "PARAMETER":
            extracted["value"] = entity_text.lower()

    return extracted

def command_to_function(command):
    if command["action"] == "save":
        return 
    elif command["action"] == "select" and command["layer"]:
        print("Selecting layer" + command["layer"])
    elif command["action"] == "select" and not command["layer"]:
        # Retrieve list of layers, and then ask which one
        print("Which layer?")
    elif (command["action"] == "remove" or command["action"] == "delete") and command["layer"]:
        print("Removing layer" + command["layer"])
    elif command["action"] == "pin" and command["layer"]:
        print("Pinning layer" + command["layer"])

result = "select the layer image test"
parsed = parse_command(result)
print(parsed)
command_to_function(parsed)