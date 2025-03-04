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



result = speech_to_text_3()

#speech_to_text()

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

action_patterns = [
    [{"LOWER": "shrink"}], [{"LOWER": "reduce"}], [{"LOWER": "hide"}], 
    [{"LOWER": "move"}], [{"LOWER": "duplicate"}], [{"LOWER": "scale"}, {"LOWER": "down"}],
    [{"LOWER": "delete"}], [{"LOWER": "start"}], [{"LOWER": "save"}]
]
matcher.add("ACTION", action_patterns)


layer_patterns = [
    [{"LOWER": "layer"}, {"IS_DIGIT": True}], 
    [{"LOWER": "the"}, {"ENT_TYPE": "ORDINAL"}, {"LOWER": "layer"}], 
]
matcher.add("LAYER", layer_patterns)

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

print(result)
print(parse_command(result))