
import pyttsx3
import pandas as pd
import sys 
def speak(audio, rate, volume, lang):
    
    converter = pyttsx3.init()
    
    converter.setProperty('rate', rate)
    converter.setProperty('volume', volume)
    converter.setProperty('voice', lang)
    
    converter.say(audio)
    converter.runAndWait()
    converter.stop()
    
def set_rv(sentiment, rate_s):
    if  sentiment == 1:
        rate = rate_s-10 
        volume = 1
    elif sentiment == 0:
        rate = rate_s 
        volume = 0.5
    else:
        volume = 1
        rate = rate_s+20   
    return rate, volume

data = pd.read_csv("/Users/aishwaryaramanath/Downloads/test_captions.csv")
sentiment = list(data["sentiment"])[73] 
if sys.argv[1] == "English":
    lang = "com.apple.speech.synthesis.voice.Alex"
    rate,volume = set_rv(sentiment,150)

elif sys.argv[1] == "Italian":
    lang = "com.apple.speech.synthesis.voice.alice"
    rate,volume=set_rv(sentiment,140)

elif sys.argv[1] == "Spanish":
    lang = "com.apple.speech.synthesis.voice.paulina"
    rate,volume=set_rv(sentiment,150)
    
else:
    lang = "com.apple.speech.synthesis.voice.thomas"
    rate,volume=set_rv(sentiment,145)

speak(list(data['pred_caption'])[73], rate, volume, lang)



