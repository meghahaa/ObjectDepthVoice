import pyttsx3
import threading

text_to_speech_engine = pyttsx3.init()
thread_lock = threading.Lock()

def say_it_aloud(text):
    with thread_lock:
        text_to_speech_engine.say(text)
        text_to_speech_engine.runAndWait()

def speak_in_bgd(sentence):
    threading.Thread(target=say_it_aloud, args=(sentence,)).start()