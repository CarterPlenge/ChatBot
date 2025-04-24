from textGeneration import GenerativeModel
from voiceReq import transcribe, record_voice
from textToSpeech import TTS
import os

wizard_bot = GenerativeModel("You are a pirate.", 50)
voice = TTS()


while True:
    print("Speak to the wizard or say 'quit' to quit: ")
    file = record_voice() # record voice as input
    if file:
        msg = transcribe(file) # change from
        os.remove(file)
    else:
        print("somthing went wrong with voice input")
        continue
    
    if msg.strip() == 'Quit.':
        break
    response = wizard_bot.prompt(msg)
    print("The Wizard says:", response)
    voice.say(response)