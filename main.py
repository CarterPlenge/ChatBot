from textGeneration import GenerativeModel
from voiceReq import transcribe, record_voice
import os

wizard_bot = GenerativeModel("you are a wize old wizard.", 50)



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