from textGeneration import GenerativeModel

wizard_bot = GenerativeModel("you are a wize old wizard.", 50)
while True:
    msg = input("Speak to the wizard or 0 to quit: ")
    if msg == '0':
        break
    response = wizard_bot.prompt(msg)
    print("The Wizard says:", response)