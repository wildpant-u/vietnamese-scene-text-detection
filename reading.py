import json
from gtts import gTTS
from playsound import playsound

def read_aloud(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()

    language = 'vi'

    # # Create audio file
    tts = gTTS(text = text, lang=language)
    tts.save('output.mp3')

    # # #Play audio file
    playsound('output.mp3')

with open('./inference_results/system_results.txt', 'r', encoding='utf-8') as f:
    words = json.load(f)

lowercase_words = [word.lower() for word in words]

with open('output.txt', mode = 'w', encoding='utf-8') as file:
    #Write each lowercased word to the file
    for word in lowercase_words:
        file.write(word + ' ')

filename = 'output.txt'
read_aloud(filename)