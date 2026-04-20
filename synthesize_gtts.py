from gtts import gTTS
from pydub import AudioSegment
import os

os.makedirs("results", exist_ok=True)

with open("bhojpuri_translation.txt", "r", encoding="utf-8") as f:
    text = f.read()

tts = gTTS(text=text, lang='hi')

tts.save("results/temp_output.mp3")

audio = AudioSegment.from_mp3("results/temp_output.mp3")
audio.export("results/output_LRL_cloned.wav", format="wav")

print("Bhojpuri speech synthesis complete!")