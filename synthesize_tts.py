from TTS.api import TTS
import os

os.makedirs("results", exist_ok=True)

with open("bhojpuri_translation.txt", "r", encoding="utf-8") as f:
    text = f.read()

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

tts.tts_to_file(
    text=text,
    speaker_wav="data/student_voice_ref.wav",
    language="hi",
    file_path="results/output_LRL_cloned_xtts.wav" 
)

print("XTTS Voice Cloning Done!")