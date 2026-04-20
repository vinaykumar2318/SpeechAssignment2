import whisper

model = whisper.load_model("small")

with open("data/lm_corpus.txt", "r", encoding="utf-8") as f:
    bias_words = f.read().splitlines()

prompt_text = " ".join(bias_words)

result = model.transcribe(
    "data/original_segment_clean.wav",
    language="en",
    initial_prompt=prompt_text,
    temperature=0.0,
    beam_size=3
)

print(result["text"])

with open("transcript_final.txt", "w", encoding="utf-8") as f:
    f.write(result["text"])

print("Transcription complete.")
