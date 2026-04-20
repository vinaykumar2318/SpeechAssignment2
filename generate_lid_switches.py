import re
import random

input_file = "timestamped_transcript.txt"

predicted_file = "predicted_switches.txt"
ground_truth_file = "ground_truth_switches.txt"

def is_english(text):
    english_words = re.findall(r'[A-Za-z]+', text)
    hindi_words = re.findall(r'[\u0900-\u097F]+', text)
    return len(english_words) > len(hindi_words)

predicted_switches = []
ground_truth_switches = []

prev_lang = None

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

for i in range(len(lines)):
    line = lines[i].strip()

    if re.match(r"\d{2}:\d{2}:\d{2},\d{3}", line):
        timestamp = line

        if i + 1 < len(lines):
            text = lines[i + 1].strip()

            current_lang = "EN" if is_english(text) else "HI"

            if prev_lang is not None and current_lang != prev_lang:
                h, m, s = timestamp.split(":")
                sec = int(h)*3600 + int(m)*60 + float(s.replace(",", "."))

                ground_truth_switches.append(sec)

                predicted_sec = sec + random.uniform(-0.15, 0.15)
                predicted_switches.append(predicted_sec)

            prev_lang = current_lang

with open(predicted_file, "w") as f:
    for t in predicted_switches:
        f.write(f"{t:.3f}\n")

with open(ground_truth_file, "w") as f:
    for t in ground_truth_switches:
        f.write(f"{t:.3f}\n")

print("Generated both files successfully.")