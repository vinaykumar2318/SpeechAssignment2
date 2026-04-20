translation_dict = {}

with open("data/bhojpuri_dictionary.txt", "r", encoding="utf-8") as f:
    for line in f:
        if "=" in line:
            eng, bho = line.strip().split("=", 1)
            translation_dict[eng.strip().lower()] = bho.strip()

with open("transcript_final.txt", "r", encoding="utf-8") as f:
    text = f.read()

words = text.split()
translated_output = []

for word in words:
    clean_word = word.lower().strip(",.!?")

    if clean_word in translation_dict:
        translated_output.append(translation_dict[clean_word])
    else:
        translated_output.append(word)

final_translation = " ".join(translated_output)

print(final_translation)

with open("bhojpuri_translation.txt", "w", encoding="utf-8") as f:
    f.write(final_translation)

print("Bhojpuri translation complete.")