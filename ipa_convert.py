custom_dict = {
    "this": "ðɪs",
    "video": "ˈvɪdioʊ",
    "is": "ɪz",
    "operating": "ˈɒpəreɪtɪŋ",
    "system": "ˈsɪstəm",
    "process": "ˈprɑːsɛs",
    "scheduling": "ˈʃɛdjuːlɪŋ",
    "deadlock": "ˈdɛdlɒk",
    "semaphore": "ˈsɛməfɔːr",
    "mutex": "ˈmjuːtɛks",
    "paging": "ˈpeɪdʒɪŋ",
    "memory": "ˈmɛməri",
    "management": "ˈmænɪdʒmənt",
    "bankers": "ˈbæŋkərz",
    "algorithm": "ˈælɡəˌrɪðəm",
    "kernel": "ˈkɜːrnəl",
    "thread": "θrɛd",
    "virtual": "ˈvɜːrtʃuəl",
    "binary": "ˈbaɪnəri",
    "round": "raʊnd",
    "robin": "ˈrɑːbɪn"
}

with open("transcript_final.txt", "r", encoding="utf-8") as f:
    text = f.read()

words = text.split()
ipa_output = []

for word in words:
    clean_word = word.lower().strip(",.!?")

    if clean_word in custom_dict:
        ipa_output.append(custom_dict[clean_word])
    else:
        ipa_output.append(clean_word)

final_ipa = " ".join(ipa_output)

print(final_ipa)

with open("ipa_output.txt", "w", encoding="utf-8") as f:
    f.write(final_ipa)

print("IPA conversion complete.")