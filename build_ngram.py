from collections import defaultdict

with open("data/lm_corpus.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

words = text.split()

bigram = defaultdict(lambda: defaultdict(int))

for i in range(len(words)-1):
    w1 = words[i]
    w2 = words[i+1]
    bigram[w1][w2] += 1

print("Bigram Model Created:\n")

for word in bigram:
    print(word, dict(bigram[word]))