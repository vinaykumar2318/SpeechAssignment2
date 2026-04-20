import numpy as np
from collections import defaultdict

corpus = [
    "process scheduling important ba",
    "memory management important ba",
    "process synchronization easy ba",
    "memory aur process dono jaruri ba"
]

bigram_counts = defaultdict(int)
unigram_counts = defaultdict(int)

for sentence in corpus:
    words = sentence.split()
    for i in range(len(words)-1):
        unigram_counts[words[i]] += 1
        bigram_counts[(words[i], words[i+1])] += 1

def bigram_prob(w1, w2):
    return bigram_counts[(w1, w2)] / (unigram_counts[w1] + 1e-6)

vocab = ["memory", "management", "important"]
logits = np.array([1.2, 1.0, 0.9])

context_word = "memory"

lambda_bias = 2.0

bias = np.array([
    np.log(bigram_prob(context_word, w) + 1e-6)
    for w in vocab
])

biased_logits = logits + lambda_bias * bias

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

original_probs = softmax(logits)
biased_probs = softmax(biased_logits)

print("Vocabulary:", vocab)
print("\nOriginal probabilities:")
for w, p in zip(vocab, original_probs):
    print(f"{w}: {p:.4f}")

print("\nBiased probabilities:")
for w, p in zip(vocab, biased_probs):
    print(f"{w}: {p:.4f}")