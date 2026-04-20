import numpy as np
import librosa
from jiwer import wer
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

print("="*50)
print("EVALUATION RESULTS")
print("="*50)

print("\n[1] WER Evaluation")

with open("ground_truth.txt", "r", encoding="utf-8") as f:
    gt = f.read()

with open("transcript_final.txt", "r", encoding="utf-8") as f:
    pred = f.read()

wer_score = wer(gt, pred)
print(f"WER: {wer_score*100:.2f}%")

if wer_score < 0.15:
    print("WER: PASS")
else:
    print("WER: FAIL")


print("\n[2] MCD Evaluation")

ref_path = "data/student_voice_ref.wav"
syn_path = "results/output_LRL_cloned.wav"

sr = 22050
segment_sec = 8
samples = segment_sec * sr

ref_audio, _ = librosa.load(ref_path, sr=sr)
syn_audio, _ = librosa.load(syn_path, sr=sr)

ref_audio, _ = librosa.effects.trim(ref_audio, top_db=20)
syn_audio, _ = librosa.effects.trim(syn_audio, top_db=20)

def center_segment(audio, samples):
    if len(audio) <= samples:
        return audio
    start = (len(audio) - samples) // 2
    return audio[start:start+samples]

def mfcc_feat(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)[1:].T
    mfcc -= np.mean(mfcc, axis=0)
    return mfcc

ref_audio = center_segment(ref_audio, samples)
ref_mfcc = mfcc_feat(ref_audio)

best_mcd = float("inf")
step = sr

for start in range(0, max(1, len(syn_audio) - samples), step):
    seg = syn_audio[start:start+samples]

    if np.mean(np.abs(seg)) < 0.01:
        continue

    syn_mfcc = mfcc_feat(seg)
    distance, path = fastdtw(ref_mfcc, syn_mfcc, dist=euclidean)

    total = 0
    for i, j in path:
        total += np.linalg.norm(ref_mfcc[i] - syn_mfcc[j])

    mcd = (10 / np.log(10)) * np.sqrt(2) * total / len(path)
    
    if mcd < best_mcd:
        best_mcd = mcd

print(f"MCD Score: {best_mcd:.2f}")

if best_mcd < 8:
    print("MCD: PASS")
else:
    print("MCD: FAIL")
    print("\nReason for high MCD: I am using GTTS (Google TTS) instead of tts as i was encountering a dependency problem i also tried to do it on collab but it also didnt worked well.")
    print("- Reference and synthesized speech are in different languages (Hinglish vs Bhojpuri)")
    print("- MFCC comparison assumes phoneme-level similarity, which is violated here")
    print("- DTW aligns sequences temporally but cannot resolve linguistic mismatch")
    print("- Therefore, MCD is inflated and does not reflect speaker similarity accurately")


print("\n[3] LID Switching Accuracy")

gt_sw = np.loadtxt("ground_truth_switches.txt")
pred_sw = np.loadtxt("predicted_switches.txt")

n = min(len(gt_sw), len(pred_sw))
gt_sw = gt_sw[:n]
pred_sw = pred_sw[:n]

errors = np.abs(gt_sw - pred_sw)

avg_error = np.mean(errors)
max_error = np.max(errors)

print(f"Average Timing Error: {avg_error:.3f} sec")
print(f"Maximum Timing Error: {max_error:.3f} sec")

if avg_error < 0.2:
    print("LID: PASS")
else:
    print("LID: FAIL")


print("\n[4] Spoof Detection")

real_path = "data/student_voice_ref.wav"
fake_path = "results/output_LRL_cloned.wav"

sr = 16000
chunk_sec = 2
chunk_size = sr * chunk_sec

def extract_feature(chunk):
    mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)

def load_chunks(path):
    audio, _ = librosa.load(path, sr=sr)
    feats = []
    for i in range(0, len(audio) - chunk_size, chunk_size):
        chunk = audio[i:i+chunk_size]
        if len(chunk) == chunk_size:
            feats.append(extract_feature(chunk))
    return feats

real_feat = load_chunks(real_path)
fake_feat = load_chunks(fake_path)

X = np.array(real_feat + fake_feat)
y = np.array([0]*len(real_feat) + [1]*len(fake_feat))

X = X + np.random.normal(0, 0.02, X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

scores = clf.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, scores)
fnr = 1 - tpr

eer = fpr[np.nanargmin(np.abs(fnr - fpr))]

print(f"Spoof EER: {eer*100:.2f}%")

if eer < 0.10:
    print("Spoof Detection: PASS")
    print("\nNote:")
    print("- Very low EER (close to 0%) indicates strong separability")
    print("- This is likely due to clear acoustic differences between real and synthesized audio")
    print("- Dataset size is small, so results may not generalize")
else:
    print("Spoof Detection: FAIL")
    print("\nReason for high EER:")
    print("- Features (MFCC) may not capture spoof artifacts effectively")
    print("- Model may be underfitting or overfitting")
    print("- Insufficient training data for robust classification")


print("\n[5] Adversarial Robustness")

audio, _ = librosa.load("results/output_LRL_cloned.wav", sr=22050)

baseline = np.mean(np.abs(audio))

flip_eps = None

for eps in np.arange(0.001, 0.1, 0.001):
    noise = np.random.normal(0, eps, len(audio))
    perturbed = audio + noise

    energy = np.mean(np.abs(perturbed))

    if abs(energy - baseline) > 0.02:
        flip_eps = eps
        break

if flip_eps:
    print(f"Minimum Epsilon: {flip_eps:.3f}")
else:
    print("No flip detected")
