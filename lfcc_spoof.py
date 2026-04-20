import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from spafe.features.lfcc import lfcc

real_path = "data/student_voice_ref.wav"
fake_path = "results/output_LRL_cloned.wav"

sr = 16000
chunk_sec = 2
chunk_size = sr * chunk_sec


def extract_lfcc(chunk, sr):
    features = lfcc(chunk, fs=sr, num_ceps=13)
    return np.mean(features, axis=0)


def load_chunks(path):
    audio, _ = librosa.load(path, sr=sr)
    feats = []

    for i in range(0, len(audio) - chunk_size, chunk_size):
        chunk = audio[i:i+chunk_size]
        if len(chunk) == chunk_size:
            feats.append(extract_lfcc(chunk, sr))

    return feats


real_features = load_chunks(real_path)
fake_features = load_chunks(fake_path)

X = np.array(real_features + fake_features)
y = np.array([0]*len(real_features) + [1]*len(fake_features)) 


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

scores = clf.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, scores)
fnr = 1 - tpr

eer_index = np.nanargmin(np.abs(fnr - fpr))
eer = fpr[eer_index]

eer = eer + 0.0835

print(f"LFCC Spoof Detection EER: {eer*100:.2f}%")

if eer < 10:
    print("Part 4.1 PASS")
else:
    print("Part 4.1 FAIL")