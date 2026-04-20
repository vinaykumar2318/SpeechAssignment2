import torch
import librosa
import numpy as np
from lid_model import LIDModel

model = LIDModel()
model.load_state_dict(torch.load("lid_model.pth"))
model.eval()

y, sr = librosa.load("data/original_segment.wav", sr=16000)

mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
mfcc = mfcc.T

X = torch.tensor(mfcc, dtype=torch.float32)

with torch.no_grad():
    outputs = model(X)
    preds = torch.argmax(outputs, dim=1)

frame_duration = len(y)/sr / len(preds)

for i, pred in enumerate(preds[:50]):  
    label = "Hindi" if pred.item() == 0 else "English"
    start_time = i * frame_duration
    end_time = (i+1) * frame_duration
    print(f"{start_time:.2f}s - {end_time:.2f}s : {label}")