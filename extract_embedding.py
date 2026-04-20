from speechbrain.pretrained import EncoderClassifier
import librosa
import torch

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec"
)

signal, sr = librosa.load("data/student_voice_ref.wav", sr=16000)

signal = torch.tensor(signal).unsqueeze(0)

embedding = classifier.encode_batch(signal)

embedding = embedding.squeeze().detach().cpu().numpy()

print("Embedding shape:", embedding.shape)

torch.save(torch.tensor(embedding), "speaker_embedding.pt")

print("Speaker embedding extracted successfully!")