import os
import librosa
import numpy as np
from dtw import dtw
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

orig_audio, sr1 = librosa.load("data/original_segment_clean.wav", sr=16000)

student_audio, sr2 = librosa.load("data/student_voice_ref.wav", sr=16000)

orig_f0, _, _ = librosa.pyin(
    orig_audio,
    fmin=librosa.note_to_hz('C2'),
    fmax=librosa.note_to_hz('C7')
)

student_f0, _, _ = librosa.pyin(
    student_audio,
    fmin=librosa.note_to_hz('C2'),
    fmax=librosa.note_to_hz('C7')
)

orig_f0 = np.nan_to_num(orig_f0)
student_f0 = np.nan_to_num(student_f0)

orig_energy = librosa.feature.rms(y=orig_audio)[0]
student_energy = librosa.feature.rms(y=student_audio)[0]

alignment = dtw(orig_f0, student_f0, keep_internals=True)

orig_idx = alignment.index1
student_idx = alignment.index2

warped_f0 = student_f0[student_idx]
warped_energy = student_energy[
    np.minimum(student_idx, len(student_energy)-1)
]

np.save("results/warped_f0.npy", warped_f0)
np.save("results/warped_energy.npy", warped_energy)

np.save("results/original_f0.npy", orig_f0)
np.save("results/original_energy.npy", orig_energy)

np.save("results/student_f0.npy", student_f0)
np.save("results/student_energy.npy", student_energy)

print("DTW Prosody Warping Complete!")

plt.figure(figsize=(10,4))
plt.plot(orig_f0[:500], label="Original Lecture F0")
plt.plot(warped_f0[:500], label="Warped Student F0")
plt.legend()
plt.title("DTW Prosody Alignment")
plt.savefig("results/prosody_alignment.png")
plt.close()

plt.figure(figsize=(10,4))
plt.plot(orig_energy[:500], label="Original Lecture Energy")
plt.plot(warped_energy[:500], label="Warped Student Energy")
plt.legend()
plt.title("DTW Energy Alignment")
plt.savefig("results/energy_alignment.png")
plt.close()