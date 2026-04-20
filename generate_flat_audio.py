import librosa
import numpy as np
import soundfile as sf

audio, sr = librosa.load("results/output_LRL_cloned.wav", sr=22050)

audio = audio / np.max(np.abs(audio))

window_size = 1000
kernel = np.ones(window_size) / window_size
flat_audio = np.convolve(audio, kernel, mode='same')

flat_audio = flat_audio / np.max(np.abs(flat_audio))

sf.write("results/flat_synthesis.wav", flat_audio, sr)

print("Flat synthesis (audible) created.")