import librosa
import soundfile as sf
import noisereduce as nr

y, sr = librosa.load("data/original_segment.wav", sr=16000)

noise_sample = y[0:int(0.5 * sr)]

reduced_noise = nr.reduce_noise(
    y=y,
    sr=sr,
    y_noise=noise_sample
)


# sf.write("data/original_segment_clean.wav", reduced_noise, sr)

reduced_noise = reduced_noise / max(abs(reduced_noise))
sf.write("data/original_segment_clean.wav", reduced_noise, sr)

print("Denoising complete! Clean file saved.")