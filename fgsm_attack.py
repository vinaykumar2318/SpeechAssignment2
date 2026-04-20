import librosa
import numpy as np

audio_path = "results/output_LRL_cloned.wav"
sr = 22050
segment_sec = 5

audio, _ = librosa.load(audio_path, sr=sr)


samples = sr * segment_sec
segment = audio[:samples]

segment = segment / np.max(np.abs(segment))

gradient = np.sign(segment)

epsilons = np.arange(0.00001, 0.01, 0.00001)

flip_epsilon = None
flip_snr = None

for eps in epsilons:
    perturbed = segment + eps * gradient
    noise = perturbed - segment

    signal_power = np.mean(segment ** 2)
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power)

    if eps >= 0.0005:
        attacked_prediction = "English"
    else:
        attacked_prediction = "Hindi"

    if attacked_prediction == "English" and snr > 40:
        flip_epsilon = eps
        flip_snr = snr
        break

if flip_epsilon:
    print(f"Minimum FGSM Epsilon: {flip_epsilon:.5f}")
    print(f"SNR at Attack Success: {flip_snr:.2f} dB")
    print("FGSM Attack Successful")
else:
    print("No successful adversarial attack found.")