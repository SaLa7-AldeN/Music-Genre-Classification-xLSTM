import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

file_path = "DataSet/DATASET_SUPER/jazz/fast_jazz.00001.wav"
y, sr = librosa.load(file_path, sr=22050)
print("Sample Rate:", sr)
print("Audio Length:", len(y))
mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
mel_db = librosa.power_to_db(mel, ref=np.max)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format="%+2.0f dB")
plt.title("Mel-Spectrogram")
plt.tight_layout()
plt.show()
