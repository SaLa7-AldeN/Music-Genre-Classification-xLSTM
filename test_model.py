import torch
import torch.nn as nn
import librosa
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GENRES = sorted(['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'])
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, 1)
    def forward(self, x):
        weights = torch.softmax(torch.tanh(self.attn(x)), dim=1)
        return torch.sum(x * weights, dim=1)
class GenreModel(nn.Module):
    def __init__(self):
        super(GenreModel, self).__init__()
        self.lstm = nn.LSTM(120, 256, num_layers=2, batch_first=True, bidirectional=True)
        self.attention = Attention(256)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)  )
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.attention(out)
        return self.fc(out)
model = GenreModel().to(device)
model.load_state_dict(torch.load("best_music_model.pth", map_location=device, weights_only=True))
model.eval()
scaling = np.load("NpyData/scaling_params.npy", allow_pickle=True).item()
def predict_top_2(file_path):
    y, sr = librosa.load(file_path, sr=22050, duration=30)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    combined = np.concatenate((mfcc, d1, d2), axis=0).T
    if len(combined) >= 430: combined = combined[:430, :]
    else: combined = np.pad(combined, ((0, 430 - len(combined)), (0, 0)))
    combined = (combined - scaling['mean']) / (scaling['std'] + 1e-7)
    input_tensor = torch.tensor(combined, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    top_indices = np.argsort(probs)[::-1]
    return (GENRES[top_indices[0]], probs[top_indices[0]], 
            GENRES[top_indices[1]], probs[top_indices[1]])
print(f" Music Genre Recognition System - Final Test")
print("-" * 85)
print(f"{'File Name':<25} | {'Primary Genre':<15} | {'Conf':<8} | {'Alternative'}")
print("-" * 85)
for file in os.listdir("test_audio"):
    if file.endswith((".wav", ".mp3")):
        try:
            g1, c1, g2, c2 = predict_top_2(os.path.join("test_audio", file))
            status = "✅" if c1 > 0.8 else "⚠️ "
            print(f"{file[:25]:<25} | {status} {g1.upper():<13} | {c1*100:>5.1f}% | {g2.upper()} ({c2*100:.1f}%)")
        except Exception as e:
            print(f"Error in {file}: {e}")