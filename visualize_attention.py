import torch
import torch.nn as nn
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GENRES = sorted(['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'])
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, 1)
    def forward(self, x):
        attn_scores = torch.tanh(self.attn(x))
        weights = torch.softmax(attn_scores, dim=1)
        return torch.sum(x * weights, dim=1), weights
class GenreModel(nn.Module):
    def __init__(self, input_size=120, hidden_size=256, num_layers=2):
        super(GenreModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10) )
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, weights = self.attention(lstm_out)
        return self.fc(attn_out), weights
model = GenreModel().to(device)
try:
    model.load_state_dict(torch.load("best_music_model.pth", map_location=device, weights_only=True))
    model.eval()
    print(" Model loaded successfully for visualization.")
except FileNotFoundError:
    print(" Error: 'fixed_model_120.pth' not found!")
    exit()
scaling = np.load("NpyData/scaling_params.npy", allow_pickle=True).item()
def visualize_music_attention(file_path):
    y, sr = librosa.load(file_path, sr=22050, duration=30)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    combined = np.concatenate((mfcc, d1, d2), axis=0).T
    if len(combined) >= 430: combined = combined[:430, :]
    else: combined = np.pad(combined, ((0, 430 - len(combined)), (0, 0)))
    combined_scaled = (combined - scaling['mean']) / (scaling['std'] + 1e-7)
    input_tensor = torch.tensor(combined_scaled, dtype=torch.float32).unsqueeze(0).to(device) 
    with torch.no_grad():
        logits, attn_weights = model(input_tensor)
        genre_idx = torch.argmax(logits).item()
        probs = torch.softmax(logits, dim=1)[0]
    weights = attn_weights.cpu().numpy().squeeze()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    librosa.display.waveshow(y, sr=sr, ax=ax1, alpha=0.5, color='steelblue')
    ax1.set_title(f"Analysis for: {os.path.basename(file_path)}\nPredicted Genre: {GENRES[genre_idx].upper()} ({probs[genre_idx]*100:.1f}%)", fontsize=14)
    ax1.set_ylabel("Amplitude")
    times = np.linspace(0, 30, num=len(weights))
    ax2.fill_between(times, weights, color='crimson', alpha=0.7)
    ax2.set_title("Attention mechanism: Where the model is 'listening' most", fontsize=12)
    ax2.set_ylabel("Importance Score")
    ax2.set_xlabel("Time (seconds)")
    plt.tight_layout()
    output_name = f"attention_{os.path.basename(file_path)}.png"
    plt.savefig(output_name, dpi=300)
    print(f" Visualization saved as: {output_name}")
    plt.show()
test_file = r"test_audio/country-30s.wav" 
if os.path.exists(test_file):
    visualize_music_attention(test_file)
else:
    print(f"⚠️ Test file not found at: {test_file}")