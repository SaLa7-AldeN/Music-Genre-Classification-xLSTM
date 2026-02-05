import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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
        out, _ = self.lstm(x)
        out = self.attention(out)
        return self.fc(out)
print(" Loading data and scaling parameters...")
X = np.load("NpyData/X_mfcc_UP.npy")
y = np.load("NpyData/y_mfcc_UP.npy")
scaling = np.load("NpyData/scaling_params.npy", allow_pickle=True).item()
X = (X - scaling['mean']) / (scaling['std'] + 1e-7)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
test_loader = DataLoader(
    TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)), 
    batch_size=32, 
    shuffle=False)
model = GenreModel().to(device)
try:
    model.load_state_dict(torch.load("best_music_model.pth", map_location=device, weights_only=True))
    print(" Model weights loaded successfully!")
except FileNotFoundError:
    print(" Error: 'best_music_model.pth' not found!")
    exit()
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        outputs = model(batch_X)
        all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        all_labels.extend(batch_y.numpy())
print("\n" + "="*60 + "\n FINAL CLASSIFICATION PERFORMANCE REPORT\n" + "="*60)
print(classification_report(all_labels, all_preds, target_names=GENRES))
fig, ax = plt.subplots(figsize=(12, 10))
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=GENRES)
disp.plot(cmap='magma', ax=ax, values_format='d', xticks_rotation=45)
test_acc = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
plt.title(f'Final Model Confusion Matrix (Overall Accuracy: {test_acc:.2f}%)')
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
print(f"\n Confusion Matrix saved as 'confusion_matrix.png'")
plt.show()