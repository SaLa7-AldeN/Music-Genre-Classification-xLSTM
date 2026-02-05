import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split

X = np.load("NpyData/X_mfcc_UP.npy")
y = np.load("NpyData/y_mfcc_UP.npy")
scaling = np.load("NpyData/scaling_params.npy", allow_pickle=True).item()
X = (X - scaling['mean']) / (scaling['std'] + 1e-7)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
train_loader = DataLoader(TensorDataset(torch.Tensor(X_train), torch.LongTensor(y_train)), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.Tensor(X_test), torch.LongTensor(y_test)), batch_size=32)
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
        self.lstm = nn.LSTM(120, 256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.attention = Attention(256)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10) )
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.attention(out)
        return self.fc(out)
device = torch.device("cuda")
model = GenreModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005) 
epochs = 70
print(" Start...")
for epoch in range(epochs): 
    model.train()
    for b_x, b_y in train_loader:
        b_x, b_y = b_x.to(device), b_y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(b_x), b_y)
        loss.backward()
        optimizer.step() 
    if (epoch+1) % 5 == 0:
        model.eval()
        correct = 0
        with torch.no_grad():
            for b_x, b_y in test_loader:
                b_x, b_y = b_x.to(device), b_y.to(device)
                pred = model(b_x).argmax(1)
                correct += (pred == b_y).sum().item()
        print(f"Epoch {epoch+1} | Accuracy: {100 * correct / len(y_test):.2f}%")
torch.save(model.state_dict(), "best_music_model.pth")
print("Done! Model saved as 'best_music_modell.pth'")