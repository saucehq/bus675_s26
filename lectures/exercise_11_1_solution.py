import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


data_path = Path(__file__).resolve().parents[1] / "data" / "winequality-red.csv"

# Read wine data and normalize
wine_df = pd.read_csv(data_path, sep=";")
features = wine_df.drop("quality", axis=1).values.astype(np.float32)
targets = wine_df["quality"].values.astype(np.float32)
feature_means = features.mean(axis=0)
feature_stds = features.std(axis=0)
features = (features - feature_means) / feature_stds


class WineDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features)
        self.targets = torch.tensor(targets).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


dataset = WineDataset(features, targets)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


class ResidualBlock(nn.Module):
    """A residual block: output = F(x) + x."""

    def __init__(self, size):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.ReLU(),
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out = out + residual
        out = self.relu(out)
        return out


def train_model(model, train_loader, val_loader, num_epochs=40, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_hist = []
    val_hist = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for features_batch, targets_batch in train_loader:
            optimizer.zero_grad()
            pred = model(features_batch)
            loss = loss_fn(pred, targets_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_hist.append(train_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features_batch, targets_batch in val_loader:
                pred = model(features_batch)
                loss = loss_fn(pred, targets_batch)
                val_loss += loss.item()
        val_hist.append(val_loss / len(val_loader))

    return train_hist, val_hist


class WineResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(11, 32),
            nn.ReLU(),
        )
        self.res_block1 = ResidualBlock(32)
        self.res_block2 = ResidualBlock(32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.output(x)
        return x


model = WineResNet()
print(model)

dummy_batch = torch.randn(8, 11)
dummy_out = model(dummy_batch)
print("Dummy output shape:", tuple(dummy_out.shape))

total_params = sum(p.numel() for p in model.parameters())
print("Total parameter count:", total_params)

train_hist, val_hist = train_model(model, train_loader, val_loader)
print("Final train loss:", round(train_hist[-1], 4))
print("Final val loss:", round(val_hist[-1], 4))

plt.plot(train_hist, label="Train Loss")
plt.plot(val_hist, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("WineResNet Training Curves")
plt.legend()
plt.tight_layout()
plt.show()
