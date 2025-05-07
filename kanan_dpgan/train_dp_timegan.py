import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess real dataset
df = pd.read_csv("time_series.csv")  # Make sure this file is in your working directory
df = df.dropna()

# Drop the 'time' column (we don't need it as input)
df = df.drop(columns=['time'])

# Normalize the data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df.values)

# Reshape into sequences (e.g., 24 timesteps per sample)
sequence_length = 24
sequences = []
for i in range(len(normalized_data) - sequence_length):
    sequences.append(normalized_data[i:i+sequence_length])

X_train = torch.tensor(np.array(sequences), dtype=torch.float32)

# Hyperparameters
# Hyperparameters
hidden_dim = 24
feature_dim = 5  # Set the feature dimension to 5
batch_size = 64
lr = 1e-3
delta = 1e-5

# ========================
# Models
# ========================
class Embedder(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(Embedder, self).__init__()
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, x):
        h, _ = self.lstm(x)
        return h, _

class Recovery(nn.Module):
    def __init__(self, hidden_dim, feature_dim):
        super(Recovery, self).__init__()
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=feature_dim, batch_first=True)

    def forward(self, h):
        x_tilde, _ = self.lstm(h)
        return x_tilde, _

class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, h):
        return self.model(h)

# Initialize models
embedder = Embedder(feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
recovery = Recovery(hidden_dim=hidden_dim, feature_dim=feature_dim).to(device)
discriminator = Discriminator(hidden_dim=hidden_dim).to(device)

# Optimizers
optimizer_autoencoder = optim.Adam(list(embedder.parameters()) + list(recovery.parameters()), lr=lr)
optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=lr)

# DataLoader
train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)

# ========================
# Train Embedder-Recovery (Autoencoder)
# ========================
print("Training Embedder-Recovery (Autoencoder)...")
criterion_mse = nn.MSELoss()

for epoch in range(50):
    for batch_x, in train_loader:
        batch_x = batch_x.to(device)
        optimizer_autoencoder.zero_grad()

        h, _ = embedder(batch_x)
        x_tilde, _ = recovery(h)

        loss = criterion_mse(x_tilde, batch_x)
        loss.backward()
        optimizer_autoencoder.step()

    print(f"(Autoencoder) Epoch [{epoch+1}/50] Loss: {loss.item():.4f}")

print("Finished training Embedder-Recovery.")

# ========================
# Train Discriminator with DP
# ========================
print("Training Discriminator with Differential Privacy...")

# Attach Privacy Engine
privacy_engine = PrivacyEngine()
discriminator, optimizer_discriminator, train_loader = privacy_engine.make_private(
    module=discriminator,
    optimizer=optimizer_discriminator,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
)

criterion_bce = nn.BCELoss()

for epoch in range(50):
    for batch_x, in train_loader:
        batch_x = batch_x.to(device)
        optimizer_discriminator.zero_grad()

        # Get hidden representations
        h_real, _ = embedder(batch_x)
        h_real = h_real[:, -1, :]  # Last timestep

        # Fake latent samples
        h_fake = torch.randn_like(h_real)

        real_labels = torch.ones(h_real.size(0), 1, device=device)
        fake_labels = torch.zeros(h_fake.size(0), 1, device=device)

        outputs_real = discriminator(h_real)
        outputs_fake = discriminator(h_fake)

        loss_real = criterion_bce(outputs_real, real_labels)
        loss_fake = criterion_bce(outputs_fake, fake_labels)

        loss = (loss_real + loss_fake) / 2
        loss.backward()
        optimizer_discriminator.step()

    epsilon = privacy_engine.get_epsilon(delta=delta)
    print(f"(D) Epoch [{epoch+1}/50] Loss_D: {loss.item():.4f}  Privacy ε: {epsilon:.2f}")

print("Finished training Discriminator.")
