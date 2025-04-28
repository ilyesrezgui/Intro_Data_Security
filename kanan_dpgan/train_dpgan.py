import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# -------------------------------
# 1. Load and Preprocess Dataset
# -------------------------------
data = pd.read_csv("time_series.csv")
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.values)

batch_size = 64
dataset = TensorDataset(torch.tensor(data_scaled, dtype=torch.float32))
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# -------------------------------
# 2. Define Generator and Discriminator
# -------------------------------
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.fc(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# Initialize models
noise_dim = 100
G = Generator(input_dim=noise_dim, output_dim=data_scaled.shape[1])
D = Discriminator(input_dim=data_scaled.shape[1])

# -------------------------------
# 3. Define Optimizers
# -------------------------------
optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# -------------------------------
# 4. Set up Privacy Engine for Discriminator
# -------------------------------
privacy_engine = PrivacyEngine()

D, optimizer_D, train_loader = privacy_engine.make_private_with_epsilon(
    module=D,
    optimizer=optimizer_D,
    data_loader=train_loader,
    target_epsilon=10.0,
    target_delta=1e-5,
    epochs=50,
    max_grad_norm=1.0,
)

# -------------------------------
# 5. Train Discriminator Only (with DP)
# -------------------------------
criterion = nn.BCELoss()
epochs_D = 50

print("Starting Discriminator training with Differential Privacy...")

for epoch in range(epochs_D):
    for real_batch, in train_loader:
        batch_size = real_batch.size(0)

        optimizer_D.zero_grad()

        # Train on real data
        real_labels = torch.ones(batch_size, 1)
        real_output = D(real_batch)
        loss_real = criterion(real_output, real_labels)

        # Train on fake data
        noise = torch.randn(batch_size, noise_dim)
        fake_data = G(noise)
        fake_labels = torch.zeros(batch_size, 1)
        fake_output = D(fake_data.detach())
        loss_fake = criterion(fake_output, fake_labels)

        # Total loss and optimization step
        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()

    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    print(f"(D) Epoch [{epoch+1}/{epochs_D}] Loss_D: {loss_D.item():.4f}  Privacy ε: {epsilon:.2f}")

print("Finished training Discriminator.")

# -------------------------------
# 6. Restore Discriminator to Normal (Remove Opacus hooks)
# -------------------------------
print("Restoring Discriminator to normal (non-private) mode...")

# Load the saved state_dict
state_dict = torch.load("discriminator_temp.pth")

# Remove the '_module.' prefix from keys if it exists
new_state_dict = {k.replace("_module.", ""): v for k, v in state_dict.items()}

# Create a fresh Discriminator model
D = Discriminator(input_dim=data_scaled.shape[1])

# Load the fixed state_dict
D.load_state_dict(new_state_dict)

# Set Discriminator to evaluation mode
D.eval()

print("Discriminator restored successfully.")


# Save D weights temporarily
torch.save(D.state_dict(), "discriminator_temp.pth")

# Reload clean D without Opacus hooks
D = Discriminator(input_dim=data_scaled.shape[1])
D.load_state_dict(torch.load("discriminator_temp.pth"))
D.eval()

print("Discriminator restored successfully.")

# -------------------------------
# 7. Train Generator Only (without DP)
# -------------------------------
print("Starting Generator training without Differential Privacy...")

epochs_G = 100

for epoch in range(epochs_G):
    optimizer_G.zero_grad()

    noise = torch.randn(batch_size, noise_dim)
    fake_data = G(noise)
    fake_output = D(fake_data)

    # Generator tries to fool Discriminator
    loss_G = criterion(fake_output, torch.ones(batch_size, 1))
    loss_G.backward()
    optimizer_G.step()

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"(G) Epoch [{epoch+1}/{epochs_G}] Loss_G: {loss_G.item():.4f}")

print("Finished training Generator.")

# -------------------------------
# 8. Save Models
# -------------------------------
torch.save(G.state_dict(), "generator.pth")
torch.save(D.state_dict(), "discriminator.pth")

print("Training completed and models saved successfully!")
