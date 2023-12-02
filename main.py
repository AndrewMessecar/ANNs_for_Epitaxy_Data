import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Load CSV data
data = pd.read_csv("GaN_Temp_Crystal.csv").values


# Define a custom dataset class
class CustomDataset(Dataset):
	def __init__(self, data):
		self.data = torch.FloatTensor(data)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]


# Generator network
class Generator(nn.Module):
	def __init__(self, input_size, output_size):
		super(Generator, self).__init__()
		self.model = nn.Sequential(
			nn.Linear(input_size, 64),
			nn.ReLU(),
			nn.Linear(64, 32),
			nn.ReLU(),
			nn.Linear(32, output_size),
			nn.Tanh()
		)

	def forward(self, x):
		return self.model(x)


# Discriminator network
class Discriminator(nn.Module):
	def __init__(self, input_size):
		super(Discriminator, self).__init__()
		self.model = nn.Sequential(
			nn.Linear(input_size, 32),
			nn.ReLU(),
			nn.Linear(32, 16),
			nn.ReLU(),
			nn.Linear(16, 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		return self.model(x)


# Hyperparameters
input_size = data.shape[1]
latent_size = 100
output_size = input_size
lr = 0.0002
batch_size = 64
epochs = 1000000

# Initialize networks and optimizers
generator = Generator(latent_size, output_size)
discriminator = Discriminator(input_size)
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)
criterion = nn.BCELoss()

# Training loop
for epoch in range(epochs):
	for real_data in DataLoader(CustomDataset(data), batch_size=batch_size, shuffle=True):
		# Train discriminator
		optimizer_D.zero_grad()
		real_labels = torch.ones(real_data.size(0), 1)
		fake_labels = torch.zeros(real_data.size(0), 1)

		real_output = discriminator(real_data)
		real_loss = criterion(real_output, real_labels)
		real_loss.backward()

		noise = torch.randn(real_data.size(0), latent_size)
		fake_data = generator(noise)
		fake_output = discriminator(fake_data.detach())
		fake_loss = criterion(fake_output, fake_labels)
		fake_loss.backward()

		optimizer_D.step()

		# Train generator
		optimizer_G.zero_grad()
		fake_output = discriminator(fake_data)
		generator_loss = criterion(fake_output, real_labels)
		generator_loss.backward()
		optimizer_G.step()

	# Print progress
	if epoch % 100 == 0:
		print(
			f"Epoch [{epoch}/{epochs}], Generator Loss: {generator_loss.item()}, Discriminator Loss: {real_loss.item() + fake_loss.item()}")

	# Generate synthetic data and plot
	if epoch % 500 == 0:
		num_samples = 1000
		noise = torch.randn(num_samples, latent_size)
		generated_data = generator(noise).detach().numpy()

		# Plot the real and generated data
		plt.scatter(data[:, 0], data[:, 1], color='blue', label='Real Data', alpha=0.7)
		plt.scatter(generated_data[:, 0], generated_data[:, 1], color='red', label='Generated Data', alpha=0.7)

		plt.title('Real vs Generated Data')
		plt.xlabel('Feature 1')
		plt.ylabel('Feature 2')
		plt.legend()

		# Save the plot as an image
		plt.savefig(f'epoch_{epoch}_plot.png')
		plt.show()
