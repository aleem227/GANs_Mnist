import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )

    def forward(self, z):
        return self.fc(z).view(z.size(0), 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(28*28, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

def save_images(real_images, fake_images, epoch):
    # Denormalize images
    real_images = (real_images + 1) / 2  # Tanh normalization
    fake_images = (fake_images + 1) / 2  # Tanh normalization

    # Convert to numpy arrays for matplotlib
    real_images = real_images.cpu().numpy()
    fake_images = fake_images.cpu().numpy()

    # Plot real vs generated images
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(8):
        axes[0, i].imshow(real_images[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(fake_images[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(f'saved_figure/epoch_{epoch+1}_comparison.png')
    plt.close()

def bce_loss(output, target):
    bce_loss_fn = nn.BCELoss()
    return bce_loss_fn(output, target)

latent_dim = 100
num_epochs = 70
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

g_optimizer = optim.Adam(generator.parameters(), lr=3e-4, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=3e-4, betas=(0.5, 0.999))

# Create directory for saving the generator model if it doesn't exist
os.makedirs('saved_models', exist_ok=True)

for epoch in tqdm(range(num_epochs), desc="Training Progress"):
    running_g_loss = 0
    running_d_loss = 0

    for real_images, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
        current_batch_size = real_images.size(0)
        real_labels = torch.ones((current_batch_size, 1), device=device).float()
        fake_labels = torch.zeros((current_batch_size, 1), device=device).float()

        real_images = real_images.to(device)

        # Train Discriminator
        d_optimizer.zero_grad()

        d_real_output = discriminator(real_images)
        d_real_output = d_real_output.view(-1, 1)
        d_real_loss = bce_loss(d_real_output, real_labels)

        z = torch.randn(current_batch_size, latent_dim).to(device)
        fake_images = generator(z).detach()
        d_fake_output = discriminator(fake_images)
        d_fake_output = d_fake_output.view(-1, 1)
        d_fake_loss = bce_loss(d_fake_output, fake_labels)

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        g_optimizer.zero_grad()

        z = torch.randn(current_batch_size, latent_dim).to(device)
        fake_images = generator(z)
        g_output = discriminator(fake_images)
        g_loss = bce_loss(g_output, real_labels)
        g_loss.backward()
        g_optimizer.step()

        running_g_loss += g_loss.item()
        running_d_loss += d_loss.item()

    avg_g_loss = running_g_loss / len(dataloader)
    avg_d_loss = running_d_loss / len(dataloader)

    print(f"Epoch [{epoch+1}/{num_epochs}] - Avg G Loss: {avg_g_loss:.4f}, Avg D Loss: {avg_d_loss:.4f}")

    with torch.no_grad():
        z = torch.randn(16, latent_dim).to(device)
        fake_images = generator(z)

        # Get a batch of real images
        real_images, _ = next(iter(dataloader))
        real_images = real_images.to(device)

        # Save real vs generated image comparison
        save_images(real_images, fake_images, epoch)

    # Save Generator model every epoch
    torch.save(generator.state_dict(), f'saved_models/generator_epoch_{epoch+1}.pth')

