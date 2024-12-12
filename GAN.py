import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Create saved_figure directory if it doesn't exist
os.makedirs('saved_figure', exist_ok=True)

# Set random seed for reproducibility
torch.manual_seed(42)

# Hyperparameters
latent_dim = 100
hidden_dim = 256
image_dim = 784  # 28x28 MNIST images
batch_size = 32
num_epochs = 70
lr = 3e-4
beta1 = 0.5

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear = nn.Linear(latent_dim, hidden_dim * 4 * 4)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_dim * 4 * 4)

        self.conv1 = nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim // 2)

        self.conv2 = nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_dim // 4)

        self.conv3 = nn.ConvTranspose2d(hidden_dim // 4, 1, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, z):
        z = z + torch.randn_like(z) * 0.01  # Add noise to inputs for stability
        x = self.linear(z)  # Output shape: [batch_size, hidden_dim * 4 * 4]
        x = self.relu(x)
        x = self.bn1(x)
        x = x.view(-1, hidden_dim, 4, 4)  # Reshape for convolutions
        x = self.relu(self.bn2(self.conv1(x)))  # Output shape: [batch_size, hidden_dim // 2, 8, 8]
        x = self.relu(self.bn3(self.conv2(x)))  # Output shape: [batch_size, hidden_dim // 4, 16, 16]
        x = self.tanh(self.conv3(x))  # Output shape: [batch_size, 1, 28, 28]
        return x


# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, hidden_dim // 4, kernel_size=4, stride=2, padding=1)
        self.lrelu = nn.LeakyReLU(0.2)
        self.bn1 = nn.BatchNorm2d(hidden_dim // 4)
        
        self.conv2 = nn.Conv2d(hidden_dim // 4, hidden_dim // 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim // 2)
        
        self.conv3 = nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_dim)
        
        self.linear = nn.Linear(hidden_dim * 4 * 4, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.lrelu(self.bn1(self.conv1(x)))
        x = self.lrelu(self.bn2(self.conv2(x)))
        x = self.lrelu(self.bn3(self.conv3(x)))
        x = x.view(-1, hidden_dim * 4 * 4)  # Flatten
        x = self.sigmoid(self.linear(x))
        return x

# Load MNIST dataset with transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] range
])

print("Loading MNIST dataset...")
mnist_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

# DataLoader
print("Preparing DataLoader...")
dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

generator = Generator().to(device)
discriminator = Discriminator().to(device)

g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.5))
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.5))

# LSGAN Loss
mse_loss_fn = nn.MSELoss()

def lsgan_loss(output, target):
    min_size = min(output.size(0), target.size(0))
    return mse_loss_fn(output[:min_size], target[:min_size])

# Visualization function
def show_real_vs_fake_images(real_images, fake_images, epoch):
    plt.figure(figsize=(10, 5))
    
    # Real Images
    plt.subplot(1, 2, 1)
    plt.title("Real Images")
    plt.imshow(torchvision.utils.make_grid(
        real_images[:16].cpu(), 
        nrow=4, normalize=True
    ).permute(1, 2, 0), cmap='gray')
    plt.axis('off')
    
    # Fake Images
    plt.subplot(1, 2, 2)
    plt.title("Fake Images")
    plt.imshow(torchvision.utils.make_grid(
        fake_images[:16].cpu(), 
        nrow=4, normalize=True
    ).permute(1, 2, 0), cmap='gray')
    plt.axis('off')
    
    plt.savefig(os.path.join('saved_figure', f'comparison_epoch_{epoch+1}.png'))
    plt.close()

# Training Loop
print("Starting training...")
for epoch in tqdm(range(num_epochs), desc="Training Progress"):
    running_g_loss = 0
    running_d_loss = 0

    for real_images, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
        current_batch_size = real_images.size(0)
        real_images = real_images[:current_batch_size].to(device)
        real_labels = torch.ones(current_batch_size, 1).to(device)
        fake_labels = torch.zeros(current_batch_size, 1).to(device)

        # Train Discriminator
        d_optimizer.zero_grad()

        # Real images
        d_real_output = discriminator(real_images)
        d_real_loss = lsgan_loss(d_real_output, real_labels)

        # Fake images
        z = torch.randn(current_batch_size, latent_dim).to(device)
        fake_images = generator(z)  # Already in shape [batch_size, 1, 28, 28]
        d_fake_output = discriminator(fake_images.detach())
        d_fake_loss = lsgan_loss(d_fake_output, fake_labels)

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        g_optimizer.zero_grad()

        g_output = discriminator(fake_images)
        g_loss = lsgan_loss(g_output, real_labels)
        g_loss.backward()
        g_optimizer.step()

        running_g_loss += g_loss.item()
        running_d_loss += d_loss.item()

    avg_g_loss = running_g_loss / len(dataloader)
    avg_d_loss = running_d_loss / len(dataloader)

    print(f"Epoch [{epoch+1}/{num_epochs}] - Avg G Loss: {avg_g_loss:.4f}, Avg D Loss: {avg_d_loss:.4f}")

    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            fake_images = generator(torch.randn(16, latent_dim).to(device))
            real_images, _ = next(iter(dataloader))
            show_real_vs_fake_images(real_images, fake_images, epoch)


# Save models
print("Saving final models...")
torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')
print("Training completed!")