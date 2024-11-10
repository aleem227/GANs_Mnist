import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
from copy import deepcopy
from torchvision.utils import save_image

# Set random seed for reproducibility
torch.manual_seed(42)

# Hyperparameters
latent_dim = 100
hidden_dim = 256
image_dim = 784  # 28x28 MNIST images
batch_size = 32
num_epochs = 100
lr = 0.0002
beta1 = 0.5

# Generator Network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, image_dim),
            nn.Tanh()  # Output values between -1 and 1 for MNIST images
        )
        # Initialize weights
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, z):
        return self.model(z)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output between 0 and 1 for real/fake classification
        )
        # Initialize weights
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.model(x)

# Load MNIST dataset with transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
])

mnist_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)

# Initialize networks and optimizers
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator().to(device)
discriminator = Discriminator().to(device)

g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

criterion = nn.BCELoss()

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
    
    def __call__(self, model, loss):
        if self.best_loss is None:
            self.best_loss = loss
            self.best_model = deepcopy(model.state_dict())
        elif loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.best_model = deepcopy(model.state_dict())
            self.counter = 0

# Initialize early stopping
g_early_stopping = EarlyStopping(patience=10, verbose=True)
d_early_stopping = EarlyStopping(patience=10, verbose=True)

# Modify training loop
best_fid_score = float('inf')
running_g_loss = 0
running_d_loss = 0

for epoch in range(num_epochs):
    running_g_loss = 0
    running_d_loss = 0
    
    for i, (real_images, _) in enumerate(dataloader):
        batch_size = real_images.size(0)
        real_images = real_images.view(-1, image_dim).to(device)
        
        # Labels for real and fake images
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # Train Discriminator
        d_optimizer.zero_grad()
        d_real_output = discriminator(real_images)
        d_real_loss = criterion(d_real_output, real_labels)
        
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = generator(z)
        d_fake_output = discriminator(fake_images.detach())
        d_fake_loss = criterion(d_fake_output, fake_labels)
        
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()
        
        # Train Generator
        g_optimizer.zero_grad()
        g_output = discriminator(fake_images)
        g_loss = criterion(g_output, real_labels)
        g_loss.backward()
        g_optimizer.step()
        
        # Update running losses
        running_g_loss += g_loss.item()
        running_d_loss += d_loss.item()
        
    # Calculate average epoch losses
    avg_g_loss = running_g_loss / len(dataloader)
    avg_d_loss = running_d_loss / len(dataloader)
    
    # Early stopping checks
    g_early_stopping(generator, avg_g_loss)
    d_early_stopping(discriminator, avg_d_loss)
    
    # Print epoch statistics
    print(f'Epoch [{epoch}/{num_epochs}], '
          f'Avg G_Loss: {avg_g_loss:.4f}, Avg D_Loss: {avg_d_loss:.4f}')
    
    # Generate and save sample images
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            fake_images = generator(torch.randn(16, latent_dim).to(device))
            fake_images = fake_images.view(-1, 1, 28, 28)  # Ensure correct shape for saving
            save_image(fake_images, f'fake_images_epoch_{epoch+1}.png', nrow=4, normalize=True)
    
    # Check if early stopping criteria met
    if g_early_stopping.early_stop or d_early_stopping.early_stop:
        print("Early stopping triggered")
        # Load best models
        generator.load_state_dict(g_early_stopping.best_model)
        discriminator.load_state_dict(d_early_stopping.best_model)
        break

# Save final models
torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')
