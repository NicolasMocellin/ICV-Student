import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

# --- Parameters ---
image_size = 64
latent_dim = 100
batch_size = 16  # Reduced for small dataset
epochs = 200  # Increased for better convergence
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Paths ---
data_path = r".\images\Train_small\35"
save_dir = r".\images\Generated\35_small"
os.makedirs(save_dir, exist_ok=True)

# --- Preprocessing with augmentation ---
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.3),  # Augmentation: horizontal flip
    transforms.RandomRotation(10),  # Augmentation: light rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Augmentation: color
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Correct normalization for RGB
])

# --- Custom dataset ---
class SimpleImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.transform = transform
        self.images = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(('.png', '.jpg', '.ppm'))
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0  # label 0 for compatibility

dataset = SimpleImageDataset(data_path, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- Generator definition ---
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            self.block(latent_dim, 512, 4, 1, 0),
            self.block(512, 256, 4, 2, 1),
            self.block(256, 128, 4, 2, 1),
            self.block(128, 64, 4, 2, 1),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def block(self, in_c, out_c, k, s, p):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True)
        )

    def forward(self, z):
        return self.model(z)

# --- Discriminator definition ---
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            self.block(3, 64, 4, 2, 1),
            self.block(64, 128, 4, 2, 1),
            self.block(128, 256, 4, 2, 1),
            self.block(256, 512, 4, 2, 1),
            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def block(self, in_c, out_c, k, s, p):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.model(x).view(-1, 1).squeeze(1)

# --- Weight initialization function ---
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# --- Initialization ---
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Weight initialization
generator.apply(weights_init)
discriminator.apply(weights_init)

loss_fn = nn.BCELoss()
# Lower learning rates for more stable training with small dataset
opt_g = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

# --- Training ---
for epoch in range(epochs):
    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
    for i, (imgs, _) in enumerate(loop):
        imgs = imgs.to(device)
        b_size = imgs.size(0)

        # Label smoothing: real=0.9, fake=0.1 instead of 1.0 and 0.0
        valid = torch.full((b_size,), 0.9, device=device)
        fake = torch.full((b_size,), 0.1, device=device)

        # ===== Train the Discriminator (2 times) =====
        for _ in range(2):
            opt_d.zero_grad()

            # Loss on real images
            real_loss = loss_fn(discriminator(imgs), valid)

            # Loss on generated images
            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
            gen_imgs = generator(noise)
            fake_loss = loss_fn(discriminator(gen_imgs.detach()), fake)

            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            opt_d.step()

        # ===== Train the Generator =====
        opt_g.zero_grad()

        # The generator wants to fool the discriminator (labels = real)
        noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
        gen_imgs = generator(noise)
        g_loss = loss_fn(discriminator(gen_imgs), torch.ones(b_size, device=device))

        g_loss.backward()
        opt_g.step()

        loop.set_postfix(d_loss=d_loss.item(), g_loss=g_loss.item())

    print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")


# --- Final image generation after training ---
generator.eval()
num_images = 25  # number of final images to generate
noise = torch.randn(num_images, latent_dim, 1, 1, device=device)
final_imgs = generator(noise)

for idx in range(num_images):
    img = final_imgs[idx]
    save_path = os.path.join(save_dir, f"final_img_{idx+1}.png")
    save_image(img, save_path, normalize=True)

print(f"{num_images} images generated in: {save_dir}")


