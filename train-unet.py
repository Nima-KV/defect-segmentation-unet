from google.colab import drive

# Mount Google Drive (run this once per session)
drive.mount('/content/drive')

# After mounting, your Drive appears here:
print("Your Drive is mounted at: /content/drive/MyDrive/")

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class DefectDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # Load as grayscale
        image = np.array(Image.open(img_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L"))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Convert mask to binary float (0 or 1), add channel dimension
        mask = (mask > 127).float().unsqueeze(0)  # 1xHxW

        return image, mask
    
    # Your dataset paths (fixed the double slash typo)
image_dir = '/content/drive/MyDrive/images-unet-train'
mask_dir  = '/content/drive/MyDrive/images-unet-train/image-unet-train-binary'

# Where to save the best model (recommended: put it in Drive)
BEST_MODEL_PATH = '/content/drive/MyDrive/images-unet-train/best_defect_unet_ROI_resnet34.pth'

# Optional: print to confirm paths exist
print("Image dir exists?", os.path.isdir(image_dir))
print("Mask dir exists?",  os.path.isdir(mask_dir))

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.8),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.ElasticTransform(alpha=1, sigma=50, p=0.5),      # Fixed: removed deprecated param
    A.GridDistortion(p=0.5),
    A.Resize(height=256, width=512),                   # Good for both small & wide images
    A.Normalize(mean=0.5, std=0.5),                    # For single-channel grayscale
    ToTensorV2(),                                      # Converts to torch Tensor (1xHxW)
])

val_transform = A.Compose([
    A.Resize(height=256, width=512),
    A.Normalize(mean=0.5, std=0.5),
    ToTensorV2(),
])

# Train on ALL images with augmentation
train_dataset = DefectDataset(image_dir, mask_dir, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)

# Validation: same images but without heavy augmentation (for monitoring)
val_dataset = DefectDataset(image_dir, mask_dir, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

print(f"Dataset loaded: {len(train_dataset)} images")

model = smp.Unet(
    encoder_name="resnet34",        # Pretrained encoder (adapts to grayscale automatically)
    encoder_weights="imagenet",
    in_channels=1,                  # Grayscale input
    classes=1,                      # Binary segmentation
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Using device: {device}")

criterion = smp.losses.DiceLoss(mode='binary')  # Great for small/imbalanced defects
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

#training loop

num_epochs = 100
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0

    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # Validation phase
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks  = masks.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, masks).item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save best model → to Google Drive
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"    → New best model saved to Drive: {BEST_MODEL_PATH}")

print("\nTraining completed!")
print(f"Best model saved as: {BEST_MODEL_PATH}")
print(f"Best validation loss achieved: {best_val_loss:.4f}")

