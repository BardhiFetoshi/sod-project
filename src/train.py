from sod_model import SODCNN
from data_loader import SODDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


train_dataset = SODDataset(
    image_dir="data/train/images",
    mask_dir="data/train/masks",
    img_size=224,
    augment=True
)

val_dataset = SODDataset(
    image_dir="data/val/images",
    mask_dir="data/val/masks",
    img_size=224,
    augment=False
)


train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)


model = SODCNN().to(device)


def dice_loss(pred, target):
    smooth = 1e-6
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def combined_loss(pred, target):
    bce = nn.BCELoss()(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
best_val_loss = float("inf")

Path("models").mkdir(exist_ok=True)


for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    print(f"\nStarting Epoch {epoch + 1}/{num_epochs}")

    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = combined_loss(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % 25 == 0:
            print(
                f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = combined_loss(outputs, masks)

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "models/best_model.pth")
        print("Saved best model.")

    print("-" * 40)


print("Training finished.")
