import torch
from torch.utils.data import DataLoader

from data_loader import SODDataset
from sod_model import SODCNN
from metrics import (
    compute_iou,
    compute_precision,
    compute_recall,
    compute_f1,
    compute_mae
)


def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_images_dir = "data/test/images"
    test_masks_dir = "data/test/masks"
    model_path = "models/best_model_20.pth"

    batch_size = 4
    img_size = 224

    test_dataset = SODDataset(
        image_dir=test_images_dir,
        mask_dir=test_masks_dir,
        img_size=img_size
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model = SODCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    total_iou = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_mae = 0
    total_batches = 0

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            total_iou += compute_iou(outputs, masks)
            total_precision += compute_precision(outputs, masks)
            total_recall += compute_recall(outputs, masks)
            total_f1 += compute_f1(outputs, masks)
            total_mae += compute_mae(outputs, masks)

            total_batches += 1

    if total_batches == 0:
        print("No test images found. Check data/test/images and data/test/masks.")
        return

    print("Evaluation Results")
    print("------------------")
    print(f"IoU:       {total_iou / total_batches:.4f}")
    print(f"Precision: {total_precision / total_batches:.4f}")
    print(f"Recall:    {total_recall / total_batches:.4f}")
    print(f"F1-score:  {total_f1 / total_batches:.4f}")
    print(f"MAE:       {total_mae / total_batches:.4f}")


if __name__ == "__main__":
    evaluate_model()
