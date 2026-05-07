import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from data_loader import SODDataset
from sod_model import SODCNN


def tensor_to_image(tensor):
    tensor = tensor.detach().cpu().numpy()

    if tensor.shape[0] == 3:
        tensor = np.transpose(tensor, (1, 2, 0))
        tensor = (tensor * 255).astype(np.uint8)
    else:
        tensor = tensor.squeeze()
        tensor = (tensor * 255).astype(np.uint8)

    return tensor


def create_overlay(image, prediction):
    # image: H x W x 3
    # prediction: H x W

    overlay = image.copy()
    mask = prediction > 127

    # highlight prediction in red
    overlay[mask] = [255, 0, 0]

    return overlay


def visualize_predictions():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_images_dir = "data/test/images"
    test_masks_dir = "data/test/masks"
    model_path = "models/best_model_20.pth"

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    dataset = SODDataset(
        image_dir=test_images_dir,
        mask_dir=test_masks_dir,
        img_size=224
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = SODCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        for idx, (image, mask) in enumerate(loader):
            image = image.to(device)

            prediction = model(image)
            prediction = (prediction > 0.5).float()

            img_np = tensor_to_image(image[0])
            mask_np = tensor_to_image(mask[0])
            pred_np = tensor_to_image(prediction[0])

            overlay = create_overlay(img_np, pred_np)

            Image.fromarray(img_np).save(f"{output_dir}/image_{idx+1}.png")
            Image.fromarray(mask_np).save(f"{output_dir}/mask_{idx+1}.png")
            Image.fromarray(pred_np).save(
                f"{output_dir}/prediction_{idx+1}.png")
            Image.fromarray(overlay).save(f"{output_dir}/overlay_{idx+1}.png")

            if idx == 9:
                break

    print(f"Saved results in: {output_dir}")


if __name__ == "__main__":
    visualize_predictions()
