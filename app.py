import os
import torch
from PIL import Image
import numpy as np

from src.sod_model import SODCNN


def preprocess_image(image_path, img_size=224):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((img_size, img_size))

    image = np.array(image) / 255.0
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    image = image.unsqueeze(0)

    return image


def save_prediction(pred, path):
    pred = pred.squeeze().detach().cpu().numpy()
    pred = (pred * 255).astype(np.uint8)
    Image.fromarray(pred).save(path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "models/best_model_20.pth"

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    model = SODCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    image_path = input("Enter image path (press Enter for default): ").strip()

    if image_path == "":
        image_path = "results/predictions/image_1.png"

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    print("Using image:", image_path)

    image = preprocess_image(image_path).to(device)

    with torch.no_grad():
        pred = model(image)
        pred = (pred > 0.5).float()

    output_path = "prediction_demo.png"
    save_prediction(pred, output_path)

    print(f"Prediction saved as {output_path}")


if __name__ == "__main__":
    main()
