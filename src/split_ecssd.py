import random
import shutil
from pathlib import Path


image_dir = Path("data/raw/images")
mask_dir = Path("data/raw/masks")

for split in ["train", "val", "test"]:
    Path(f"data/{split}/images").mkdir(parents=True, exist_ok=True)
    Path(f"data/{split}/masks").mkdir(parents=True, exist_ok=True)

images = sorted(image_dir.glob("*.jpg"))

random.seed(42)
random.shuffle(images)

total = len(images)
train_end = int(0.70 * total)
val_end = int(0.85 * total)

train_files = images[:train_end]
val_files = images[train_end:val_end]
test_files = images[val_end:]


def copy_files(files, split):
    for img_path in files:
        mask_path = mask_dir / f"{img_path.stem}.png"

        if not mask_path.exists():
            print("Missing mask:", mask_path.name)
            continue

        shutil.copy(img_path, Path(f"data/{split}/images") / img_path.name)
        shutil.copy(mask_path, Path(f"data/{split}/masks") / mask_path.name)


copy_files(train_files, "train")
copy_files(val_files, "val")
copy_files(test_files, "test")

print("ECSSD split completed.")
print("Train:", len(train_files))
print("Val:", len(val_files))
print("Test:", len(test_files))
