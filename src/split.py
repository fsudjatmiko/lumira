import os
import shutil
import random

def split_dataset(dataset_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # Ensure the ratios sum to 1
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"

    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get all class directories
    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        images = [f for f in os.listdir(cls_dir) if f.endswith('.jpg')]
        random.shuffle(images)

        train_count = int(len(images) * train_ratio)
        val_count = int(len(images) * val_ratio)
        test_count = len(images) - train_count - val_count

        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        test_images = images[train_count + val_count:]

        # Create class directories in output folders
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

        # Copy images to respective directories
        for img in train_images:
            shutil.copy(os.path.join(cls_dir, img), os.path.join(train_dir, cls, img))
        for img in val_images:
            shutil.copy(os.path.join(cls_dir, img), os.path.join(val_dir, cls, img))
        for img in test_images:
            shutil.copy(os.path.join(cls_dir, img), os.path.join(test_dir, cls, img))

        print(f"Class {cls}: {train_count} train, {val_count} val, {test_count} test")

if __name__ == "__main__":
    dataset_dir = 'dataset'
    output_dir = 'splitted'
    split_dataset(dataset_dir, output_dir)
