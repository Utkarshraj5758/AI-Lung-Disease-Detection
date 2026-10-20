import os

base_path = "dataset"

print("Dataset root contains:", os.listdir(base_path))

for split in ["train", "val", "test"]:
    split_path = os.path.join(base_path, split)
    print(f"\nChecking {split_path}")
    print("Contains:", os.listdir(split_path))

    for category in ["NORMAL", "PNEUMONIA"]:
        path = os.path.join(split_path, category)
        files = os.listdir(path)
        print(category, "->", len(files))
