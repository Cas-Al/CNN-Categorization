

import os
import shutil
import random

#  Configuration 
RAW_DIR    = "raw_dataset"
OUT_DIR    = "dataset"
CATEGORIES = ["comedy", "fashion", "gaming", "music"]
TEST_SPLIT = 0.2
RANDOM_SEED = 42


def prepare_dataset():
    random.seed(RANDOM_SEED)

    # Create output folders
    for split in ["train", "test"]:
        for cat in CATEGORIES:
            os.makedirs(os.path.join(OUT_DIR, split, cat), exist_ok=True)

    # Case-insensitive folder matching (handles Comedy, COMEDY, comedy etc.)
    raw_subdirs = {
        d.lower(): d
        for d in os.listdir(RAW_DIR)
        if os.path.isdir(os.path.join(RAW_DIR, d))
    }

    summary = {}

    for cat in CATEGORIES:
        src_folder_name = raw_subdirs.get(cat.lower())
        if src_folder_name is None:
            print(f"[WARNING] Folder '{cat}' not found in {RAW_DIR} — skipping.")
            continue

        src_folder = os.path.join(RAW_DIR, src_folder_name)

        images = [
            f for f in os.listdir(src_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]
        random.shuffle(images)

        split_idx  = int(len(images) * (1 - TEST_SPLIT))
        train_imgs = images[:split_idx]
        test_imgs  = images[split_idx:]

        for fname in train_imgs:
            shutil.copy(os.path.join(src_folder, fname),
                        os.path.join(OUT_DIR, "train", cat, fname))

        for fname in test_imgs:
            shutil.copy(os.path.join(src_folder, fname),
                        os.path.join(OUT_DIR, "test", cat, fname))

        summary[cat] = {"train": len(train_imgs), "test": len(test_imgs)}

    print("\n===== Dataset Preparation Complete =====")
    print(f"{'Category':<15} {'Train':>8} {'Test':>8} {'Total':>8}")
    print("-" * 42)
    for cat, counts in summary.items():
        total = counts["train"] + counts["test"]
        print(f"{cat:<15} {counts['train']:>8} {counts['test']:>8} {total:>8}")
    print(f"\nSaved to: {os.path.abspath(OUT_DIR)}/")


if __name__ == "__main__":
    prepare_dataset()
