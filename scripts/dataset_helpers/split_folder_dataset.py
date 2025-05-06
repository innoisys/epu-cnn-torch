import os
import shutil
import random
import argparse

from tqdm import tqdm
from pathlib import Path


def user_arguments() -> argparse.Namespace:
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Split folder dataset into train/val/test')
    parser.add_argument('--dataset-src', type=str, required=True, help='Path to the src dataset')
    parser.add_argument('--dataset-dest', type=str, required=True, help='Path to the dest dataset')
    parser.add_argument('--train-proportion', type=float, required=False, help='Proportion of train set', default=0.7)
    parser.add_argument('--validation-proportion', type=float, required=False, help='Proportion of validation set', default=0.2)
    parser.add_argument('--test-proportion', type=float, required=False, help='Proportion of test set', default=0.1)
    args = parser.parse_args()
    return args


def count_images(path):
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    return sum(1 for p in Path(path).rglob("*")
                  if p.suffix.lower() in image_exts)


def sanity_check(dest_dir: str):
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

    if not os.path.exists(dest_dir):
        raise FileNotFoundError(f"Destination directory {dest_dir} does not exist")
    
    train_path = Path(dest_dir) / "train"
    all_train_files = [p for p in train_path.rglob("*") if p.suffix.lower() in image_exts]
    validation_path = Path(dest_dir) / "validation"
    all_validation_files = [p for p in validation_path.rglob("*") if p.suffix.lower() in image_exts]
    test_path = Path(dest_dir) / "test"
    all_test_files = [p for p in test_path.rglob("*") if p.suffix.lower() in image_exts]

    if len(all_train_files) == 0 or len(all_validation_files) == 0 or len(all_test_files) == 0:
        raise ValueError("No files found in the destination directory")
    
    for file in tqdm(all_train_files, desc="[+] Checking train files"):
        if file in all_validation_files or file in all_test_files:
            raise ValueError(f"File {file} is present in both train and validation/test sets")
        
    for file in tqdm(all_validation_files, desc="[+] Checking validation files"):
        if file in all_test_files:
            raise ValueError(f"File {file} is present in both validation and test sets")


def main():
    args = user_arguments()
    # Paths & proportions
    src_dir = args.dataset_src
    dest_dir = args.dataset_dest
    
    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"Source directory {src_dir} does not exist")

    # Make the destination directory
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    # Make the splits
    splits_proportions = {
        "train": args.train_proportion,
        "validation": args.validation_proportion,
        "test": args.test_proportion
    }

    splits = {
        "train": [],
        "validation": [],
        "test": []
    }

    # Make target folders
    for split in splits_proportions:
        for cls in tqdm(os.listdir(src_dir), desc="[+] Making target folders"):
            os.makedirs(os.path.join(dest_dir, split, cls), exist_ok=True)

    # For each class, shuffle & distribute
    for cls in tqdm(os.listdir(src_dir), desc="[+] Splitting classes"):
        cls_src = os.path.join(src_dir, cls)
        files = [f for f in os.listdir(cls_src) if f.lower().endswith(".jpg")]
        random.shuffle(files)
        n = len(files)
        
        # compute cut-offs
        train_end = int(splits_proportions["train"] * n)
        val_end  = train_end + int(splits_proportions["validation"] * n)

        splits = {
            "train": files[:train_end],
            "validation": files[train_end:val_end],
            "test": files[val_end:]
        }

        # copy into place
        for split, file_list in splits.items():
            for fname in tqdm(file_list, desc=f"[+] Copying {split} files"):
                src_path = os.path.join(cls_src, fname)
                dst_path = os.path.join(dest_dir, split, cls, fname)
                shutil.copy2(src_path, dst_path)

    sanity_check(dest_dir)
    print("[+] Done! Your data is split into train/val/test.")


if __name__ == "__main__":
    main()
