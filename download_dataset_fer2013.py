#!/usr/bin/env python3
from pathlib import Path
import shutil
import kagglehub

CLASSES = ["happy", "sad"]
SPLITS = ["train", "test"]
OUT = Path("data/raw/fer2013")

def find_split_dir(root: Path, split: str) -> Path | None:
    for d in root.rglob(split):
        if d.is_dir() and all((d / c).is_dir() for c in CLASSES):
            return d
    return None

def main():
    src_root = Path(kagglehub.dataset_download("msambare/fer2013"))
    split_dirs = {s: find_split_dir(src_root, s) for s in SPLITS}
    if any(v is None for v in split_dirs.values()):
        raise SystemExit("Could not find expected 'train'/'test' with 'happy'/'sad'.")

    for split in SPLITS:
        for cls in CLASSES:
            src = split_dirs[split] / cls
            dst = OUT / split / cls
            shutil.copytree(src, dst, dirs_exist_ok=True)

    print(f"OK: fer2013 (happy/sad) → {OUT}")

if __name__ == "__main__":
    main()
