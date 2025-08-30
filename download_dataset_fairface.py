#!/usr/bin/env python3
from pathlib import Path
import shutil
import kagglehub

OUT = Path("data/raw/fairface/FairFace")

def find_fairface_root(root: Path) -> Path | None:
    for cand in root.rglob("*"):
        if (cand / "train").is_dir() and (cand / "val").is_dir():
            return cand
    return None

def main():
    src_root = Path(kagglehub.dataset_download("aibloy/fairface"))
    ff_src = find_fairface_root(src_root)
    if ff_src is None:
        raise SystemExit("Could not locate FairFace train/ and val/ folders.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ff_src / "train", OUT / "train", dirs_exist_ok=True)
    shutil.copytree(ff_src / "val",   OUT / "val",   dirs_exist_ok=True)

    for csv in ("train_labels.csv", "val_labels.csv"):
        src_csv = ff_src / csv
        if src_csv.exists():
            shutil.copy2(src_csv, OUT / csv)

    print(f"OK: Copied FairFace to {OUT}")

if __name__ == "__main__":
    main()
