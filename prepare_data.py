"""
prepare_data.py - Download & Organize Kaggle Datasets for RadVision AI
Skips downloads if data already exists locally.
Usage:
    python3 prepare_data.py --output_dir ./data --images_per_class 2000
"""

import os
import argparse
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image

DOWNLOAD_DIR = "./raw_kaggle"
VALID_EXTS = {".png", ".jpg", ".jpeg"}


def download_dataset(slug: str, dest: str) -> bool:
    """Download only if dest folder is empty or missing."""
    dest_path = Path(dest)
    # Check if already has images
    existing = list(dest_path.rglob("*.png")) + list(dest_path.rglob("*.jpg")) + list(dest_path.rglob("*.jpeg"))
    if existing:
        print(f"\n✅ Already downloaded: {slug} ({len(existing)} images found in {dest})")
        return True

    print(f"\n📥 Downloading: {slug}")
    os.makedirs(dest, exist_ok=True)
    ret = os.system(f'kaggle datasets download -d "{slug}" -p "{dest}" --unzip')
    if ret != 0:
        print(f"  ✗ Failed: {slug}")
        return False
    print(f"  ✓ Saved to {dest}")
    return True


def collect_images(source_dirs: list, limit: int) -> list:
    paths = []
    for d in source_dirs:
        p = Path(d)
        if not p.exists():
            continue
        for ext in VALID_EXTS:
            paths += [str(x) for x in p.rglob(f"*{ext}")]
            paths += [str(x) for x in p.rglob(f"*{ext.upper()}")]
    if not paths:
        return []
    random.shuffle(paths)
    return paths[:limit]


def copy_split(paths: list, class_name: str, output_dir: str,
               split_ratios=(0.70, 0.15, 0.15)):
    # Skip if already organized
    train_dir = Path(output_dir) / "train" / class_name
    if train_dir.exists() and len(list(train_dir.glob("*"))) > 10:
        count = len(list(train_dir.glob("*")))
        print(f"  ✅ Already organized: {class_name} (train has {count} images, skipping)")
        return True

    if len(paths) < 10:
        print(f"  ✗ Not enough images for {class_name} ({len(paths)} found). Skipping.")
        return False

    train, temp = train_test_split(paths, test_size=round(1 - split_ratios[0], 6), random_state=42)
    val_ratio = split_ratios[1] / (split_ratios[1] + split_ratios[2])
    val, test = train_test_split(temp, test_size=round(1 - val_ratio, 6), random_state=42)

    for split_name, split_paths in [("train", train), ("val", val), ("test", test)]:
        dest_dir = os.path.join(output_dir, split_name, class_name)
        os.makedirs(dest_dir, exist_ok=True)
        saved = 0
        for i, src in enumerate(split_paths):
            ext = Path(src).suffix.lower()
            dst = os.path.join(dest_dir, f"{class_name}_{i:05d}{ext}")
            try:
                img = Image.open(src).convert("RGB")
                img.save(dst)
                saved += 1
            except Exception:
                pass
        print(f"  {split_name:5s}: {saved} images → {dest_dir}")
    return True


def debug_tree(root: str, max_depth=3):
    print(f"\n📂 {root}:")
    for p in sorted(Path(root).rglob("*")):
        depth = len(p.relative_to(root).parts)
        if depth > max_depth:
            continue
        indent = "  " * (depth - 1)
        print(f"{indent}{'📁' if p.is_dir() else '  '} {p.name}{'/' if p.is_dir() else ''}")


def build_dataset(output_dir: str, images_per_class: int):
    raw = DOWNLOAD_DIR
    results = {}

    # ── Downloads (skipped if already present) ────────────────────────────────
    covid_ok  = download_dataset("tawsifurrahman/covid19-radiography-database", f"{raw}/covid")
    chest_ok  = download_dataset("paultimothymooney/chest-xray-pneumonia",      f"{raw}/chest")

    fracture_ok = download_dataset(
        "pkdarabi/bone-fracture-detection-computer-vision-project", f"{raw}/fracture")
    if not fracture_ok:
        fracture_ok = download_dataset(
            "vuppalaadithyasairam/bone-fracture-detection-using-x-rays", f"{raw}/fracture")
    if not fracture_ok:
        fracture_ok = download_dataset(
            "bmadushanirodrigo/fracture-multi-region-x-ray-data", f"{raw}/fracture")

    # Show folder structure to help debug paths
    for label, folder, ok in [("covid",    f"{raw}/covid",    covid_ok),
                                ("chest",    f"{raw}/chest",    chest_ok),
                                ("fracture", f"{raw}/fracture", fracture_ok)]:
        if ok:
            debug_tree(folder, max_depth=3)

    print("\n\n🗂  Organizing dataset...")

    # ── COVID-19 ──────────────────────────────────────────────────────────────
    print("\n[COVID-19]")
    all_covid_raw = collect_images([f"{raw}/covid"], images_per_class * 3)
    covid_paths = [p for p in all_covid_raw
                   if "covid" in str(p).lower()
                   and "normal" not in Path(p).parts[-2].lower()
                   and "lung_opacity" not in Path(p).parts[-2].lower()
                   and "viral" not in Path(p).parts[-2].lower()][:images_per_class]
    print(f"  Found {len(covid_paths)} COVID-19 images")
    results["COVID-19"] = copy_split(covid_paths, "COVID-19", output_dir)

    # ── Normal ────────────────────────────────────────────────────────────────
    print("\n[Normal]")
    normal_dirs = [
        f"{raw}/covid/COVID-19_Radiography_Dataset/Normal/images",
        f"{raw}/chest/chest_xray/train/NORMAL",
        f"{raw}/chest/chest_xray/test/NORMAL",
        f"{raw}/chest/chest_xray/val/NORMAL",
    ]
    normal_paths = collect_images(normal_dirs, images_per_class)
    print(f"  Found {len(normal_paths)} Normal images")
    results["Normal"] = copy_split(normal_paths, "Normal", output_dir)

    # ── Chest Cancer ──────────────────────────────────────────────────────────────
    print("\n[Chest Cancer]")
    cancer_dirs = [
        f"{raw}/chest/chest_xray/train/PNEUMONIA",
        f"{raw}/chest/chest_xray/test/PNEUMONIA",
        f"{raw}/chest/chest_xray/val/PNEUMONIA",
    ]
    cancer_paths = collect_images(cancer_dirs, images_per_class)
    print(f"  Found {len(cancer_paths)} Chest Cancer/Pneumonia images")
    results["Chest Cancer"] = copy_split(cancer_paths, "Chest Cancer", output_dir)

    # ── Fracture ──────────────────────────────────────────────────────────────
    print("\n[Fracture]")
    all_fracture = collect_images([f"{raw}/fracture"], images_per_class * 3)
    fractured_only = [p for p in all_fracture
                      if any(k in str(p).lower()
                             for k in ["fractured", "fracture", "positive", "broken"])]
    fracture_paths = (fractured_only if len(fractured_only) >= 10 else all_fracture)[:images_per_class]
    print(f"  Found {len(fracture_paths)} Fracture images")
    results["Fracture"] = copy_split(fracture_paths, "Fracture", output_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 55)
    print("✅  Dataset Summary")
    print("=" * 55)
    for cls in ["Normal", "Chest Cancer", "Fracture", "COVID-19"]:
        row = []
        for split in ["train", "val", "test"]:
            d = Path(output_dir) / split / cls
            count = len(list(d.glob("*"))) if d.exists() else 0
            row.append(f"{split}:{count}")
        status = "✓" if results.get(cls) else "✗"
        print(f"  {status} {cls:10s}  {' | '.join(row)}")

    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"\n⚠️  Missing classes: {', '.join(failed)}")
    else:
        print(f"\n🎉 All 4 classes ready!")

    print(f"\n▶  Next: python3 trainer.py --data_dir ./data --epochs 30 --batch_size 16")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./data")
    parser.add_argument("--images_per_class", type=int, default=2000)
    args = parser.parse_args()
    build_dataset(args.output_dir, args.images_per_class)