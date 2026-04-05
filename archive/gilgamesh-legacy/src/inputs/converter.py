#!/usr/bin/env python3
"""
MNIST → 7x7 converter

Outputs per split (training_set, testing_set):
- converted/train_7x7.csv and test_7x7.csv        : CSV, header: label,p00,...,p66 (u8 0..255)
- converted/train_7x7.bin and test_7x7.bin        : Binary, simple header + u8 pixels + label
- converted/train_preview.png and test_preview.png: Visual grid of first N samples

Optionally, dump per-image PNGs:
- OG/        : original 28x28 greyscale digits as PNGs
- resized/   : 7x7 greyscale digits as PNGs (tiny), for human viewing use the preview PNGs

Usage:
    python converter.py
    python converter.py --preview-samples 400 --dump-images 200
    python converter.py --dump-images all   # careful, creates tens of thousands of PNGs

Directory layout expected (relative to this script):
.
├── converter.py
├── mnist
│   ├── train-images.idx3-ubyte
│   ├── train-labels.idx1-ubyte
│   ├── t10k-images.idx3-ubyte
│   └── t10k-labels.idx1-ubyte
└── {training_set, testing_set}/{OG, resized, converted}/

If your MNIST files also exist inside subfolders named like 'train-images-idx3-ubyte/train-images-idx3-ubyte',
the script will auto-detect those too.

Binary file format (little endian):
-  0..3   : magic bytes b"M7x7"
-  4..7   : u32 num_samples
-  8      : u8  rows (=7)
-  9      : u8  cols (=7)
- 10..11  : u16 reserved (=0)
- 12..    : for each sample: 49 bytes u8 pixels row-major, then 1 byte u8 label

Minimal Rust reader is included at the bottom of this file.
"""

import argparse
import os
import struct
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------- IDX readers ----------

def _pick_existing(base: Path, primary: str, fallback_dir: str) -> Path:
    """
    Prefer base/primary if it exists, otherwise try base/fallback_dir/primary.
    """
    p1 = base / primary
    if p1.exists():
        return p1
    p2 = base / fallback_dir / primary
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Could not find {primary} in {base} or {base/fallback_dir}")

def read_idx_images(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Bad magic for images in {path}, expected 2051, got {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape((num, rows, cols))
    return data

def read_idx_labels(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Bad magic for labels in {path}, expected 2049, got {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        if data.shape[0] != num:
            raise ValueError(f"Label count mismatch in {path}")
    return data


# ---------- Image processing ----------

def to_7x7(images28: np.ndarray) -> np.ndarray:
    """
    Downscale 28x28 to 7x7 using area resampling. Returns uint8 [0..255].
    """
    out = np.empty((images28.shape[0], 7, 7), dtype=np.uint8)
    for i, img in enumerate(images28):
        pil = Image.fromarray(img, mode="L")
        small = pil.resize((7, 7), Image.Resampling.BOX)
        out[i] = np.asarray(small, dtype=np.uint8)
    return out


# ---------- Writers ----------

def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def write_csv(csv_path: Path, images7: np.ndarray, labels: np.ndarray) -> None:
    """
    CSV header: label,p00,p01,...,p66
    """
    images_flat = images7.reshape(images7.shape[0], -1)
    header_cols = ["label"] + [f"p{r}{c}" for r in range(7) for c in range(7)]
    with open(csv_path, "w", newline="") as f:
        f.write(",".join(header_cols) + "\n")
        # Write in chunks to avoid huge strings
        for i in range(images_flat.shape[0]):
            row = [str(int(labels[i]))] + [str(int(v)) for v in images_flat[i]]
            f.write(",".join(row) + "\n")

def write_bin(bin_path: Path, images7: np.ndarray, labels: np.ndarray) -> None:
    """
    Custom compact binary, described in the module docstring.
    """
    n = images7.shape[0]
    rows, cols = images7.shape[1], images7.shape[2]
    with open(bin_path, "wb") as f:
        header = struct.pack("<4sIBBH", b"M7x7", n, rows, cols, 0)
        f.write(header)
        pixels = images7.reshape(n, rows * cols).astype(np.uint8)
        labs = labels.reshape(n, 1).astype(np.uint8)
        combined = np.concatenate([pixels, labs], axis=1)
        combined.tofile(f)

def save_previews(preview_path: Path, images7: np.ndarray, labels: np.ndarray,
                  max_samples: int = 200, cols: int = 20, scale: int = 16) -> None:
    """
    Create a grid image from first max_samples samples. Each 7x7 tile is scaled by 'scale'.
    """
    n = min(max_samples, images7.shape[0])
    if n == 0:
        return
    rows = math.ceil(n / cols)
    tile_w = 7 * scale
    tile_h = 7 * scale
    grid = Image.new("L", (cols * tile_w, rows * tile_h), color=255)

    # Try default font for small labels, if available
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for idx in range(n):
        r = idx // cols
        c = idx % cols
        tile = Image.fromarray(images7[idx], mode="L").resize((tile_w, tile_h), Image.Resampling.NEAREST)
        grid.paste(tile, (c * tile_w, r * tile_h))
        if font is not None:
            draw = ImageDraw.Draw(grid)
            # draw label in the top-left corner
            draw.text((c * tile_w + 2, r * tile_h + 2), str(int(labels[idx])), fill=0, font=font)

    grid.save(preview_path)

def dump_pngs(base_og: Path, base_resized: Path,
              images28: np.ndarray, images7: np.ndarray, labels: np.ndarray,
              limit: int | None) -> None:
    """
    Dump per-image PNGs into OG/ and resized/. Set limit=None to dump all.
    """
    total = images28.shape[0] if limit is None else min(limit, images28.shape[0])
    for i in range(total):
        label = int(labels[i])
        name = f"{i:06d}_{label}.png"

        img28 = Image.fromarray(images28[i], mode="L")
        img28.save(base_og / name)

        img7 = Image.fromarray(images7[i], mode="L")
        img7.save(base_resized / name)

def write_readme(readme_path: Path, split_name: str, csv_name: str, bin_name: str) -> None:
    content = f"""This folder contains converted {split_name} data.

Files:
- {csv_name} : CSV with header "label,p00,p01,...,p66", values are u8 in 0..255
- {bin_name} : Binary with header + interleaved pixels and labels as described below
- *_preview.png : Visual grid of the first samples for quick inspection

Binary format (little endian):
  offset  size  type    meaning
  0       4     bytes   magic = "M7x7"
  4       4     u32     num_samples
  8       1     u8      rows = 7
  9       1     u8      cols = 7
  10      2     u16     reserved = 0
  12      ...           for each sample: 49 bytes u8 pixels (row-major), then 1 byte u8 label

Minimal Rust reader:

use std::fs::File;
use std::io::{{Read, BufReader}};
use byteorder::{{LittleEndian, ReadBytesExt}};

fn read_m7x7(path: &str) -> std::io::Result<(usize, Vec<[u8; 49]>, Vec<u8>)> {{
    let f = File::open(path)?;
    let mut r = BufReader::new(f);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    assert_eq!(&magic, b"M7x7");

    let n = r.read_u32::<LittleEndian>()? as usize;
    let rows = r.read_u8()? as usize; assert_eq!(rows, 7);
    let cols = r.read_u8()? as usize; assert_eq!(cols, 7);
    let _reserved = r.read_u16::<LittleEndian>()?;

    let mut pixels = vec![[0u8; 49]; n];
    let mut labels = vec![0u8; n];

    for i in 0..n {{
        r.read_exact(&mut pixels[i])?;
        r.read_exact(&mut labels[i..i+1])?;
    }}

    Ok((n, pixels, labels))
}}
"""
    with open(readme_path, "w") as f:
        f.write(content)


# ---------- Main pipeline ----------

def main():
    parser = argparse.ArgumentParser(description="Convert MNIST to 7x7 CSV, BIN, and previews.")
    parser.add_argument("--mnist-dir", type=Path, default=Path("mnist"), help="Folder with IDX files")
    parser.add_argument("--training-dir", type=Path, default=Path("training_set"))
    parser.add_argument("--testing-dir", type=Path, default=Path("testing_set"))
    parser.add_argument("--preview-samples", type=int, default=200, help="Samples to include in preview grid")
    parser.add_argument("--preview-cols", type=int, default=20, help="Columns in preview grid")
    parser.add_argument("--preview-scale", type=int, default=16, help="Scaling factor for each 7x7 tile")
    parser.add_argument("--dump-images", default="0",
                        help="Dump per-image PNGs into OG/ and resized/. "
                             "Use a number like 200, or 'all', or '0' to disable.")
    args = parser.parse_args()

    # Resolve MNIST file paths with a little lenience for nested duplicates
    mnist_base = args.mnist_dir
    train_images_path = _pick_existing(mnist_base, "train-images.idx3-ubyte", "train-images-idx3-ubyte")
    train_labels_path = _pick_existing(mnist_base, "train-labels.idx1-ubyte", "train-labels-idx1-ubyte")
    test_images_path  = _pick_existing(mnist_base, "t10k-images.idx3-ubyte", "t10k-images-idx3-ubyte")
    test_labels_path  = _pick_existing(mnist_base, "t10k-labels.idx1-ubyte", "t10k-labels-idx1-ubyte")

    # Prepare output directories
    tr_og = args.training_dir / "OG"
    tr_resized = args.training_dir / "resized"
    tr_conv = args.training_dir / "converted"
    te_og = args.testing_dir / "OG"
    te_resized = args.testing_dir / "resized"
    te_conv = args.testing_dir / "converted"
    ensure_dirs(tr_og, tr_resized, tr_conv, te_og, te_resized, te_conv)

    # Read MNIST
    print("Reading MNIST training set...")
    Xtr28 = read_idx_images(train_images_path)
    ytr = read_idx_labels(train_labels_path)
    print(f"  Train: {Xtr28.shape[0]} samples")

    print("Reading MNIST test set...")
    Xte28 = read_idx_images(test_images_path)
    yte = read_idx_labels(test_labels_path)
    print(f"  Test : {Xte28.shape[0]} samples")

    # Convert to 7x7
    print("Downscaling to 7x7...")
    Xtr7 = to_7x7(Xtr28)
    Xte7 = to_7x7(Xte28)

    # Write converted formats
    print("Writing CSV and BIN...")
    write_csv(tr_conv / "train_7x7.csv", Xtr7, ytr)
    write_bin(tr_conv / "train_7x7.bin", Xtr7, ytr)
    write_csv(te_conv / "test_7x7.csv", Xte7, yte)
    write_bin(te_conv / "test_7x7.bin", Xte7, yte)

    # Previews
    print("Generating preview grids...")
    save_previews(tr_conv / "train_preview.png", Xtr7, ytr,
                  max_samples=args.preview_samples, cols=args.preview_cols, scale=args.preview_scale)
    save_previews(te_conv / "test_preview.png", Xte7, yte,
                  max_samples=args.preview_samples, cols=args.preview_cols, scale=args.preview_scale)

    # Optional: dump per-image PNGs
    limit_arg = args.dump_images.strip().lower()
    if limit_arg == "0":
        limit = 0
    elif limit_arg == "all":
        limit = None
    else:
        try:
            limit = int(limit_arg)
        except ValueError:
            raise SystemExit("--dump-images must be '0', 'all', or an integer")

    if limit is None or limit > 0:
        print(f"Dumping per-image PNGs (limit = {'all' if limit is None else limit})...")
        dump_pngs(tr_og, tr_resized, Xtr28, Xtr7, ytr, limit)
        dump_pngs(te_og, te_resized, Xte28, Xte7, yte, limit)
    else:
        print("Skipping per-image PNG dump. Use --dump-images N or --dump-images all to enable.")

    # README with Rust snippet and format notes
    write_readme(tr_conv / "README.txt", "training", "train_7x7.csv", "train_7x7.bin")
    write_readme(te_conv / "README.txt", "testing", "test_7x7.csv", "test_7x7.bin")

    print("Done. Nice and tidy. Time for tea.")


if __name__ == "__main__":
    main()


# ------------------ RUST SNIPPET (copy-paste friendly) ------------------
# Put in README too, but kept here for convenience.
r"""
use std::fs::File;
use std::io::{Read, BufReader};
use byteorder::{LittleEndian, ReadBytesExt};

fn read_m7x7(path: &str) -> std::io::Result<(usize, Vec<[u8; 49]>, Vec<u8>)> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    assert_eq!(&magic, b"M7x7");

    let n = r.read_u32::<LittleEndian>()? as usize;
    let rows = r.read_u8()? as usize; assert_eq!(rows, 7);
    let cols = r.read_u8()? as usize; assert_eq!(cols, 7);
    let _reserved = r.read_u16::<LittleEndian>()?;

    let mut pixels = vec![[0u8; 49]; n];
    let mut labels = vec![0u8; n];

    for i in 0..n {
        r.read_exact(&mut pixels[i])?;
        r.read_exact(&mut labels[i..i+1])?;
    }

    Ok((n, pixels, labels))
}
"""
