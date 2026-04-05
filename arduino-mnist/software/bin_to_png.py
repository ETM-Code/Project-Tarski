#!/bin/env python3
"""
Convert .bin files back to PNG images.

The .bin files are 37 bytes: 36 int8 pixels for 6x6 image, last byte ignored.
Int8 range: -128 (black) to 127 (white).

Usage:
    python bin_to_png.py input.bin output.png
    python bin_to_png.py input_dir/ output_dir/ --batch
"""

import argparse
import struct
import sys
from pathlib import Path

from PIL import Image


def bin_to_png(bin_path: Path, png_path: Path):
    """Convert a single .bin file to PNG."""
    with open(bin_path, 'rb') as f:
        data = f.read(37)
        if len(data) != 37:
            raise ValueError(f"File {bin_path} is not 37 bytes long")
        
        # Read first 36 bytes as int8
        pixels = []
        for i in range(36):
            val = struct.unpack('b', data[i:i+1])[0]
            # Convert int8 [-128, 127] to uint8 [0, 255]
            pixel = max(0, min(255, val + 128))
            pixels.append(pixel)
        
        # Create 6x6 image
        img = Image.new('L', (6, 6))
        img.putdata(pixels)
        img.save(png_path)
        print(f"Converted {bin_path} to {png_path}")


def process_batch(input_dir: Path, output_dir: Path):
    """Process all .bin files in a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    bin_files = [f for f in input_dir.iterdir() if f.suffix.lower() == '.bin']

    if not bin_files:
        print(f"No .bin files found in {input_dir}")
        return

    for bin_path in sorted(bin_files):
        png_path = output_dir / (bin_path.stem + '.png')
        bin_to_png(bin_path, png_path)

    print(f"\nProcessed {len(bin_files)} files")


def main():
    parser = argparse.ArgumentParser(
        description="Convert .bin files back to PNG images"
    )
    parser.add_argument("input", type=Path, help="Input .bin file or directory")
    parser.add_argument("output", type=Path, help="Output .png file or directory")
    parser.add_argument(
        "--batch", "-b", action="store_true", help="Process entire directory"
    )

    args = parser.parse_args()

    if args.batch:
        if not args.input.is_dir():
            print(f"Error: {args.input} is not a directory", file=sys.stderr)
            sys.exit(1)
        process_batch(args.input, args.output)
    else:
        if not args.input.is_file():
            print(f"Error: {args.input} does not exist", file=sys.stderr)
            sys.exit(1)
        bin_to_png(args.input, args.output)


if __name__ == "__main__":
    main()
