#!/usr/bin/env python3
"""
Test the Arduino MNIST emulator against the full MNIST test set.

Loads MNIST, downsamples to 6x6, converts to int8 (pixel - 128),
pipes each sample through the compiled emulate binary, and reports accuracy.

Usage:
    g++ -O2 -o emulate emulate.cpp
    python test_emulate.py
"""

import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
from torchvision import datasets, transforms
from PIL import Image

ANN_INPUT_SIZE = 36  # 6x6


def get_mnist_int8_test_data(data_dir: str = "./data"):
    """
    Load MNIST test set with the same preprocessing as training:
    Resize to 6x6, ToTensor (gives [0,1] float), then x * 255 - 128.
    Clamp to [-128, 127] and cast to int8.
    """
    transform = transforms.Compose([
        transforms.Resize((6, 6)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0 - 128.0),
        transforms.Lambda(lambda x: x.view(-1)),
    ])

    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    samples = []
    labels = []
    for img_tensor, label in test_dataset:
        # Round and clamp to int8 range, matching what the Arduino receives
        arr = img_tensor.numpy()
        arr = np.clip(np.round(arr), -128, 127).astype(np.int8)
        samples.append(arr)
        labels.append(label)

    return samples, labels


def main():
    emulate_bin = Path(__file__).parent / "emulate"

    if not emulate_bin.exists():
        print("Compiling emulate.cpp...")
        result = subprocess.run(
            ["g++", "-O2", "-o", str(emulate_bin), str(emulate_bin.with_suffix(".cpp"))],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"Compilation failed:\n{result.stderr}", file=sys.stderr)
            sys.exit(1)
        print("Compiled successfully.")

    print("Loading MNIST test set...")
    samples, labels = get_mnist_int8_test_data()
    n_samples = len(samples)
    print(f"Loaded {n_samples} test samples.")

    # Pack all samples into a single binary blob
    blob = b"".join(s.tobytes() for s in samples)

    print("Running inference through emulator...")
    proc = subprocess.run(
        [str(emulate_bin)],
        input=blob,
        capture_output=True,
    )

    if proc.returncode != 0:
        print(f"Emulator error:\n{proc.stderr.decode()}", file=sys.stderr)
        sys.exit(1)

    predictions = list(proc.stdout)

    if len(predictions) != n_samples:
        print(
            f"Expected {n_samples} predictions, got {len(predictions)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Tally accuracy
    correct = sum(p == l for p, l in zip(predictions, labels))
    accuracy = correct / n_samples * 100

    # Per-class breakdown
    class_correct = [0] * 10
    class_total = [0] * 10
    for p, l in zip(predictions, labels):
        class_total[l] += 1
        if p == l:
            class_correct[l] += 1

    print(f"\nResults: {correct}/{n_samples} correct ({accuracy:.2f}%)")
    print("\nPer-class accuracy:")
    for digit in range(10):
        if class_total[digit] > 0:
            pct = class_correct[digit] / class_total[digit] * 100
            print(f"  {digit}: {class_correct[digit]}/{class_total[digit]} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
