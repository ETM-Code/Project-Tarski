#!/usr/bin/env python3
"""
SNN comparison using snnTorch to establish baseline accuracy.
Matches Gilgamesh architecture as closely as possible:
- 7x7 downsampled MNIST (49 input features)
- 100 hidden spiking neurons
- 10 output neurons
- ~15 epochs
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters matching Gilgamesh
BATCH_SIZE = 128
NUM_EPOCHS = 15
NUM_STEPS = 25  # Temporal steps (similar to our row-by-row presentation)
HIDDEN_SIZE = 100
LR = 1e-3
BETA = 0.9  # Membrane decay (similar to our tau)

def get_dataloaders():
    """Load MNIST downsampled to 7x7"""
    transform = transforms.Compose([
        transforms.Resize((7, 7)),  # Downsample to 7x7
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader


class SpikingNet(nn.Module):
    """
    Simple feedforward SNN matching Gilgamesh architecture:
    Input(49) -> Hidden(100, spiking) -> Output(10, spiking)
    """
    def __init__(self):
        super().__init__()

        # Surrogate gradient (similar to our superspike-style gradient)
        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Layers - input is 7 (one row at a time, like Gilgamesh)
        self.fc1 = nn.Linear(7, HIDDEN_SIZE)
        self.lif1 = snn.Leaky(beta=BETA, spike_grad=spike_grad)

        self.fc2 = nn.Linear(HIDDEN_SIZE, 10)
        self.lif2 = snn.Leaky(beta=BETA, spike_grad=spike_grad)

    def forward(self, x):
        # Initialize hidden states
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record output spikes
        spk2_rec = []
        mem2_rec = []

        # Time loop - present each row as a timestep (like Gilgamesh)
        # x shape: [batch, 1, 7, 7]
        batch_size = x.size(0)
        x = x.view(batch_size, 7, 7)  # [batch, rows, cols]

        for step in range(7):  # 7 rows = 7 timesteps
            # Get this row's pixels as input
            row_input = x[:, step, :]  # [batch, 7]

            # Repeat for multiple steps per row to allow temporal dynamics
            for _ in range(NUM_STEPS // 7):
                cur1 = self.fc1(row_input)
                spk1, mem1 = self.lif1(cur1, mem1)

                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)

                spk2_rec.append(spk2)
                mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


class SpikingNetRateCoded(nn.Module):
    """
    Alternative: Rate-coded input (more standard for SNNs)
    Presents all 49 pixels at each timestep
    """
    def __init__(self):
        super().__init__()

        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.fc1 = nn.Linear(49, HIDDEN_SIZE)
        self.lif1 = snn.Leaky(beta=BETA, spike_grad=spike_grad)

        self.fc2 = nn.Linear(HIDDEN_SIZE, 10)
        self.lif2 = snn.Leaky(beta=BETA, spike_grad=spike_grad)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []
        mem2_rec = []

        # Flatten image
        x = x.view(x.size(0), -1)  # [batch, 49]

        for _ in range(NUM_STEPS):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


def train(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        spk_rec, mem_rec = model(data)

        # Loss: Cross-entropy on spike count
        loss = nn.CrossEntropyLoss()(spk_rec.sum(dim=0), target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Accuracy: class with most spikes
        _, predicted = spk_rec.sum(dim=0).max(1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    acc = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Train Acc: {acc:.2f}%")
    return avg_loss, acc


def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            spk_rec, _ = model(data)

            _, predicted = spk_rec.sum(dim=0).max(1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    acc = 100. * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    return acc


def main():
    parser = argparse.ArgumentParser(description='snnTorch MNIST comparison')
    parser.add_argument('--temporal', action='store_true',
                        help='Use temporal row-by-row encoding (like Gilgamesh)')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)

    print(f"=== snnTorch MNIST Comparison ===")
    print(f"Seed: {args.seed}")
    print(f"Hidden neurons: {HIDDEN_SIZE}")
    print(f"Encoding: {'Temporal (row-by-row)' if args.temporal else 'Rate-coded'}")
    print(f"Epochs: {args.epochs}")
    print()

    train_loader, test_loader = get_dataloaders()

    if args.temporal:
        model = SpikingNet().to(device)
    else:
        model = SpikingNetRateCoded().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_accs = []
    test_accs = []

    for epoch in range(1, args.epochs + 1):
        _, train_acc = train(model, train_loader, optimizer, epoch)
        test_acc = test(model, test_loader)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        print()

    print(f"=== Final Results ===")
    print(f"Best Test Accuracy: {max(test_accs):.2f}%")
    print(f"Final Test Accuracy: {test_accs[-1]:.2f}%")

    # Save plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label='Train')
    plt.plot(test_accs, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'snnTorch MNIST (7x7, {HIDDEN_SIZE} hidden)')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_plot.png')
    print("Saved accuracy_plot.png")


if __name__ == '__main__':
    main()
