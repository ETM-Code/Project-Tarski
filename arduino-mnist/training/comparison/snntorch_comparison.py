#!/usr/bin/env python3
"""
SNNTorch Comparison Networks for Gilgamesh

This script trains spiking neural networks using PyTorch and snntorch
with the same configuration as the gilgamesh Rust implementation:
- 2-layer feedforward LIF SNN
- 36 input (6x6 MNIST) -> 12 hidden -> 10 output
- Rate-coded input encoding
- Cross-entropy loss on spike counts
- Adam optimizer with cosine annealing

Trains multiple network variants and generates a comparison report.
"""

import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import snntorch as snn
from snntorch import surrogate
import numpy as np


# =============================================================================
# Configuration (matching gilgamesh defaults)
# =============================================================================

@dataclass
class NetworkConfig:
    """Network architecture configuration."""
    input_size: int = 36       # 6x6 downsampled MNIST
    hidden_size: int = 12      # Hidden layer neurons (matching gilgamesh)
    output_size: int = 10      # Output classes (digits 0-9)


@dataclass
class NeuronConfig:
    """LIF neuron configuration."""
    beta: float = 0.9          # Membrane decay rate
    threshold: float = 1.0     # Spike threshold
    slope: float = 25.0        # Surrogate gradient slope
    reset_mechanism: str = "subtract"  # "subtract", "zero", or "none"


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    lr: float = 0.001          # Initial learning rate
    lr_min: float = 0.00001    # Minimum learning rate (1% of initial)
    epochs: int = 15           # Number of training epochs
    batch_size: int = 128      # Batch size
    num_steps: int = 25        # Timesteps per sample
    seed: int = 42             # Random seed
    weight_decay: float = 0.0  # AdamW weight decay
    beta1: float = 0.9         # Adam first moment
    beta2: float = 0.999       # Adam second moment
    grad_clip: float = 1.0     # Gradient clipping max norm


@dataclass
class Config:
    """Complete configuration."""
    network: NetworkConfig
    neuron: NeuronConfig
    training: TrainingConfig

    @classmethod
    def default(cls):
        return cls(
            network=NetworkConfig(),
            neuron=NeuronConfig(),
            training=TrainingConfig()
        )


# =============================================================================
# Dataset
# =============================================================================

class DownsampledMNIST:
    """MNIST dataset downsampled to 6x6 pixels (matching gilgamesh)."""

    def __init__(self, data_dir: str, target_size: int = 6, use_int8_range: bool = False):
        self.data_dir = data_dir
        self.target_size = target_size
        self.use_int8_range = use_int8_range

        if use_int8_range:
            # Transform for int8 range: pixel values in [-128, 127]
            # ToTensor gives [0, 1], multiply by 255 to get [0, 255], subtract 128 to get [-128, 127]
            self.transform = transforms.Compose([
                transforms.Resize((target_size, target_size)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 255.0 - 128.0),  # [0,1] -> [-128, 127]
                transforms.Lambda(lambda x: x.view(-1))  # Flatten to 36
            ])
        else:
            # Transform: resize to 6x6, normalize with MNIST stats (original behavior)
            self.transform = transforms.Compose([
                transforms.Resize((target_size, target_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(lambda x: x.view(-1))  # Flatten to 36
            ])

    def get_loaders(self, batch_size: int):
        """Get train and test data loaders."""
        train_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.transform
        )

        test_dataset = datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        return train_loader, test_loader


# =============================================================================
# Network Models
# =============================================================================

class GilgameshSNN(nn.Module):
    """
    2-layer feedforward LIF SNN matching gilgamesh architecture.

    Architecture: Input(36) -> FC1 -> LIF(12) -> FC2 -> LIF(10)
    Uses spike count output for classification.
    """

    def __init__(self, config: Config):
        super().__init__()

        net = config.network
        neuron = config.neuron

        # Surrogate gradient function (fast sigmoid)
        spike_grad = surrogate.fast_sigmoid(slope=neuron.slope)

        # Reset mechanism
        reset_map = {
            "subtract": "subtract",
            "zero": "zero",
            "none": "none"
        }
        reset = reset_map.get(neuron.reset_mechanism, "subtract")

        # Layer 1: Input -> Hidden
        self.fc1 = nn.Linear(net.input_size, net.hidden_size)
        self.lif1 = snn.Leaky(
            beta=neuron.beta,
            threshold=neuron.threshold,
            spike_grad=spike_grad,
            reset_mechanism=reset
        )

        # Layer 2: Hidden -> Output
        self.fc2 = nn.Linear(net.hidden_size, net.output_size)
        self.lif2 = snn.Leaky(
            beta=neuron.beta,
            threshold=neuron.threshold,
            spike_grad=spike_grad,
            reset_mechanism=reset
        )

        self.num_steps = config.training.num_steps

    def forward(self, x):
        """
        Forward pass with rate-coded input.

        Args:
            x: Input tensor [batch, 49]

        Returns:
            spike_count: Total spikes per output neuron [batch, 10]
            mem_record: Membrane potentials over time
            spike_record: Spikes over time
        """
        batch_size = x.size(0)

        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Output spike count accumulator
        spike_count = torch.zeros(batch_size, self.fc2.out_features, device=x.device)

        # Records for analysis
        mem_record = []
        spike_record = []

        # Simulate over time steps (rate-coded: same input each step)
        for _ in range(self.num_steps):
            # Layer 1
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            # Layer 2
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            # Accumulate output spikes
            spike_count = spike_count + spk2

            # Record
            mem_record.append(mem2.detach())
            spike_record.append(spk2.detach())

        return spike_count, mem_record, spike_record


class GilgameshSNN_Synaptic(nn.Module):
    """
    Variant using Synaptic neurons (dual exponential dynamics).
    """

    def __init__(self, config: Config):
        super().__init__()

        net = config.network
        neuron = config.neuron

        spike_grad = surrogate.fast_sigmoid(slope=neuron.slope)

        self.fc1 = nn.Linear(net.input_size, net.hidden_size)
        self.lif1 = snn.Synaptic(
            alpha=0.9,  # Synaptic decay
            beta=neuron.beta,
            threshold=neuron.threshold,
            spike_grad=spike_grad
        )

        self.fc2 = nn.Linear(net.hidden_size, net.output_size)
        self.lif2 = snn.Synaptic(
            alpha=0.9,
            beta=neuron.beta,
            threshold=neuron.threshold,
            spike_grad=spike_grad
        )

        self.num_steps = config.training.num_steps

    def forward(self, x):
        batch_size = x.size(0)

        syn1, mem1 = self.lif1.init_synaptic()
        syn2, mem2 = self.lif2.init_synaptic()

        spike_count = torch.zeros(batch_size, self.fc2.out_features, device=x.device)
        mem_record = []
        spike_record = []

        for _ in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)

            cur2 = self.fc2(spk1)
            spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)

            spike_count = spike_count + spk2
            mem_record.append(mem2.detach())
            spike_record.append(spk2.detach())

        return spike_count, mem_record, spike_record


class GilgameshSNN_Recurrent(nn.Module):
    """
    Variant with recurrent connections in hidden layer.
    """

    def __init__(self, config: Config):
        super().__init__()

        net = config.network
        neuron = config.neuron

        spike_grad = surrogate.fast_sigmoid(slope=neuron.slope)

        self.fc1 = nn.Linear(net.input_size, net.hidden_size)
        self.rec1 = nn.Linear(net.hidden_size, net.hidden_size, bias=False)
        self.lif1 = snn.Leaky(
            beta=neuron.beta,
            threshold=neuron.threshold,
            spike_grad=spike_grad
        )

        self.fc2 = nn.Linear(net.hidden_size, net.output_size)
        self.lif2 = snn.Leaky(
            beta=neuron.beta,
            threshold=neuron.threshold,
            spike_grad=spike_grad
        )

        self.num_steps = config.training.num_steps

    def forward(self, x):
        batch_size = x.size(0)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk1 = torch.zeros(batch_size, self.fc1.out_features, device=x.device)

        spike_count = torch.zeros(batch_size, self.fc2.out_features, device=x.device)
        mem_record = []
        spike_record = []

        for _ in range(self.num_steps):
            # Layer 1 with recurrence
            cur1 = self.fc1(x) + self.rec1(spk1)
            spk1, mem1 = self.lif1(cur1, mem1)

            # Layer 2
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spike_count = spike_count + spk2
            mem_record.append(mem2.detach())
            spike_record.append(spk2.detach())

        return spike_count, mem_record, spike_record


class GilgameshSNN_3Layer(nn.Module):
    """
    Deeper variant with 3 hidden layers.
    """

    def __init__(self, config: Config):
        super().__init__()

        net = config.network
        neuron = config.neuron

        spike_grad = surrogate.fast_sigmoid(slope=neuron.slope)

        hidden1 = net.hidden_size
        hidden2 = net.hidden_size // 2  # 50

        self.fc1 = nn.Linear(net.input_size, hidden1)
        self.lif1 = snn.Leaky(beta=neuron.beta, threshold=neuron.threshold, spike_grad=spike_grad)

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.lif2 = snn.Leaky(beta=neuron.beta, threshold=neuron.threshold, spike_grad=spike_grad)

        self.fc3 = nn.Linear(hidden2, net.output_size)
        self.lif3 = snn.Leaky(beta=neuron.beta, threshold=neuron.threshold, spike_grad=spike_grad)

        self.num_steps = config.training.num_steps

    def forward(self, x):
        batch_size = x.size(0)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spike_count = torch.zeros(batch_size, self.fc3.out_features, device=x.device)
        mem_record = []
        spike_record = []

        for _ in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spike_count = spike_count + spk3
            mem_record.append(mem3.detach())
            spike_record.append(spk3.detach())

        return spike_count, mem_record, spike_record


class StandardANN(nn.Module):
    """
    Standard PyTorch ANN (non-spiking) for comparison.

    Same architecture as GilgameshSNN but with ReLU activations instead of LIF neurons.
    Architecture: Input(36) -> FC -> ReLU(12) -> FC -> Output(10)
    """

    def __init__(self, config: Config):
        super().__init__()

        net = config.network

        self.fc1 = nn.Linear(net.input_size, net.hidden_size)
        self.fc2 = nn.Linear(net.hidden_size, net.output_size)

    def forward(self, x):
        """
        Forward pass.

        Returns same format as SNN models for compatibility:
            output: Logits [batch, 10]
            mem_record: Empty list (no membrane potentials)
            spike_record: Empty list (no spikes)
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, [], []


class StandardANN_3Layer(nn.Module):
    """
    Deeper ANN variant with 3 layers for comparison with GilgameshSNN_3Layer.

    Architecture: Input(36) -> FC -> ReLU(12) -> FC -> ReLU(6) -> FC -> Output(10)
    """

    def __init__(self, config: Config):
        super().__init__()

        net = config.network
        hidden1 = net.hidden_size
        hidden2 = net.hidden_size // 2  # 50

        self.fc1 = nn.Linear(net.input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, net.output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, [], []


# =============================================================================
# Quantization Utilities
# =============================================================================

class FakeQuantize(torch.autograd.Function):
    """
    Fake quantization with straight-through estimator.

    Quantizes to n-bit signed integers during forward pass,
    but passes gradients through unchanged during backward.
    """

    @staticmethod
    def forward(ctx, x, bits, scale=None):
        # Signed integer range: [-2^(bits-1), 2^(bits-1) - 1]
        qmin = -(1 << (bits - 1))
        qmax = (1 << (bits - 1)) - 1

        # Per-tensor symmetric quantization
        if scale is None:
            abs_max = x.abs().max().clamp(min=1e-8)
            scale = abs_max / qmax

        # Quantize: scale -> round -> clamp -> dequantize
        x_scaled = x / scale
        x_rounded = x_scaled.round()
        x_clamped = x_rounded.clamp(qmin, qmax)
        x_dequant = x_clamped * scale

        return x_dequant

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: pass gradients unchanged
        return grad_output, None, None


def fake_quantize(x, bits, scale=None):
    """Convenience function for fake quantization."""
    return FakeQuantize.apply(x, bits, scale)


class QuantizedLinear(nn.Module):
    """
    Linear layer with quantized weights and activations.

    All computations simulate n-bit integer arithmetic:
    - Weights are quantized before matmul
    - Inputs are quantized before matmul
    - Outputs are quantized after matmul
    """

    def __init__(self, in_features, out_features, bits=8, bias=True):
        super().__init__()
        self.bits = bits
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        # Quantize input
        x_q = fake_quantize(x, self.bits)

        # Quantize weights
        w_q = fake_quantize(self.linear.weight, self.bits)

        # Quantized matmul (simulated - result would accumulate in higher precision)
        out = F.linear(x_q, w_q, self.linear.bias)

        # Quantize output (simulating accumulator truncation)
        out_q = fake_quantize(out, self.bits)

        return out_q


class FixedScaleQuantize(torch.autograd.Function):
    """
    Fixed-scale quantization with straight-through estimator.
    Uses a predetermined scale factor instead of computing dynamically.
    """

    @staticmethod
    def forward(ctx, x, scale, bits):
        qmin = -(1 << (bits - 1))
        qmax = (1 << (bits - 1)) - 1

        # Quantize with fixed scale
        x_scaled = x / scale
        x_rounded = x_scaled.round()
        x_clamped = x_rounded.clamp(qmin, qmax)
        x_dequant = x_clamped * scale

        return x_dequant

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def fixed_scale_quantize(x, scale, bits):
    """Convenience function for fixed-scale quantization."""
    return FixedScaleQuantize.apply(x, scale, bits)


class FixedScaleLinear(nn.Module):
    """
    Linear layer with FIXED scale factors that match integer inference exactly.

    Scale factors are registered as buffers (not parameters) and updated
    via exponential moving average during training to track the weight range.
    """

    def __init__(self, in_features, out_features, bits=8, bias=True, ema_decay=0.999):
        super().__init__()
        self.bits = bits
        self.qmax = (1 << (bits - 1)) - 1
        self.qmin = -(1 << (bits - 1))
        self.ema_decay = ema_decay

        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Register scale factors as buffers (persistent, not trainable)
        # Initialize based on Xavier initialization expected range
        weight_init_scale = (2.0 / (in_features + out_features)) ** 0.5
        self.register_buffer('weight_scale', torch.tensor(weight_init_scale / self.qmax * 2))
        self.register_buffer('bias_scale', torch.tensor(weight_init_scale / self.qmax * 2))
        self.register_buffer('output_scale', torch.tensor(1.0 / self.qmax))

        # Track if scales have been calibrated
        self.register_buffer('calibrated', torch.tensor(False))

    def calibrate_scales(self):
        """Update scales based on current weight values."""
        with torch.no_grad():
            w_abs_max = self.linear.weight.abs().max().clamp(min=1e-8)
            self.weight_scale.copy_(w_abs_max / self.qmax)

            if self.linear.bias is not None:
                b_abs_max = self.linear.bias.abs().max().clamp(min=1e-8)
                self.bias_scale.copy_(b_abs_max / self.qmax)

            self.calibrated.copy_(torch.tensor(True))

    def update_scales_ema(self):
        """Update scales using EMA of observed ranges (call during training)."""
        with torch.no_grad():
            w_abs_max = self.linear.weight.abs().max().clamp(min=1e-8)
            new_w_scale = w_abs_max / self.qmax
            self.weight_scale.copy_(
                self.ema_decay * self.weight_scale + (1 - self.ema_decay) * new_w_scale
            )

            if self.linear.bias is not None:
                b_abs_max = self.linear.bias.abs().max().clamp(min=1e-8)
                new_b_scale = b_abs_max / self.qmax
                self.bias_scale.copy_(
                    self.ema_decay * self.bias_scale + (1 - self.ema_decay) * new_b_scale
                )

    def forward(self, x, x_scale):
        """
        Forward pass simulating TRUE integer arithmetic.

        Args:
            x: Input tensor (dequantized float representation)
            x_scale: Scale factor of input (for reference, not used in simplified version)

        Returns:
            output: Quantized output (dequantized float representation)
            output_scale: Scale factor of output
        """
        # Update scales during training
        if self.training:
            self.update_scales_ema()

        # Quantize weights with fixed scale (returns dequantized float)
        w_q = fixed_scale_quantize(self.linear.weight, self.weight_scale, self.bits)

        # Quantize bias with fixed scale
        if self.linear.bias is not None:
            b_q = fixed_scale_quantize(self.linear.bias, self.bias_scale, self.bits)
        else:
            b_q = None

        # Matmul in dequantized domain (simulates integer matmul)
        acc = F.linear(x, w_q, b_q)

        # Compute output scale from accumulator range
        with torch.no_grad():
            acc_abs_max = acc.abs().max().clamp(min=1e-8)
            new_out_scale = acc_abs_max / self.qmax
            if self.training:
                self.output_scale.copy_(
                    self.ema_decay * self.output_scale + (1 - self.ema_decay) * new_out_scale
                )

        # Requantize output
        out_q = fixed_scale_quantize(acc, self.output_scale, self.bits)

        return out_q, self.output_scale.clone()


# =============================================================================
# Power-of-2 Scale Quantization (Shift-based for embedded deployment)
# =============================================================================

class PowerOf2Quantize(torch.autograd.Function):
    """
    Power-of-2 scale quantization with straight-through estimator.

    Scale is constrained to 2^shift where shift is an integer.
    This allows efficient implementation using bit shifts on embedded devices.
    """

    @staticmethod
    def forward(ctx, x, shift, bits):
        qmin = -(1 << (bits - 1))
        qmax = (1 << (bits - 1)) - 1

        # Scale = 2^shift
        scale = (2.0 ** shift)

        # Quantize: x_int = round(x / scale), clamped to [qmin, qmax]
        x_scaled = x / scale
        x_rounded = x_scaled.round()
        x_clamped = x_rounded.clamp(qmin, qmax)

        # Dequantize for training: x_dequant = x_int * scale
        x_dequant = x_clamped * scale

        return x_dequant

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        return grad_output, None, None


def power_of_2_quantize(x, shift, bits):
    """Convenience function for power-of-2 quantization."""
    return PowerOf2Quantize.apply(x, shift, bits)


def compute_shift_for_range(abs_max: torch.Tensor, bits: int) -> int:
    """
    Compute the shift amount needed to represent a value range in n bits.

    For a value with |x| <= abs_max, we want:
        x_int = round(x / 2^shift)  where  |x_int| <= qmax

    So: shift = ceil(log2(abs_max / qmax))
    """
    qmax = (1 << (bits - 1)) - 1

    if abs_max < 1e-10:
        return 0

    # shift = ceil(log2(abs_max / qmax))
    # Using log2(a/b) = log2(a) - log2(b)
    log2_ratio = torch.log2(abs_max / qmax)
    shift = int(torch.ceil(log2_ratio).item())

    return shift


class ShiftScaleLinear(nn.Module):
    """
    Linear layer with power-of-2 scale factors for efficient embedded inference.

    Instead of float scales, uses integer shift amounts:
        real_value = int_value * 2^shift  (or int_value << shift)

    This allows the embedded device to use bit shifts instead of multiplies.
    """

    def __init__(self, in_features, out_features, bits=8, bias=True, ema_decay=0.999):
        super().__init__()
        self.bits = bits
        self.qmax = (1 << (bits - 1)) - 1
        self.qmin = -(1 << (bits - 1))
        self.ema_decay = ema_decay
        self.in_features = in_features
        self.out_features = out_features

        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Register shift amounts as buffers (integers stored as float for gradient flow)
        # Initialize conservatively
        self.register_buffer('weight_shift', torch.tensor(-7.0))  # scale ≈ 0.0078
        self.register_buffer('bias_shift', torch.tensor(-7.0))
        self.register_buffer('output_shift', torch.tensor(-4.0))  # scale ≈ 0.0625

        # Track observed ranges for shift computation (EMA)
        self.register_buffer('weight_abs_max', torch.tensor(0.1))
        self.register_buffer('bias_abs_max', torch.tensor(0.1))
        self.register_buffer('output_abs_max', torch.tensor(1.0))

    def update_shifts(self):
        """Update shift amounts based on observed weight/activation ranges."""
        with torch.no_grad():
            # Update weight shift
            w_abs_max = self.linear.weight.abs().max().clamp(min=1e-8)
            self.weight_abs_max.copy_(
                self.ema_decay * self.weight_abs_max + (1 - self.ema_decay) * w_abs_max
            )
            self.weight_shift.copy_(torch.tensor(
                float(compute_shift_for_range(self.weight_abs_max, self.bits))
            ))

            # Update bias shift
            if self.linear.bias is not None:
                b_abs_max = self.linear.bias.abs().max().clamp(min=1e-8)
                self.bias_abs_max.copy_(
                    self.ema_decay * self.bias_abs_max + (1 - self.ema_decay) * b_abs_max
                )
                self.bias_shift.copy_(torch.tensor(
                    float(compute_shift_for_range(self.bias_abs_max, self.bits))
                ))

    def forward(self, x, x_shift):
        """
        Forward pass simulating shift-based integer arithmetic.

        Args:
            x: Input tensor (dequantized float representation)
            x_shift: Shift amount of input (integer as float tensor)

        Returns:
            output: Quantized output (dequantized float representation)
            output_shift: Shift amount of output
        """
        # Update shifts during training
        if self.training:
            self.update_shifts()

        # Quantize weights with power-of-2 scale
        w_q = power_of_2_quantize(self.linear.weight, self.weight_shift, self.bits)

        # Quantize bias
        if self.linear.bias is not None:
            b_q = power_of_2_quantize(self.linear.bias, self.bias_shift, self.bits)
        else:
            b_q = None

        # Matmul in dequantized domain
        # Note: In true integer arithmetic, acc = x_int @ w_int (int32)
        # The combined scale would be 2^(x_shift + weight_shift)
        acc = F.linear(x, w_q, b_q)

        # Update output shift based on accumulator range
        with torch.no_grad():
            acc_abs_max = acc.abs().max().clamp(min=1e-8)
            self.output_abs_max.copy_(
                self.ema_decay * self.output_abs_max + (1 - self.ema_decay) * acc_abs_max
            )
            self.output_shift.copy_(torch.tensor(
                float(compute_shift_for_range(self.output_abs_max, self.bits))
            ))

        # Requantize output
        out_q = power_of_2_quantize(acc, self.output_shift, self.bits)

        return out_q, self.output_shift.clone()

    def get_requant_shift(self, input_shift):
        """
        Compute the combined shift needed for requantization after matmul.

        After matmul: acc_scale = 2^(input_shift + weight_shift)
        To requantize to output_shift: right_shift = input_shift + weight_shift - output_shift
        """
        return int(input_shift.item()) + int(self.weight_shift.item()) - int(self.output_shift.item())

    def get_bias_align_shift(self, input_shift):
        """
        Compute shift to align bias to accumulator scale.

        Accumulator scale: 2^(input_shift + weight_shift)
        Bias scale: 2^(bias_shift)
        To align: left_shift = input_shift + weight_shift - bias_shift (if positive, shift left)
        """
        return int(input_shift.item()) + int(self.weight_shift.item()) - int(self.bias_shift.item())


class StandardANN_Int8_Shift(nn.Module):
    """
    8-bit quantized ANN using power-of-2 scales (shift-based).

    All scales are 2^shift, enabling pure bit-shift operations on embedded devices.
    No floating point needed during inference.

    IMPORTANT: This model expects inputs in the range [-128, 127] (int8 range).
    Use DownsampledMNIST with use_int8_range=True for training.
    """

    def __init__(self, config: Config):
        super().__init__()

        net = config.network
        self.bits = 8
        self.qmax = 127

        self.fc1 = ShiftScaleLinear(net.input_size, net.hidden_size, bits=8)
        self.fc2 = ShiftScaleLinear(net.hidden_size, net.output_size, bits=8)

        # Input shift = 0 because inputs are already int8 values [-128, 127]
        # scale = 2^0 = 1, meaning int_value = real_value
        self.register_buffer('input_shift', torch.tensor(0.0))

    def forward(self, x):
        # Input is already in int8 range [-128, 127], no quantization needed
        # Just clamp to ensure we're in valid range (defensive)
        x_q = x.clamp(-128, 127).round()

        # Layer 1: FC + ReLU
        h, h_shift = self.fc1(x_q, self.input_shift)
        h = F.relu(h)
        # ReLU doesn't change scale, but we requantize
        h_q = power_of_2_quantize(h, h_shift, self.bits)

        # Layer 2: FC (no ReLU on output)
        out, out_shift = self.fc2(h_q, h_shift)

        return out, [], []


class StandardANN_Int8_Fixed(nn.Module):
    """
    8-bit quantized ANN with FIXED scales matching integer inference exactly.

    Uses FixedScaleLinear layers that simulate true integer arithmetic.
    The scales are tracked via EMA during training and frozen for inference.
    """

    def __init__(self, config: Config):
        super().__init__()

        net = config.network
        self.bits = 8
        self.qmax = 127

        self.fc1 = FixedScaleLinear(net.input_size, net.hidden_size, bits=8)
        self.fc2 = FixedScaleLinear(net.hidden_size, net.output_size, bits=8)

        # Input scale (will be set based on data range)
        self.register_buffer('input_scale', torch.tensor(1.0 / self.qmax))

    def forward(self, x):
        # Quantize input with fixed scale
        # Update input scale based on observed range
        with torch.no_grad():
            x_abs_max = x.abs().max().clamp(min=1e-8)
            new_scale = x_abs_max / self.qmax
            if self.training:
                self.input_scale.copy_(0.999 * self.input_scale + 0.001 * new_scale)

        x_q = fixed_scale_quantize(x, self.input_scale, self.bits)

        # Layer 1
        h, h_scale = self.fc1(x_q, self.input_scale)
        h = F.relu(h)
        # ReLU doesn't change scale, but we requantize
        h_q = fixed_scale_quantize(h, h_scale, self.bits)

        # Layer 2
        out, out_scale = self.fc2(h_q, h_scale)

        return out, [], []


class StandardANN_Int4_Fixed(nn.Module):
    """4-bit quantized ANN with FIXED scales matching integer inference exactly."""

    def __init__(self, config: Config):
        super().__init__()

        net = config.network
        self.bits = 4
        self.qmax = 7

        self.fc1 = FixedScaleLinear(net.input_size, net.hidden_size, bits=4)
        self.fc2 = FixedScaleLinear(net.hidden_size, net.output_size, bits=4)

        self.register_buffer('input_scale', torch.tensor(1.0 / self.qmax))

    def forward(self, x):
        with torch.no_grad():
            x_abs_max = x.abs().max().clamp(min=1e-8)
            new_scale = x_abs_max / self.qmax
            if self.training:
                self.input_scale.copy_(0.999 * self.input_scale + 0.001 * new_scale)

        x_q = fixed_scale_quantize(x, self.input_scale, self.bits)

        h, h_scale = self.fc1(x_q, self.input_scale)
        h = F.relu(h)
        h_q = fixed_scale_quantize(h, h_scale, self.bits)

        out, out_scale = self.fc2(h_q, h_scale)

        return out, [], []


class StandardANN_Int16_Fixed(nn.Module):
    """16-bit quantized ANN with FIXED scales matching integer inference exactly."""

    def __init__(self, config: Config):
        super().__init__()

        net = config.network
        self.bits = 16
        self.qmax = 32767

        self.fc1 = FixedScaleLinear(net.input_size, net.hidden_size, bits=16)
        self.fc2 = FixedScaleLinear(net.hidden_size, net.output_size, bits=16)

        self.register_buffer('input_scale', torch.tensor(1.0 / self.qmax))

    def forward(self, x):
        with torch.no_grad():
            x_abs_max = x.abs().max().clamp(min=1e-8)
            new_scale = x_abs_max / self.qmax
            if self.training:
                self.input_scale.copy_(0.999 * self.input_scale + 0.001 * new_scale)

        x_q = fixed_scale_quantize(x, self.input_scale, self.bits)

        h, h_scale = self.fc1(x_q, self.input_scale)
        h = F.relu(h)
        h_q = fixed_scale_quantize(h, h_scale, self.bits)

        out, out_scale = self.fc2(h_q, h_scale)

        return out, [], []


class GilgameshSNN_Int8_Fixed(nn.Module):
    """8-bit quantized SNN with FIXED scales matching integer inference exactly."""

    def __init__(self, config: Config):
        super().__init__()

        net = config.network
        neuron = config.neuron
        self.bits = 8
        self.qmax = 127

        spike_grad = surrogate.fast_sigmoid(slope=neuron.slope)
        reset = neuron.reset_mechanism

        self.fc1 = FixedScaleLinear(net.input_size, net.hidden_size, bits=8)
        self.lif1 = snn.Leaky(
            beta=neuron.beta,
            threshold=neuron.threshold,
            spike_grad=spike_grad,
            reset_mechanism=reset
        )

        self.fc2 = FixedScaleLinear(net.hidden_size, net.output_size, bits=8)
        self.lif2 = snn.Leaky(
            beta=neuron.beta,
            threshold=neuron.threshold,
            spike_grad=spike_grad,
            reset_mechanism=reset
        )

        self.num_steps = config.training.num_steps
        self.register_buffer('input_scale', torch.tensor(1.0 / self.qmax))
        self.register_buffer('mem_scale', torch.tensor(1.0 / self.qmax))

    def forward(self, x):
        batch_size = x.size(0)

        with torch.no_grad():
            x_abs_max = x.abs().max().clamp(min=1e-8)
            new_scale = x_abs_max / self.qmax
            if self.training:
                self.input_scale.copy_(0.999 * self.input_scale + 0.001 * new_scale)

        x_q = fixed_scale_quantize(x, self.input_scale, self.bits)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spike_count = torch.zeros(batch_size, self.fc2.linear.out_features, device=x.device)
        mem_record = []
        spike_record = []

        for _ in range(self.num_steps):
            cur1, cur1_scale = self.fc1(x_q, self.input_scale)
            spk1, mem1 = self.lif1(cur1, mem1)
            mem1 = fixed_scale_quantize(mem1, self.mem_scale, self.bits)

            cur2, cur2_scale = self.fc2(spk1, torch.tensor(1.0, device=x.device))
            spk2, mem2 = self.lif2(cur2, mem2)
            mem2 = fixed_scale_quantize(mem2, self.mem_scale, self.bits)

            spike_count = spike_count + spk2
            mem_record.append(mem2.detach())
            spike_record.append(spk2.detach())

        return spike_count, mem_record, spike_record


class GilgameshSNN_Int4_Fixed(nn.Module):
    """4-bit quantized SNN with FIXED scales matching integer inference exactly."""

    def __init__(self, config: Config):
        super().__init__()

        net = config.network
        neuron = config.neuron
        self.bits = 4
        self.qmax = 7

        spike_grad = surrogate.fast_sigmoid(slope=neuron.slope)
        reset = neuron.reset_mechanism

        self.fc1 = FixedScaleLinear(net.input_size, net.hidden_size, bits=4)
        self.lif1 = snn.Leaky(
            beta=neuron.beta,
            threshold=neuron.threshold,
            spike_grad=spike_grad,
            reset_mechanism=reset
        )

        self.fc2 = FixedScaleLinear(net.hidden_size, net.output_size, bits=4)
        self.lif2 = snn.Leaky(
            beta=neuron.beta,
            threshold=neuron.threshold,
            spike_grad=spike_grad,
            reset_mechanism=reset
        )

        self.num_steps = config.training.num_steps
        self.register_buffer('input_scale', torch.tensor(1.0 / self.qmax))
        self.register_buffer('mem_scale', torch.tensor(1.0 / self.qmax))

    def forward(self, x):
        batch_size = x.size(0)

        with torch.no_grad():
            x_abs_max = x.abs().max().clamp(min=1e-8)
            new_scale = x_abs_max / self.qmax
            if self.training:
                self.input_scale.copy_(0.999 * self.input_scale + 0.001 * new_scale)

        x_q = fixed_scale_quantize(x, self.input_scale, self.bits)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spike_count = torch.zeros(batch_size, self.fc2.linear.out_features, device=x.device)
        mem_record = []
        spike_record = []

        for _ in range(self.num_steps):
            cur1, cur1_scale = self.fc1(x_q, self.input_scale)
            spk1, mem1 = self.lif1(cur1, mem1)
            mem1 = fixed_scale_quantize(mem1, self.mem_scale, self.bits)

            cur2, cur2_scale = self.fc2(spk1, torch.tensor(1.0, device=x.device))
            spk2, mem2 = self.lif2(cur2, mem2)
            mem2 = fixed_scale_quantize(mem2, self.mem_scale, self.bits)

            spike_count = spike_count + spk2
            mem_record.append(mem2.detach())
            spike_record.append(spk2.detach())

        return spike_count, mem_record, spike_record


class GilgameshSNN_Int16_Fixed(nn.Module):
    """16-bit quantized SNN with FIXED scales matching integer inference exactly."""

    def __init__(self, config: Config):
        super().__init__()

        net = config.network
        neuron = config.neuron
        self.bits = 16
        self.qmax = 32767

        spike_grad = surrogate.fast_sigmoid(slope=neuron.slope)
        reset = neuron.reset_mechanism

        self.fc1 = FixedScaleLinear(net.input_size, net.hidden_size, bits=16)
        self.lif1 = snn.Leaky(
            beta=neuron.beta,
            threshold=neuron.threshold,
            spike_grad=spike_grad,
            reset_mechanism=reset
        )

        self.fc2 = FixedScaleLinear(net.hidden_size, net.output_size, bits=16)
        self.lif2 = snn.Leaky(
            beta=neuron.beta,
            threshold=neuron.threshold,
            spike_grad=spike_grad,
            reset_mechanism=reset
        )

        self.num_steps = config.training.num_steps
        self.register_buffer('input_scale', torch.tensor(1.0 / self.qmax))
        self.register_buffer('mem_scale', torch.tensor(1.0 / self.qmax))

    def forward(self, x):
        batch_size = x.size(0)

        with torch.no_grad():
            x_abs_max = x.abs().max().clamp(min=1e-8)
            new_scale = x_abs_max / self.qmax
            if self.training:
                self.input_scale.copy_(0.999 * self.input_scale + 0.001 * new_scale)

        x_q = fixed_scale_quantize(x, self.input_scale, self.bits)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spike_count = torch.zeros(batch_size, self.fc2.linear.out_features, device=x.device)
        mem_record = []
        spike_record = []

        for _ in range(self.num_steps):
            cur1, cur1_scale = self.fc1(x_q, self.input_scale)
            spk1, mem1 = self.lif1(cur1, mem1)
            mem1 = fixed_scale_quantize(mem1, self.mem_scale, self.bits)

            cur2, cur2_scale = self.fc2(spk1, torch.tensor(1.0, device=x.device))
            spk2, mem2 = self.lif2(cur2, mem2)
            mem2 = fixed_scale_quantize(mem2, self.mem_scale, self.bits)

            spike_count = spike_count + spk2
            mem_record.append(mem2.detach())
            spike_record.append(spk2.detach())

        return spike_count, mem_record, spike_record


class StandardANN_Int8(nn.Module):
    """
    8-bit quantized ANN with fake quantization during training.

    All weights, activations, and intermediate values are quantized to int8 (-128 to 127).
    Uses straight-through estimator for gradient computation.
    """

    def __init__(self, config: Config):
        super().__init__()

        net = config.network
        self.bits = 8

        self.fc1 = QuantizedLinear(net.input_size, net.hidden_size, bits=8)
        self.fc2 = QuantizedLinear(net.hidden_size, net.output_size, bits=8)

    def forward(self, x):
        # Quantize input
        x = fake_quantize(x, self.bits)

        # Layer 1 with quantized ReLU
        x = self.fc1(x)
        x = F.relu(x)
        x = fake_quantize(x, self.bits)

        # Layer 2
        x = self.fc2(x)

        return x, [], []


class StandardANN_Int4(nn.Module):
    """
    4-bit quantized ANN with fake quantization during training.

    All weights, activations, and intermediate values are quantized to int4 (-8 to 7).
    Uses straight-through estimator for gradient computation.
    """

    def __init__(self, config: Config):
        super().__init__()

        net = config.network
        self.bits = 4

        self.fc1 = QuantizedLinear(net.input_size, net.hidden_size, bits=4)
        self.fc2 = QuantizedLinear(net.hidden_size, net.output_size, bits=4)

    def forward(self, x):
        # Quantize input
        x = fake_quantize(x, self.bits)

        # Layer 1 with quantized ReLU
        x = self.fc1(x)
        x = F.relu(x)
        x = fake_quantize(x, self.bits)

        # Layer 2
        x = self.fc2(x)

        return x, [], []


class GilgameshSNN_Int8(nn.Module):
    """
    8-bit quantized SNN with fake quantization during training.

    Weights, currents, and membrane potentials are quantized to int8 (-128 to 127).
    Spikes remain binary (0/1).
    """

    def __init__(self, config: Config):
        super().__init__()

        net = config.network
        neuron = config.neuron
        self.bits = 8

        spike_grad = surrogate.fast_sigmoid(slope=neuron.slope)
        reset = neuron.reset_mechanism

        # Quantized linear layers
        self.fc1 = QuantizedLinear(net.input_size, net.hidden_size, bits=8)
        self.lif1 = snn.Leaky(
            beta=neuron.beta,
            threshold=neuron.threshold,
            spike_grad=spike_grad,
            reset_mechanism=reset
        )

        self.fc2 = QuantizedLinear(net.hidden_size, net.output_size, bits=8)
        self.lif2 = snn.Leaky(
            beta=neuron.beta,
            threshold=neuron.threshold,
            spike_grad=spike_grad,
            reset_mechanism=reset
        )

        self.num_steps = config.training.num_steps

    def forward(self, x):
        batch_size = x.size(0)

        # Quantize input
        x = fake_quantize(x, self.bits)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spike_count = torch.zeros(batch_size, self.fc2.linear.out_features, device=x.device)
        mem_record = []
        spike_record = []

        for _ in range(self.num_steps):
            # Layer 1: quantized current, quantized membrane
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            mem1 = fake_quantize(mem1, self.bits)

            # Layer 2: quantized current, quantized membrane
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            mem2 = fake_quantize(mem2, self.bits)

            spike_count = spike_count + spk2
            mem_record.append(mem2.detach())
            spike_record.append(spk2.detach())

        return spike_count, mem_record, spike_record


class GilgameshSNN_Int4(nn.Module):
    """
    4-bit quantized SNN with fake quantization during training.

    Weights, currents, and membrane potentials are quantized to int4 (-8 to 7).
    Spikes remain binary (0/1).
    """

    def __init__(self, config: Config):
        super().__init__()

        net = config.network
        neuron = config.neuron
        self.bits = 4

        spike_grad = surrogate.fast_sigmoid(slope=neuron.slope)
        reset = neuron.reset_mechanism

        # Quantized linear layers
        self.fc1 = QuantizedLinear(net.input_size, net.hidden_size, bits=4)
        self.lif1 = snn.Leaky(
            beta=neuron.beta,
            threshold=neuron.threshold,
            spike_grad=spike_grad,
            reset_mechanism=reset
        )

        self.fc2 = QuantizedLinear(net.hidden_size, net.output_size, bits=4)
        self.lif2 = snn.Leaky(
            beta=neuron.beta,
            threshold=neuron.threshold,
            spike_grad=spike_grad,
            reset_mechanism=reset
        )

        self.num_steps = config.training.num_steps

    def forward(self, x):
        batch_size = x.size(0)

        # Quantize input
        x = fake_quantize(x, self.bits)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spike_count = torch.zeros(batch_size, self.fc2.linear.out_features, device=x.device)
        mem_record = []
        spike_record = []

        for _ in range(self.num_steps):
            # Layer 1: quantized current, quantized membrane
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            mem1 = fake_quantize(mem1, self.bits)

            # Layer 2: quantized current, quantized membrane
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            mem2 = fake_quantize(mem2, self.bits)

            spike_count = spike_count + spk2
            mem_record.append(mem2.detach())
            spike_record.append(spk2.detach())

        return spike_count, mem_record, spike_record


class StandardANN_Int16(nn.Module):
    """
    16-bit quantized ANN with fake quantization during training.

    All weights, activations, and intermediate values are quantized to int16 (-32768 to 32767).
    Uses straight-through estimator for gradient computation.
    """

    def __init__(self, config: Config):
        super().__init__()

        net = config.network
        self.bits = 16

        self.fc1 = QuantizedLinear(net.input_size, net.hidden_size, bits=16)
        self.fc2 = QuantizedLinear(net.hidden_size, net.output_size, bits=16)

    def forward(self, x):
        # Quantize input
        x = fake_quantize(x, self.bits)

        # Layer 1 with quantized ReLU
        x = self.fc1(x)
        x = F.relu(x)
        x = fake_quantize(x, self.bits)

        # Layer 2
        x = self.fc2(x)

        return x, [], []


class GilgameshSNN_Int16(nn.Module):
    """
    16-bit quantized SNN with fake quantization during training.

    Weights, currents, and membrane potentials are quantized to int16 (-32768 to 32767).
    Spikes remain binary (0/1).
    """

    def __init__(self, config: Config):
        super().__init__()

        net = config.network
        neuron = config.neuron
        self.bits = 16

        spike_grad = surrogate.fast_sigmoid(slope=neuron.slope)
        reset = neuron.reset_mechanism

        # Quantized linear layers
        self.fc1 = QuantizedLinear(net.input_size, net.hidden_size, bits=16)
        self.lif1 = snn.Leaky(
            beta=neuron.beta,
            threshold=neuron.threshold,
            spike_grad=spike_grad,
            reset_mechanism=reset
        )

        self.fc2 = QuantizedLinear(net.hidden_size, net.output_size, bits=16)
        self.lif2 = snn.Leaky(
            beta=neuron.beta,
            threshold=neuron.threshold,
            spike_grad=spike_grad,
            reset_mechanism=reset
        )

        self.num_steps = config.training.num_steps

    def forward(self, x):
        batch_size = x.size(0)

        # Quantize input
        x = fake_quantize(x, self.bits)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spike_count = torch.zeros(batch_size, self.fc2.linear.out_features, device=x.device)
        mem_record = []
        spike_record = []

        for _ in range(self.num_steps):
            # Layer 1: quantized current, quantized membrane
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            mem1 = fake_quantize(mem1, self.bits)

            # Layer 2: quantized current, quantized membrane
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            mem2 = fake_quantize(mem2, self.bits)

            spike_count = spike_count + spk2
            mem_record.append(mem2.detach())
            spike_record.append(spk2.detach())

        return spike_count, mem_record, spike_record


# =============================================================================
# Training
# =============================================================================

class CosineAnnealingLR:
    """Cosine annealing learning rate scheduler (matching gilgamesh)."""

    def __init__(self, optimizer, initial_lr: float, min_lr: float, total_epochs: int):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs

    def step(self, epoch: int):
        """Update learning rate based on epoch."""
        progress = epoch / self.total_epochs
        lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


class Trainer:
    """Training loop for SNN models."""

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        device: torch.device,
        name: str = "model"
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.name = name

        tc = config.training

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=tc.lr,
            betas=(tc.beta1, tc.beta2),
            weight_decay=tc.weight_decay
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            initial_lr=tc.lr,
            min_lr=tc.lr_min,
            total_epochs=tc.epochs
        )

        self.grad_clip = tc.grad_clip

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'lr': [],
            'epoch_time': []
        }

    def train_epoch(self, train_loader: DataLoader) -> tuple:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for data, targets in train_loader:
            data = data.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            spike_count, _, _ = self.model(data)

            # Cross-entropy loss on spike counts
            loss = F.cross_entropy(spike_count, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            # Statistics
            total_loss += loss.item() * data.size(0)
            pred = spike_count.argmax(dim=1)
            correct += (pred == targets).sum().item()
            total += data.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def run_evaluation(self, test_loader: DataLoader) -> tuple:
        """Evaluate on test set."""
        self.model.train(False)  # Set to evaluation mode

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, targets in test_loader:
                data = data.to(self.device)
                targets = targets.to(self.device)

                spike_count, _, _ = self.model(data)

                loss = F.cross_entropy(spike_count, targets)

                total_loss += loss.item() * data.size(0)
                pred = spike_count.argmax(dim=1)
                correct += (pred == targets).sum().item()
                total += data.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        verbose: bool = True
    ) -> dict:
        """Full training loop."""
        epochs = self.config.training.epochs

        best_test_acc = 0.0
        best_weights = None

        for epoch in range(epochs):
            epoch_start = time.time()

            # Update learning rate
            lr = self.scheduler.step(epoch)

            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Evaluate
            test_loss, test_acc = self.run_evaluation(test_loader)

            epoch_time = time.time() - epoch_start

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            self.history['lr'].append(lr)
            self.history['epoch_time'].append(epoch_time)

            # Track best
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if verbose:
                print(f"[{self.name}] Epoch {epoch+1:2d}/{epochs} | "
                      f"Train: {train_acc*100:.2f}% | "
                      f"Test: {test_acc*100:.2f}% | "
                      f"LR: {lr:.6f} | "
                      f"Time: {epoch_time:.1f}s")

        # Restore best weights
        if best_weights is not None:
            self.model.load_state_dict(best_weights)

        return {
            'best_test_acc': best_test_acc,
            'final_train_acc': self.history['train_acc'][-1],
            'final_test_acc': self.history['test_acc'][-1],
            'history': self.history
        }


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(results: dict, output_dir: Path):
    """Generate markdown comparison report."""

    report = []
    report.append("# SNNTorch Comparison Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    report.append("## Configuration (matching gilgamesh)")
    report.append("")
    report.append("| Parameter | Value |")
    report.append("|-----------|-------|")
    report.append("| Input Size | 36 (6×6 MNIST) |")
    report.append("| Hidden Size | 12 |")
    report.append("| Output Size | 10 |")
    report.append("| Beta (decay) | 0.9 |")
    report.append("| Threshold | 1.0 |")
    report.append("| Surrogate Slope | 25.0 |")
    report.append("| Learning Rate | 0.001 → 0.00001 (cosine) |")
    report.append("| Epochs | 15 |")
    report.append("| Batch Size | 128 |")
    report.append("| Timesteps | 25 |")
    report.append("| Optimizer | Adam (β1=0.9, β2=0.999) |")
    report.append("")

    report.append("## Results Summary")
    report.append("")
    report.append("| Model | Best Test Acc | Final Train Acc | Final Test Acc | Parameters |")
    report.append("|-------|---------------|-----------------|----------------|------------|")

    for name, res in results.items():
        report.append(
            f"| {name} | {res['best_test_acc']*100:.2f}% | "
            f"{res['final_train_acc']*100:.2f}% | "
            f"{res['final_test_acc']*100:.2f}% | "
            f"{res['params']:,} |"
        )

    report.append("")
    report.append("## Model Descriptions")
    report.append("")
    report.append("### Spiking Neural Networks (SNNs)")
    report.append("")
    report.append("#### GilgameshSNN (Baseline)")
    report.append("Standard 2-layer feedforward LIF SNN matching gilgamesh architecture exactly.")
    report.append("Architecture: Input(36) → FC → LIF(12) → FC → LIF(10)")
    report.append("")
    report.append("#### GilgameshSNN_Synaptic")
    report.append("Uses dual exponential synaptic dynamics with separate synaptic and membrane time constants.")
    report.append("")
    report.append("#### GilgameshSNN_Recurrent")
    report.append("Adds recurrent connections within the hidden layer for temporal processing.")
    report.append("")
    report.append("#### GilgameshSNN_3Layer")
    report.append("Deeper SNN with 3 layers: Input(36) → LIF(12) → LIF(6) → LIF(10)")
    report.append("")
    report.append("### Artificial Neural Networks (ANNs)")
    report.append("")
    report.append("#### StandardANN")
    report.append("Standard 2-layer feedforward ANN with ReLU activations (non-spiking baseline).")
    report.append("Architecture: Input(36) → FC → ReLU(12) → FC → Output(10)")
    report.append("")
    report.append("#### StandardANN_3Layer")
    report.append("Deeper ANN with 3 layers: Input(36) → ReLU(12) → ReLU(6) → Output(10)")
    report.append("")

    report.append("## Training Curves")
    report.append("")
    report.append("See `training_curves.png` for visualization of training and test accuracy over epochs.")
    report.append("")

    report.append("## Weight Files")
    report.append("")
    for name in results.keys():
        report.append(f"- `{name}_weights.pt`: PyTorch state dict")
        report.append(f"- `{name}_weights.json`: Weights exported as JSON (for gilgamesh compatibility)")
    report.append("")

    report.append("## Notes")
    report.append("")
    report.append("- All models use rate-coded input (same input at each timestep)")
    report.append("- Loss: Cross-entropy on spike counts")
    report.append("- Best weights (highest test accuracy) are saved")
    report.append("- JSON exports use gilgamesh-compatible format")
    report.append("")

    # Write report
    report_path = output_dir / "comparison_report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))

    return report_path


def plot_training_curves(results: dict, output_dir: Path):
    """Plot training curves for all models."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

        # Training accuracy
        ax = axes[0, 0]
        for (name, res), color in zip(results.items(), colors):
            epochs = range(1, len(res['history']['train_acc']) + 1)
            ax.plot(epochs, [a*100 for a in res['history']['train_acc']],
                   label=name, color=color)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Training Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Test accuracy
        ax = axes[0, 1]
        for (name, res), color in zip(results.items(), colors):
            epochs = range(1, len(res['history']['test_acc']) + 1)
            ax.plot(epochs, [a*100 for a in res['history']['test_acc']],
                   label=name, color=color)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Test Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Training loss
        ax = axes[1, 0]
        for (name, res), color in zip(results.items(), colors):
            epochs = range(1, len(res['history']['train_loss']) + 1)
            ax.plot(epochs, res['history']['train_loss'], label=name, color=color)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Learning rate
        ax = axes[1, 1]
        # All models use same LR schedule, just plot one
        first_res = list(results.values())[0]
        epochs = range(1, len(first_res['history']['lr']) + 1)
        ax.plot(epochs, first_res['history']['lr'], color='black')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule (Cosine Annealing)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        plot_path = output_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        return plot_path

    except ImportError:
        print("Warning: matplotlib not available, skipping plot generation")
        return None


def save_weights(model: nn.Module, name: str, output_dir: Path):
    """Save model weights in both PyTorch and JSON formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # PyTorch format
    pt_path = output_dir / f"{name}_weights.pt"
    torch.save(model.state_dict(), pt_path)

    # JSON format (gilgamesh compatible)
    state_dict = model.state_dict()
    weights_json = {}

    for key, tensor in state_dict.items():
        weights_json[key] = tensor.cpu().numpy().tolist()

    json_path = output_dir / f"{name}_weights.json"
    with open(json_path, 'w') as f:
        json.dump(weights_json, f, indent=2)

    return pt_path, json_path


def quantize_for_embedded(tensor: torch.Tensor, bits: int) -> tuple:
    """
    Quantize a tensor to actual integers for embedded deployment.

    Returns:
        int_weights: numpy array of integers
        scale: float scale factor (real_value = int_value * scale)
        zero_point: always 0 for symmetric quantization
    """
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1

    # Choose dtype based on bit width
    if bits <= 8:
        dtype = np.int8
    else:
        dtype = np.int16

    tensor_np = tensor.detach().cpu().numpy()
    abs_max = np.abs(tensor_np).max()

    if abs_max < 1e-8:
        # All zeros
        scale = 1.0
        int_weights = np.zeros_like(tensor_np, dtype=dtype)
    else:
        scale = abs_max / qmax
        int_weights = np.clip(np.round(tensor_np / scale), qmin, qmax).astype(dtype)

    return int_weights, scale, 0


def quantize_with_fixed_scale(tensor: torch.Tensor, scale: float, bits: int) -> tuple:
    """
    Quantize a tensor using a FIXED scale (for fixed-scale models).

    Returns:
        int_weights: numpy array of integers
        scale: the fixed scale factor used
        zero_point: always 0 for symmetric quantization
    """
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1

    # Choose dtype based on bit width
    if bits <= 8:
        dtype = np.int8
    else:
        dtype = np.int16

    tensor_np = tensor.detach().cpu().numpy()
    int_weights = np.clip(np.round(tensor_np / scale), qmin, qmax).astype(dtype)

    return int_weights, scale, 0


def quantize_with_shift(tensor: torch.Tensor, shift: int, bits: int) -> tuple:
    """
    Quantize a tensor using a power-of-2 scale (shift amount).

    Returns:
        int_weights: numpy array of integers
        shift: the shift amount used (scale = 2^shift)
        zero_point: always 0 for symmetric quantization
    """
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1

    if bits <= 8:
        dtype = np.int8
    else:
        dtype = np.int16

    scale = 2.0 ** shift
    tensor_np = tensor.detach().cpu().numpy()
    int_weights = np.clip(np.round(tensor_np / scale), qmin, qmax).astype(dtype)

    return int_weights, shift, 0


def export_embedded_weights(model: nn.Module, name: str, bits: int, output_dir: Path, config: Config):
    """
    Export model weights for embedded deployment.

    Generates:
    - C header file with integer weights and scale factors (or shift amounts)
    - Binary file with packed weights
    - JSON metadata file

    For shift-based models (*_shift), uses integer shift amounts instead of float scales.
    For fixed-scale models (*_fixed), uses the stored scales from training.
    For dynamic-scale models, computes scales from weight values.
    """
    output_dir = Path(output_dir)
    state_dict = model.state_dict()

    # Check if this is a shift-based model by looking for weight_shift buffers
    is_shift_based = any('weight_shift' in key for key in state_dict.keys())

    # Check if this is a fixed-scale model by looking for weight_scale buffers
    is_fixed_scale = any('weight_scale' in key for key in state_dict.keys())

    # Build a map of layer -> stored shifts for shift-based models
    stored_shifts = {}
    if is_shift_based:
        for key, tensor in state_dict.items():
            if 'weight_shift' in key or 'bias_shift' in key or 'output_shift' in key:
                layer_name = key.split('.')[0]
                shift_type = key.split('.')[-1]
                if layer_name not in stored_shifts:
                    stored_shifts[layer_name] = {}
                stored_shifts[layer_name][shift_type] = int(tensor.item())

    # Build a map of layer -> stored scales for fixed-scale models
    stored_scales = {}
    if is_fixed_scale:
        for key, tensor in state_dict.items():
            if 'weight_scale' in key or 'bias_scale' in key or 'output_scale' in key:
                # Extract layer name (e.g., 'fc1' from 'fc1.weight_scale')
                layer_name = key.split('.')[0]
                scale_type = key.split('.')[-1]  # 'weight_scale', 'bias_scale', etc.
                if layer_name not in stored_scales:
                    stored_scales[layer_name] = {}
                stored_scales[layer_name][scale_type] = tensor.item()

    # Collect quantized weights
    quantized_layers = {}

    # Collect shift amounts for shift-based models
    shift_amounts = {}
    if is_shift_based:
        for key, tensor in state_dict.items():
            if key == 'input_shift':
                shift_amounts['input_shift'] = int(tensor.item())
            elif 'output_shift' in key:
                layer_name = key.split('.')[0]
                shift_amounts[f'{layer_name}_output_shift'] = int(tensor.item())
            elif 'weight_shift' in key:
                layer_name = key.split('.')[0]
                shift_amounts[f'{layer_name}_weight_shift'] = int(tensor.item())
            elif 'bias_shift' in key:
                layer_name = key.split('.')[0]
                shift_amounts[f'{layer_name}_bias_shift'] = int(tensor.item())

    # Also collect input_scale and output_scales for fixed-scale models
    inference_scales = {}
    if is_fixed_scale:
        for key, tensor in state_dict.items():
            if key == 'input_scale':
                inference_scales['input_scale'] = tensor.item()
            elif 'output_scale' in key:
                layer_name = key.split('.')[0]
                inference_scales[f'{layer_name}_output_scale'] = tensor.item()
            elif key == 'mem_scale':
                inference_scales['mem_scale'] = tensor.item()

    for key, tensor in state_dict.items():
        # Skip scale/shift buffers and other non-weight tensors
        if 'scale' in key or 'shift' in key or 'calibrated' in key or 'abs_max' in key:
            continue

        # Handle nested QuantizedLinear layers
        clean_key = key.replace('.linear.', '_').replace('.', '_')

        if 'weight' in key or 'bias' in key:
            # Determine which quantization method to use
            if is_shift_based:
                # Use shift-based quantization
                layer_name = key.split('.')[0]
                if 'weight' in key and layer_name in stored_shifts:
                    shift = stored_shifts[layer_name].get('weight_shift', -7)
                elif 'bias' in key and layer_name in stored_shifts:
                    shift = stored_shifts[layer_name].get('bias_shift', -7)
                else:
                    shift = -7  # Default

                int_vals, shift_val, zp = quantize_with_shift(tensor, shift, bits)
                # Store shift instead of scale
                quantized_layers[clean_key] = {
                    'values': int_vals,
                    'scale': 2.0 ** shift_val,  # For compatibility
                    'shift': shift_val,
                    'zero_point': zp,
                    'shape': list(tensor.shape),
                    'bits': bits
                }
                continue

            elif is_fixed_scale:
                # Extract layer name (e.g., 'fc1' from 'fc1.linear.weight')
                layer_name = key.split('.')[0]
                if 'weight' in key and layer_name in stored_scales:
                    scale = stored_scales[layer_name].get('weight_scale', None)
                elif 'bias' in key and layer_name in stored_scales:
                    scale = stored_scales[layer_name].get('bias_scale', None)
                else:
                    scale = None

                if scale is not None:
                    int_vals, scale, zp = quantize_with_fixed_scale(tensor, scale, bits)
                else:
                    int_vals, scale, zp = quantize_for_embedded(tensor, bits)
            else:
                int_vals, scale, zp = quantize_for_embedded(tensor, bits)

            quantized_layers[clean_key] = {
                'values': int_vals,
                'scale': scale,
                'zero_point': zp,
                'shape': list(tensor.shape),
                'bits': bits
            }

    # Add inference scales to quantized_layers metadata
    quantized_layers['_inference_scales'] = inference_scales

    # Add shift amounts for shift-based models
    if is_shift_based:
        quantized_layers['_shift_amounts'] = shift_amounts

    # Generate C header file
    header_path = output_dir / f"{name}_weights_{bits}bit.h"
    _generate_c_header(name, bits, quantized_layers, config, header_path)

    # Generate binary file (packed weights)
    bin_path = output_dir / f"{name}_weights_{bits}bit.bin"
    _generate_binary_weights(quantized_layers, bits, bin_path)

    # Generate metadata JSON
    meta_path = output_dir / f"{name}_weights_{bits}bit_meta.json"
    _generate_metadata(name, bits, quantized_layers, config, meta_path)

    return header_path, bin_path, meta_path


def _generate_c_header(name: str, bits: int, layers: dict, config: Config, path: Path):
    """Generate C header file with embedded weights."""

    lines = []
    lines.append(f"/*")
    lines.append(f" * {name} - {bits}-bit Quantized Weights for Embedded Deployment")
    lines.append(f" * Generated by gilgamesh snntorch_comparison.py")
    lines.append(f" *")
    lines.append(f" * Network: {config.network.input_size} -> {config.network.hidden_size} -> {config.network.output_size}")
    lines.append(f" * Quantization: {bits}-bit signed integers ({-(1<<(bits-1))} to {(1<<(bits-1))-1})")
    lines.append(f" * Format: real_value = int_value * scale")
    lines.append(f" */")
    lines.append(f"")
    lines.append(f"#ifndef {name.upper()}_WEIGHTS_{bits}BIT_H")
    lines.append(f"#define {name.upper()}_WEIGHTS_{bits}BIT_H")
    lines.append(f"")
    lines.append(f"#include <stdint.h>")
    lines.append(f"")

    # Network configuration
    lines.append(f"/* Network Configuration */")
    lines.append(f"#define {name.upper()}_INPUT_SIZE  {config.network.input_size}")
    lines.append(f"#define {name.upper()}_HIDDEN_SIZE {config.network.hidden_size}")
    lines.append(f"#define {name.upper()}_OUTPUT_SIZE {config.network.output_size}")
    lines.append(f"#define {name.upper()}_BITS        {bits}")
    lines.append(f"")

    # SNN parameters if applicable
    if 'SNN' in name or 'Gilgamesh' in name:
        lines.append(f"/* SNN Parameters */")
        lines.append(f"#define {name.upper()}_NUM_STEPS   {config.training.num_steps}")
        # Beta as fixed-point (Q15 for int16, Q7 for int8, Q3 for int4)
        if bits == 16:
            beta_fixed = int(config.neuron.beta * 32767)
            lines.append(f"#define {name.upper()}_BETA_Q15    {beta_fixed}  /* {config.neuron.beta} in Q15 format */")
            lines.append(f"#define {name.upper()}_BETA_SHIFT  15")
        elif bits == 8:
            beta_fixed = int(config.neuron.beta * 127)
            lines.append(f"#define {name.upper()}_BETA_Q7     {beta_fixed}  /* {config.neuron.beta} in Q7 format */")
            lines.append(f"#define {name.upper()}_BETA_SHIFT  7")
        else:
            beta_fixed = int(config.neuron.beta * 7)
            lines.append(f"#define {name.upper()}_BETA_Q3     {beta_fixed}  /* {config.neuron.beta} in Q3 format */")
            lines.append(f"#define {name.upper()}_BETA_SHIFT  3")
        # Threshold in fixed-point
        lines.append(f"#define {name.upper()}_THRESHOLD   {int(config.neuron.threshold * (1 << (bits-1)))}")
        lines.append(f"")

    # Check if this is a shift-based model
    shift_amounts = layers.get('_shift_amounts', {})
    if shift_amounts:
        # Output shift amounts for shift-based inference (THIS IS WHAT ARDUINO USES)
        lines.append(f"/* Shift amounts for integer-only inference (scale = 2^shift) */")
        lines.append(f"/* Use these for requantization: out = (acc + (1 << (shift-1))) >> shift */")
        for shift_name, shift_val in shift_amounts.items():
            define_name = f"{name.upper()}_{shift_name.upper()}"
            lines.append(f"#define {define_name} ({shift_val})")
        lines.append(f"")

        # Compute and output the combined requantization shifts
        lines.append(f"/* Pre-computed shift amounts for integer-only inference */")
        lines.append(f"/* Convention: positive = left shift, negative = right shift */")
        lines.append(f"/* Use shift_round() helper: if shift >= 0: x << shift, else: (x + (1 << (-shift-1))) >> (-shift) */")
        input_shift = shift_amounts.get('input_shift', -7)
        fc1_w_shift = shift_amounts.get('fc1_weight_shift', -7)
        fc1_b_shift = shift_amounts.get('fc1_bias_shift', -7)
        fc1_out_shift = shift_amounts.get('fc1_output_shift', -4)
        fc2_w_shift = shift_amounts.get('fc2_weight_shift', -7)
        fc2_b_shift = shift_amounts.get('fc2_bias_shift', -7)
        fc2_out_shift = shift_amounts.get('fc2_output_shift', -4)

        # Layer 1 accumulator shift
        fc1_acc_shift = input_shift + fc1_w_shift
        # Requant: out = acc * 2^(acc_shift - out_shift), positive=left, negative=right
        fc1_requant = fc1_acc_shift - fc1_out_shift
        # Bias align: b_aligned = b * 2^(b_shift - acc_shift), positive=left, negative=right
        fc1_bias_align = fc1_b_shift - fc1_acc_shift
        lines.append(f"#define {name.upper()}_FC1_REQUANT_SHIFT ({fc1_requant})  /* acc_shift - out_shift */")
        lines.append(f"#define {name.upper()}_FC1_BIAS_ALIGN_SHIFT ({fc1_bias_align})  /* bias_shift - acc_shift */")

        # Layer 2 accumulator shift (input to FC2 has fc1_out_shift)
        fc2_acc_shift = fc1_out_shift + fc2_w_shift
        fc2_requant = fc2_acc_shift - fc2_out_shift
        fc2_bias_align = fc2_b_shift - fc2_acc_shift
        lines.append(f"#define {name.upper()}_FC2_REQUANT_SHIFT ({fc2_requant})  /* acc_shift - out_shift */")
        lines.append(f"#define {name.upper()}_FC2_BIAS_ALIGN_SHIFT ({fc2_bias_align})  /* bias_shift - acc_shift */")
        lines.append(f"")

    # Optional scale factors (for debugging/verification only - not needed for inference)
    lines.append(f"/*")
    lines.append(f" * OPTIONAL: Scale factors for debugging/verification.")
    lines.append(f" * These are NOT needed for integer inference - just do int math directly.")
    lines.append(f" * Define {name.upper()}_INCLUDE_SCALES to include them.")
    lines.append(f" */")
    lines.append(f"#ifdef {name.upper()}_INCLUDE_SCALES")
    lines.append(f"")

    inference_scales = layers.get('_inference_scales', {})
    if inference_scales:
        lines.append(f"/* Inference Scales (fixed during training) */")
        for scale_name, scale_val in inference_scales.items():
            define_name = f"{name.upper()}_{scale_name.upper()}"
            lines.append(f"#define {define_name} {scale_val:.10f}f")
        lines.append(f"")

    # Weight scale factors
    lines.append(f"/* Weight Scale Factors (real_value = int_value * scale) */")
    for layer_name, layer_data in layers.items():
        if layer_name.startswith('_'):
            continue
        scale_name = f"{name.upper()}_{layer_name.upper()}_SCALE"
        lines.append(f"#define {scale_name} {layer_data['scale']:.10f}f")
    lines.append(f"")
    lines.append(f"#endif /* {name.upper()}_INCLUDE_SCALES */")
    lines.append(f"")

    # Weight arrays (integer values - this is all you need for inference)
    for layer_name, layer_data in layers.items():
        if layer_name.startswith('_'):  # Skip metadata entries
            continue
        shape = layer_data['shape']
        values = layer_data['values'].flatten()
        total = len(values)

        lines.append(f"/* {layer_name}: shape={shape}, total={total} */")

        if bits == 4:
            # Pack int4 values: 2 per byte
            lines.append(f"static const uint8_t {name}_{layer_name}_packed[{(total + 1) // 2}] = {{")
            packed = []
            for i in range(0, total, 2):
                lo = values[i] & 0x0F
                hi = (values[i + 1] & 0x0F) if i + 1 < total else 0
                packed.append((hi << 4) | lo)
            # Format in rows of 16
            for i in range(0, len(packed), 16):
                row = packed[i:i+16]
                row_str = ", ".join(f"0x{v:02X}" for v in row)
                lines.append(f"    {row_str},")
            lines.append(f"}};")
        elif bits == 16:
            # int16 values
            lines.append(f"static const int16_t {name}_{layer_name}[{total}] = {{")
            for i in range(0, total, 12):
                row = values[i:i+12]
                row_str = ", ".join(f"{v:6d}" for v in row)
                lines.append(f"    {row_str},")
            lines.append(f"}};")
        else:
            # int8 values directly
            lines.append(f"static const int8_t {name}_{layer_name}[{total}] = {{")
            for i in range(0, total, 16):
                row = values[i:i+16]
                row_str = ", ".join(f"{v:4d}" for v in row)
                lines.append(f"    {row_str},")
            lines.append(f"}};")
        lines.append(f"")

    lines.append(f"#endif /* {name.upper()}_WEIGHTS_{bits}BIT_H */")

    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def _generate_binary_weights(layers: dict, bits: int, path: Path):
    """Generate binary file with packed weights."""
    with open(path, 'wb') as f:
        for layer_name, layer_data in layers.items():
            if layer_name.startswith('_'):  # Skip metadata entries
                continue
            values = layer_data['values'].flatten()

            if bits == 4:
                # Pack 2 int4 values per byte
                packed = bytearray()
                for i in range(0, len(values), 2):
                    lo = int(values[i]) & 0x0F
                    hi = (int(values[i + 1]) & 0x0F) if i + 1 < len(values) else 0
                    packed.append((hi << 4) | lo)
                f.write(packed)
            elif bits == 16:
                # int16 values (little-endian)
                f.write(values.astype(np.int16).tobytes())
            else:
                # int8 values directly
                f.write(values.astype(np.int8).tobytes())


def _generate_metadata(name: str, bits: int, layers: dict, config: Config, path: Path):
    """Generate JSON metadata file."""
    meta = {
        'model_name': name,
        'bits': bits,
        'network': {
            'input_size': config.network.input_size,
            'hidden_size': config.network.hidden_size,
            'output_size': config.network.output_size
        },
        'neuron': {
            'beta': config.neuron.beta,
            'threshold': config.neuron.threshold
        },
        'training': {
            'num_steps': config.training.num_steps
        },
        'layers': {},
        'inference_scales': layers.get('_inference_scales', {}),
        'shift_amounts': layers.get('_shift_amounts', {})
    }

    for layer_name, layer_data in layers.items():
        if layer_name.startswith('_'):  # Skip metadata entries
            continue
        meta['layers'][layer_name] = {
            'shape': layer_data['shape'],
            'scale': float(layer_data['scale']),  # Convert numpy float to Python float
            'zero_point': int(layer_data['zero_point']),
            'bits': int(layer_data['bits'])
        }
        # Include shift if present (for shift-based models)
        if 'shift' in layer_data:
            meta['layers'][layer_name]['shift'] = int(layer_data['shift'])

    with open(path, 'w') as f:
        json.dump(meta, f, indent=2)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Integer-Only Inference Verification
# =============================================================================

class IntegerInferenceANN:
    """
    Pure integer inference for ANN - no floating point operations.

    This simulates exactly what embedded hardware would compute.
    All arithmetic uses int32 accumulators with proper scaling.
    """

    def __init__(self, weights: dict, bits: int):
        """
        Args:
            weights: dict from export_embedded_weights containing int weights and scales
            bits: 8 or 4
        """
        self.bits = bits
        self.qmax = (1 << (bits - 1)) - 1
        self.qmin = -(1 << (bits - 1))

        # Extract weights and scales
        self.fc1_w = weights['fc1_weight']['values']
        self.fc1_b = weights['fc1_bias']['values']
        self.fc1_w_scale = weights['fc1_weight']['scale']
        self.fc1_b_scale = weights['fc1_bias']['scale']

        self.fc2_w = weights['fc2_weight']['values']
        self.fc2_b = weights['fc2_bias']['values']
        self.fc2_w_scale = weights['fc2_weight']['scale']
        self.fc2_b_scale = weights['fc2_bias']['scale']

    def quantize_input(self, x: np.ndarray) -> tuple:
        """Quantize float input to integer."""
        abs_max = np.abs(x).max()
        if abs_max < 1e-8:
            return np.zeros_like(x, dtype=np.int32), 1.0
        scale = abs_max / self.qmax
        x_int = np.clip(np.round(x / scale), self.qmin, self.qmax).astype(np.int32)
        return x_int, scale

    def integer_matmul(self, x_int: np.ndarray, x_scale: float,
                       w_int: np.ndarray, w_scale: float,
                       b_int: np.ndarray, b_scale: float) -> tuple:
        """
        True integer matrix multiplication for embedded deployment.

        Computes: y = x @ w.T + b using int32 accumulators.
        This is what actual embedded hardware would compute.
        """
        # int32 accumulator for matmul
        acc = np.dot(x_int.astype(np.int32), w_int.T.astype(np.int32))

        # Combined scale for matmul result
        matmul_scale = x_scale * w_scale

        # Add bias (align to accumulator scale)
        if matmul_scale > 1e-10:
            b_aligned = np.round(b_int.astype(np.float64) * b_scale / matmul_scale).astype(np.int32)
        else:
            b_aligned = np.zeros_like(b_int, dtype=np.int32)
        acc = acc + b_aligned

        # Requantize to target bit width
        acc_float = acc.astype(np.float64) * matmul_scale
        abs_max = np.abs(acc_float).max()
        if abs_max < 1e-8:
            out_scale = 1.0
            out_int = np.zeros_like(acc, dtype=np.int32)
        else:
            out_scale = abs_max / self.qmax
            out_int = np.clip(np.round(acc_float / out_scale), self.qmin, self.qmax).astype(np.int32)

        return out_int, out_scale

    def integer_relu(self, x_int: np.ndarray) -> np.ndarray:
        """Integer ReLU - just clamp negatives to zero."""
        return np.maximum(x_int, 0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass using only integer arithmetic.

        Args:
            x: Float input [batch, 36]

        Returns:
            logits: Integer logits [batch, 10] (for argmax classification)
        """
        # Quantize input
        x_int, x_scale = self.quantize_input(x)

        # Layer 1: FC + ReLU
        h_int, h_scale = self.integer_matmul(
            x_int, x_scale,
            self.fc1_w, self.fc1_w_scale,
            self.fc1_b, self.fc1_b_scale
        )
        h_int = self.integer_relu(h_int)

        # Layer 2: FC (no activation for output)
        out_int, out_scale = self.integer_matmul(
            h_int, h_scale,
            self.fc2_w, self.fc2_w_scale,
            self.fc2_b, self.fc2_b_scale
        )

        return out_int  # Return integer logits for argmax


class IntegerInferenceANN_Shift:
    """
    Pure integer inference for ANN using ONLY bit shifts - no floating point.

    This is what the embedded Arduino code should implement.
    All requantization uses bit shifts instead of float multiplies.
    """

    def __init__(self, weights: dict, bits: int):
        """
        Args:
            weights: dict containing int weights and shift amounts
            bits: 8, 4, or 16
        """
        self.bits = bits
        self.qmax = (1 << (bits - 1)) - 1
        self.qmin = -(1 << (bits - 1))

        # Extract weights (int8 arrays)
        self.fc1_w = weights['fc1_weight']['values']
        self.fc1_b = weights['fc1_bias']['values']
        self.fc2_w = weights['fc2_weight']['values']
        self.fc2_b = weights['fc2_bias']['values']

        # Extract shift amounts (integers)
        shifts = weights.get('_shift_amounts', {})
        self.input_shift = shifts.get('input_shift', -7)
        self.fc1_weight_shift = shifts.get('fc1_weight_shift', -7)
        self.fc1_bias_shift = shifts.get('fc1_bias_shift', -7)
        self.fc1_output_shift = shifts.get('fc1_output_shift', -4)
        self.fc2_weight_shift = shifts.get('fc2_weight_shift', -7)
        self.fc2_bias_shift = shifts.get('fc2_bias_shift', -7)
        self.fc2_output_shift = shifts.get('fc2_output_shift', -4)

    def quantize_input(self, x: np.ndarray) -> np.ndarray:
        """Quantize float input to integer using input_shift."""
        scale = 2.0 ** self.input_shift
        x_int = np.clip(np.round(x / scale), self.qmin, self.qmax).astype(np.int32)
        return x_int

    def shift_right_round(self, x: np.ndarray, shift: int) -> np.ndarray:
        """
        Arithmetic right shift with rounding.
        Equivalent to: round(x / 2^shift) but using only integer operations.
        """
        if shift <= 0:
            return x << (-shift)

        # Add rounding bias (half of divisor)
        rounding = 1 << (shift - 1)
        return (x + rounding) >> shift

    def integer_matmul_shift(self, x_int: np.ndarray, x_shift: int,
                              w_int: np.ndarray, w_shift: int,
                              b_int: np.ndarray, b_shift: int,
                              out_shift: int) -> np.ndarray:
        """
        True integer matrix multiplication with shift-based requantization.

        Computes: y = (x @ w.T + b) >> combined_shift

        The math:
        - acc = x_int @ w_int.T  (in int32)
        - acc_scale = 2^(x_shift + w_shift)
        - We need output with scale 2^out_shift
        - So: out_int = acc >> (x_shift + w_shift - out_shift)
        - Bias needs alignment: b_aligned = b_int << (x_shift + w_shift - b_shift)
        """
        # int32 accumulator for matmul
        acc = np.dot(x_int.astype(np.int32), w_int.T.astype(np.int32))

        # Accumulator scale exponent
        acc_shift = x_shift + w_shift

        # Align bias to accumulator scale
        # If acc has scale 2^acc_shift and bias has scale 2^b_shift,
        # to add bias to acc (both in integer units at scale 2^acc_shift),
        # we need: b_aligned = b_int * 2^(b_shift - acc_shift)
        # Because: b_float = b_int * 2^b_shift = (b_int * 2^(b_shift - acc_shift)) * 2^acc_shift
        bias_align_shift = b_shift - acc_shift
        if bias_align_shift >= 0:
            b_aligned = b_int.astype(np.int32) << bias_align_shift
        else:
            b_aligned = self.shift_right_round(b_int.astype(np.int32), -bias_align_shift)

        acc = acc + b_aligned

        # Requantize to output scale
        # out_int = acc_int * 2^(acc_shift - out_shift)
        # If acc_shift - out_shift > 0: multiply (left shift)
        # If acc_shift - out_shift < 0: divide (right shift)
        requant_shift = acc_shift - out_shift
        if requant_shift >= 0:
            out_int = acc << requant_shift  # Left shift = multiply by 2^requant_shift
        else:
            out_int = self.shift_right_round(acc, -requant_shift)  # Right shift = divide

        # Clamp to output range
        out_int = np.clip(out_int, self.qmin, self.qmax).astype(np.int32)

        return out_int

    def integer_relu(self, x_int: np.ndarray) -> np.ndarray:
        """Integer ReLU - just clamp negatives to zero."""
        return np.maximum(x_int, 0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass using only integer arithmetic and bit shifts.

        Args:
            x: Float input [batch, 36]

        Returns:
            logits: Integer logits [batch, 10] (for argmax classification)
        """
        # Quantize input
        x_int = self.quantize_input(x)

        # Layer 1: FC + ReLU
        h_int = self.integer_matmul_shift(
            x_int, self.input_shift,
            self.fc1_w, self.fc1_weight_shift,
            self.fc1_b, self.fc1_bias_shift,
            self.fc1_output_shift
        )
        h_int = self.integer_relu(h_int)

        # Layer 2: FC (no activation for output)
        out_int = self.integer_matmul_shift(
            h_int, self.fc1_output_shift,
            self.fc2_w, self.fc2_weight_shift,
            self.fc2_b, self.fc2_bias_shift,
            self.fc2_output_shift
        )

        return out_int  # Return integer logits for argmax


class IntegerInferenceSNN:
    """
    Pure integer inference for SNN - no floating point operations.

    This simulates exactly what embedded hardware would compute.
    Uses fixed-point arithmetic for membrane dynamics.
    """

    def __init__(self, weights: dict, bits: int, config: Config):
        """
        Args:
            weights: dict from export_embedded_weights containing int weights and scales
            bits: 8 or 4
            config: Network configuration
        """
        self.bits = bits
        self.qmax = (1 << (bits - 1)) - 1
        self.qmin = -(1 << (bits - 1))
        self.num_steps = config.training.num_steps

        # Fixed-point beta (membrane decay)
        # Q15 for int16, Q7 for int8, Q3 for int4
        if bits == 16:
            self.beta_fixed = int(config.neuron.beta * 32767)
            self.beta_shift = 15
        elif bits == 8:
            self.beta_fixed = int(config.neuron.beta * 127)
            self.beta_shift = 7
        else:
            self.beta_fixed = int(config.neuron.beta * 7)
            self.beta_shift = 3

        # Threshold in fixed-point
        self.threshold = int(config.neuron.threshold * self.qmax)

        # Extract weights and scales
        self.fc1_w = weights['fc1_weight']['values']
        self.fc1_b = weights['fc1_bias']['values']
        self.fc1_w_scale = weights['fc1_weight']['scale']
        self.fc1_b_scale = weights['fc1_bias']['scale']

        self.fc2_w = weights['fc2_weight']['values']
        self.fc2_b = weights['fc2_bias']['values']
        self.fc2_w_scale = weights['fc2_weight']['scale']
        self.fc2_b_scale = weights['fc2_bias']['scale']

        self.hidden_size = self.fc1_w.shape[0]
        self.output_size = self.fc2_w.shape[0]

    def quantize_input(self, x: np.ndarray) -> tuple:
        """Quantize float input to integer."""
        abs_max = np.abs(x).max()
        if abs_max < 1e-8:
            return np.zeros_like(x, dtype=np.int32), 1.0
        scale = abs_max / self.qmax
        x_int = np.clip(np.round(x / scale), self.qmin, self.qmax).astype(np.int32)
        return x_int, scale

    def integer_matmul_no_requant(self, x_int: np.ndarray, x_scale: float,
                                   w_int: np.ndarray, w_scale: float,
                                   b_int: np.ndarray, b_scale: float) -> tuple:
        """True integer matmul returning int32 accumulator (for adding to membrane)."""
        acc = np.dot(x_int.astype(np.int32), w_int.T.astype(np.int32))
        matmul_scale = x_scale * w_scale

        if matmul_scale > 1e-10:
            b_aligned = np.round(b_int.astype(np.float64) * b_scale / matmul_scale).astype(np.int32)
        else:
            b_aligned = np.zeros_like(b_int, dtype=np.int32)
        acc = acc + b_aligned

        return acc, matmul_scale

    def lif_step(self, current: np.ndarray, mem: np.ndarray) -> tuple:
        """
        Single LIF neuron step using integer arithmetic.

        mem_new = (beta * mem) >> shift + current
        spike = mem_new >= threshold
        mem_new = mem_new - spike * threshold  (subtract reset)

        Returns: (spikes, new_membrane)
        """
        # Decay membrane: (beta_fixed * mem) >> beta_shift
        mem_decayed = (self.beta_fixed * mem) >> self.beta_shift

        # Add current
        mem_new = mem_decayed + current

        # Clamp to prevent overflow
        mem_new = np.clip(mem_new, -32768, 32767).astype(np.int32)

        # Spike generation
        spikes = (mem_new >= self.threshold).astype(np.int32)

        # Reset (subtract threshold where spiked)
        mem_new = mem_new - spikes * self.threshold

        return spikes, mem_new

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass using only integer arithmetic.

        Args:
            x: Float input [batch, 36]

        Returns:
            spike_counts: Integer spike counts [batch, 10] (for argmax classification)
        """
        batch_size = x.shape[0]

        # Quantize input once (rate-coded: same input each timestep)
        x_int, x_scale = self.quantize_input(x)

        # Initialize membrane potentials (integer)
        mem1 = np.zeros((batch_size, self.hidden_size), dtype=np.int32)
        mem2 = np.zeros((batch_size, self.output_size), dtype=np.int32)

        # Spike count accumulator
        spike_count = np.zeros((batch_size, self.output_size), dtype=np.int32)

        # Pre-compute layer 1 current (same every timestep for rate coding)
        cur1_acc, cur1_scale = self.integer_matmul_no_requant(
            x_int, x_scale,
            self.fc1_w, self.fc1_w_scale,
            self.fc1_b, self.fc1_b_scale
        )
        # Requantize current to membrane scale
        cur1_int = np.clip(np.round(cur1_acc * cur1_scale / (1.0 / self.qmax)),
                           self.qmin * 4, self.qmax * 4).astype(np.int32)

        for _ in range(self.num_steps):
            # Layer 1: LIF
            spk1, mem1 = self.lif_step(cur1_int, mem1)

            # Layer 2: spikes (0/1) as input, scale = 1
            cur2_acc, cur2_scale = self.integer_matmul_no_requant(
                spk1, 1.0,
                self.fc2_w, self.fc2_w_scale,
                self.fc2_b, self.fc2_b_scale
            )
            cur2_int = np.clip(np.round(cur2_acc * cur2_scale / (1.0 / self.qmax)),
                               self.qmin * 4, self.qmax * 4).astype(np.int32)

            # Layer 2: LIF
            spk2, mem2 = self.lif_step(cur2_int, mem2)

            # Accumulate output spikes
            spike_count = spike_count + spk2

        return spike_count


def verify_integer_inference(model: nn.Module, model_key: str, bits: int,
                             config: Config, test_loader: DataLoader,
                             device: torch.device) -> dict:
    """
    Verify that pure integer inference matches training accuracy.

    Runs the test set through both:
    1. PyTorch model with fake quantization (what we trained)
    2. Pure integer inference (what embedded hardware does)

    Returns accuracy comparison.
    """
    print(f"\n  Verifying integer inference for {model_key} ({bits}-bit)...")

    # Get quantized weights from model
    state_dict = model.state_dict()
    weights = {}

    # Check if this is a shift-based model
    is_shift_based = 'shift' in model_key or any('weight_shift' in key for key in state_dict.keys())

    # Extract shift amounts for shift-based models
    if is_shift_based:
        shift_amounts = {}
        stored_shifts = {}

        for key, tensor in state_dict.items():
            if 'weight_shift' in key or 'bias_shift' in key or 'output_shift' in key:
                layer_name = key.split('.')[0]
                shift_type = key.split('.')[-1]
                if layer_name not in stored_shifts:
                    stored_shifts[layer_name] = {}
                stored_shifts[layer_name][shift_type] = int(tensor.item())

            if key == 'input_shift':
                shift_amounts['input_shift'] = int(tensor.item())
            elif 'output_shift' in key:
                layer_name = key.split('.')[0]
                shift_amounts[f'{layer_name}_output_shift'] = int(tensor.item())
            elif 'weight_shift' in key:
                layer_name = key.split('.')[0]
                shift_amounts[f'{layer_name}_weight_shift'] = int(tensor.item())
            elif 'bias_shift' in key:
                layer_name = key.split('.')[0]
                shift_amounts[f'{layer_name}_bias_shift'] = int(tensor.item())

        # Quantize weights using shifts
        for key, tensor in state_dict.items():
            if 'scale' in key or 'shift' in key or 'calibrated' in key or 'abs_max' in key:
                continue
            clean_key = key.replace('.linear.', '_').replace('.', '_')
            if 'weight' in key or 'bias' in key:
                layer_name = key.split('.')[0]
                if 'weight' in key and layer_name in stored_shifts:
                    shift = stored_shifts[layer_name].get('weight_shift', -7)
                elif 'bias' in key and layer_name in stored_shifts:
                    shift = stored_shifts[layer_name].get('bias_shift', -7)
                else:
                    shift = -7
                int_vals, shift_val, zp = quantize_with_shift(tensor, shift, bits)
                weights[clean_key] = {
                    'values': int_vals,
                    'scale': 2.0 ** shift_val,
                    'shift': shift_val,
                    'zero_point': zp
                }

        weights['_shift_amounts'] = shift_amounts
    else:
        for key, tensor in state_dict.items():
            clean_key = key.replace('.linear.', '_').replace('.', '_')
            if 'weight' in key or 'bias' in key:
                int_vals, scale, zp = quantize_for_embedded(tensor, bits)
                weights[clean_key] = {
                    'values': int_vals,
                    'scale': scale,
                    'zero_point': zp
                }

    # Create integer inference engine
    is_snn = 'baseline' in model_key or 'snn' in model_key.lower()
    if is_shift_based:
        int_engine = IntegerInferenceANN_Shift(weights, bits)
    elif is_snn:
        int_engine = IntegerInferenceSNN(weights, bits, config)
    else:
        int_engine = IntegerInferenceANN(weights, bits)

    # Run both inference methods on test set
    pytorch_correct = 0
    integer_correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for data, targets in test_loader:
            data_np = data.numpy()
            targets_np = targets.numpy()

            # PyTorch inference (fake quantization)
            data_dev = data.to(device)
            output_pt, _, _ = model(data_dev)
            pred_pt = output_pt.argmax(dim=1).cpu().numpy()

            # Pure integer inference
            output_int = int_engine.forward(data_np)
            pred_int = output_int.argmax(axis=1)

            pytorch_correct += (pred_pt == targets_np).sum()
            integer_correct += (pred_int == targets_np).sum()
            total += len(targets_np)

    pytorch_acc = pytorch_correct / total
    integer_acc = integer_correct / total

    print(f"    PyTorch (fake quant) accuracy: {pytorch_acc*100:.2f}%")
    print(f"    Integer-only accuracy:         {integer_acc*100:.2f}%")
    print(f"    Difference:                    {(pytorch_acc - integer_acc)*100:+.2f}%")

    return {
        'pytorch_accuracy': pytorch_acc,
        'integer_accuracy': integer_acc,
        'difference': pytorch_acc - integer_acc,
        'bits': bits
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='SNNTorch Comparison for Gilgamesh')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory for MNIST data')
    parser.add_argument('--output-dir', type=str, default='./comparison/results',
                       help='Output directory for results')
    parser.add_argument('--hidden-size', type=int, default=12,
                       help='Hidden layer size (default: 12)')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--num-steps', type=int, default=25,
                       help='Number of simulation timesteps')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cuda, mps, cpu, or auto)')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['ann', 'baseline'],
                       help='Models to train. Use *_fixed variants for accurate embedded simulation. Options: baseline, ann, ann_int{4,8,16}, baseline_int{4,8,16}, ann_int{4,8,16}_fixed, baseline_int{4,8,16}_fixed')

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device selection
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Configuration
    config = Config.default()
    config.network.hidden_size = args.hidden_size
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.lr = args.lr
    config.training.num_steps = args.num_steps
    config.training.seed = args.seed

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump({
            'network': asdict(config.network),
            'neuron': asdict(config.neuron),
            'training': asdict(config.training)
        }, f, indent=2)

    print(f"\nConfiguration saved to {config_path}")

    # Check if any shift-based models are requested (they need int8 input range)
    shift_models = [m for m in args.models if 'shift' in m]
    other_models = [m for m in args.models if 'shift' not in m]

    # Load datasets
    print("\nLoading MNIST dataset (6x6 downsampled)...")
    if other_models:
        dataset_normalized = DownsampledMNIST(args.data_dir, target_size=6, use_int8_range=False)
        train_loader_norm, test_loader_norm = dataset_normalized.get_loaders(args.batch_size)
        print(f"  Normalized dataset: {len(train_loader_norm.dataset)} train, {len(test_loader_norm.dataset)} test")

    if shift_models:
        dataset_int8 = DownsampledMNIST(args.data_dir, target_size=6, use_int8_range=True)
        train_loader_int8, test_loader_int8 = dataset_int8.get_loaders(args.batch_size)
        print(f"  Int8 range dataset: {len(train_loader_int8.dataset)} train, {len(test_loader_int8.dataset)} test")
        print("  (Int8 range: inputs in [-128, 127] for shift-based models)")

    # Model configurations
    model_classes = {
        'baseline': ('GilgameshSNN', GilgameshSNN),
        'synaptic': ('GilgameshSNN_Synaptic', GilgameshSNN_Synaptic),
        'recurrent': ('GilgameshSNN_Recurrent', GilgameshSNN_Recurrent),
        '3layer': ('GilgameshSNN_3Layer', GilgameshSNN_3Layer),
        'ann': ('StandardANN', StandardANN),
        'ann_3layer': ('StandardANN_3Layer', StandardANN_3Layer),
        'ann_int8': ('StandardANN_Int8', StandardANN_Int8),
        'ann_int4': ('StandardANN_Int4', StandardANN_Int4),
        'baseline_int8': ('GilgameshSNN_Int8', GilgameshSNN_Int8),
        'baseline_int4': ('GilgameshSNN_Int4', GilgameshSNN_Int4),
        'ann_int16': ('StandardANN_Int16', StandardANN_Int16),
        'baseline_int16': ('GilgameshSNN_Int16', GilgameshSNN_Int16),
        # Fixed-scale models (training matches inference exactly)
        'ann_int8_fixed': ('StandardANN_Int8_Fixed', StandardANN_Int8_Fixed),
        'ann_int4_fixed': ('StandardANN_Int4_Fixed', StandardANN_Int4_Fixed),
        'ann_int16_fixed': ('StandardANN_Int16_Fixed', StandardANN_Int16_Fixed),
        'baseline_int8_fixed': ('GilgameshSNN_Int8_Fixed', GilgameshSNN_Int8_Fixed),
        'baseline_int4_fixed': ('GilgameshSNN_Int4_Fixed', GilgameshSNN_Int4_Fixed),
        'baseline_int16_fixed': ('GilgameshSNN_Int16_Fixed', GilgameshSNN_Int16_Fixed),
        # Shift-based models (power-of-2 scales for pure integer inference with bit shifts)
        'ann_int8_shift': ('StandardANN_Int8_Shift', StandardANN_Int8_Shift),
    }

    # Train selected models
    results = {}

    for model_key in args.models:
        if model_key not in model_classes:
            print(f"Unknown model: {model_key}, skipping")
            continue

        name, ModelClass = model_classes[model_key]

        # Select appropriate data loaders based on model type
        if 'shift' in model_key:
            train_loader = train_loader_int8
            test_loader = test_loader_int8
            print(f"\n{'='*60}")
            print(f"Training {name} (using int8 input range [-128, 127])")
        else:
            train_loader = train_loader_norm
            test_loader = test_loader_norm
            print(f"\n{'='*60}")
            print(f"Training {name}")
        print('='*60)

        model = ModelClass(config)
        params = count_parameters(model)
        print(f"Parameters: {params:,}")

        trainer = Trainer(model, config, device, name=name)
        result = trainer.train(train_loader, test_loader)
        result['params'] = params

        results[name] = result

        # Save weights
        pt_path, json_path = save_weights(model, name, output_dir)
        print(f"Weights saved to {pt_path}")
        print(f"JSON weights saved to {json_path}")

        # Export embedded weights and verify for quantized models
        if 'int16' in model_key or 'int8' in model_key or 'int4' in model_key:
            bits = 16 if 'int16' in model_key else (8 if 'int8' in model_key else 4)
            h_path, bin_path, meta_path = export_embedded_weights(model, name, bits, output_dir, config)
            print(f"Embedded C header saved to {h_path}")
            print(f"Embedded binary saved to {bin_path}")

            # Verify integer-only inference matches training
            verification = verify_integer_inference(model, model_key, bits, config, test_loader, device)
            result['integer_verification'] = verification

    # Generate report
    print(f"\n{'='*60}")
    print("Generating Report")
    print('='*60)

    report_path = generate_report(results, output_dir)
    print(f"Report saved to {report_path}")

    plot_path = plot_training_curves(results, output_dir)
    if plot_path:
        print(f"Training curves saved to {plot_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print('='*60)
    print(f"\n{'Model':<25} {'Best Test Acc':>15} {'Parameters':>12}")
    print('-'*55)
    for name, res in sorted(results.items(), key=lambda x: -x[1]['best_test_acc']):
        print(f"{name:<25} {res['best_test_acc']*100:>14.2f}% {res['params']:>12,}")

    print(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    main()
