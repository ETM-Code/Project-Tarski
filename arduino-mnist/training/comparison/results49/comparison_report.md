# SNNTorch Comparison Report

Generated: 2026-01-20 16:43:29

## Configuration (matching gilgamesh)

| Parameter | Value |
|-----------|-------|
| Input Size | 49 (7×7 MNIST) |
| Hidden Size | 100 |
| Output Size | 10 |
| Beta (decay) | 0.9 |
| Threshold | 1.0 |
| Surrogate Slope | 25.0 |
| Learning Rate | 0.001 → 0.00001 (cosine) |
| Epochs | 15 |
| Batch Size | 128 |
| Timesteps | 25 |
| Optimizer | Adam (β1=0.9, β2=0.999) |

## Results Summary

| Model | Best Test Acc | Final Train Acc | Final Test Acc | Parameters |
|-------|---------------|-----------------|----------------|------------|
| GilgameshSNN | 91.31% | 90.45% | 91.22% | 550 |
| GilgameshSNN_Synaptic | 73.13% | 73.56% | 72.65% | 550 |
| GilgameshSNN_Recurrent | 91.27% | 90.82% | 91.27% | 631 |
| GilgameshSNN_3Layer | 87.85% | 87.01% | 87.85% | 540 |
| StandardANN | 89.71% | 88.98% | 89.71% | 550 |
| StandardANN_3Layer | 78.63% | 77.91% | 78.63% | 540 |

## Model Descriptions

### Spiking Neural Networks (SNNs)

#### GilgameshSNN (Baseline)
Standard 2-layer feedforward LIF SNN matching gilgamesh architecture exactly.
Architecture: Input(49) → FC → LIF(100) → FC → LIF(10)

#### GilgameshSNN_Synaptic
Uses dual exponential synaptic dynamics with separate synaptic and membrane time constants.

#### GilgameshSNN_Recurrent
Adds recurrent connections within the hidden layer for temporal processing.

#### GilgameshSNN_3Layer
Deeper SNN with 3 layers: Input(49) → LIF(100) → LIF(50) → LIF(10)

### Artificial Neural Networks (ANNs)

#### StandardANN
Standard 2-layer feedforward ANN with ReLU activations (non-spiking baseline).
Architecture: Input(49) → FC → ReLU(100) → FC → Output(10)

#### StandardANN_3Layer
Deeper ANN with 3 layers: Input(49) → ReLU(100) → ReLU(50) → Output(10)

## Training Curves

See `training_curves.png` for visualization of training and test accuracy over epochs.

## Weight Files

- `GilgameshSNN_weights.pt`: PyTorch state dict
- `GilgameshSNN_weights.json`: Weights exported as JSON (for gilgamesh compatibility)
- `GilgameshSNN_Synaptic_weights.pt`: PyTorch state dict
- `GilgameshSNN_Synaptic_weights.json`: Weights exported as JSON (for gilgamesh compatibility)
- `GilgameshSNN_Recurrent_weights.pt`: PyTorch state dict
- `GilgameshSNN_Recurrent_weights.json`: Weights exported as JSON (for gilgamesh compatibility)
- `GilgameshSNN_3Layer_weights.pt`: PyTorch state dict
- `GilgameshSNN_3Layer_weights.json`: Weights exported as JSON (for gilgamesh compatibility)
- `StandardANN_weights.pt`: PyTorch state dict
- `StandardANN_weights.json`: Weights exported as JSON (for gilgamesh compatibility)
- `StandardANN_3Layer_weights.pt`: PyTorch state dict
- `StandardANN_3Layer_weights.json`: Weights exported as JSON (for gilgamesh compatibility)

## Notes

- All models use rate-coded input (same input at each timestep)
- Loss: Cross-entropy on spike counts
- Best weights (highest test accuracy) are saved
- JSON exports use gilgamesh-compatible format
