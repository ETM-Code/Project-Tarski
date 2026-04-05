# SNNTorch Comparison Report

Generated: 2026-02-03 11:49:09

## Configuration (matching gilgamesh)

| Parameter | Value |
|-----------|-------|
| Input Size | 36 (6×6 MNIST) |
| Hidden Size | 12 |
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
| StandardANN_Int8_Shift | 86.91% | 86.41% | 86.91% | 668 |

## Model Descriptions

### Spiking Neural Networks (SNNs)

#### GilgameshSNN (Baseline)
Standard 2-layer feedforward LIF SNN matching gilgamesh architecture exactly.
Architecture: Input(36) → FC → LIF(12) → FC → LIF(10)

#### GilgameshSNN_Synaptic
Uses dual exponential synaptic dynamics with separate synaptic and membrane time constants.

#### GilgameshSNN_Recurrent
Adds recurrent connections within the hidden layer for temporal processing.

#### GilgameshSNN_3Layer
Deeper SNN with 3 layers: Input(36) → LIF(12) → LIF(6) → LIF(10)

### Artificial Neural Networks (ANNs)

#### StandardANN
Standard 2-layer feedforward ANN with ReLU activations (non-spiking baseline).
Architecture: Input(36) → FC → ReLU(12) → FC → Output(10)

#### StandardANN_3Layer
Deeper ANN with 3 layers: Input(36) → ReLU(12) → ReLU(6) → Output(10)

## Training Curves

See `training_curves.png` for visualization of training and test accuracy over epochs.

## Weight Files

- `StandardANN_Int8_Shift_weights.pt`: PyTorch state dict
- `StandardANN_Int8_Shift_weights.json`: Weights exported as JSON (for gilgamesh compatibility)

## Notes

- All models use rate-coded input (same input at each timestep)
- Loss: Cross-entropy on spike counts
- Best weights (highest test accuracy) are saved
- JSON exports use gilgamesh-compatible format
