# Arduino Int8 Neural Network Inference

This document explains how to implement the trained neural network on Arduino using pure integer arithmetic with bit shifts. No floating-point operations are required.

## Network Architecture

```
Input Layer      Hidden Layer       Output Layer
   [36]      →      [20]        →      [10]
  (6x6 image)    (FC1 + ReLU)      (FC2 + argmax)
```

- **Input**: 36 int8 values representing a 6x6 grayscale image
- **Hidden**: 20 neurons with ReLU activation
- **Output**: 10 neurons (digits 0-9), classification by argmax

## Data Format

### Input Data
- **Type**: `int8_t` (signed 8-bit integer)
- **Range**: [-128, 127]
- **Mapping**:
  - -128 = black (pixel value 0)
  - 127 = white (pixel value 255)
- **Conversion**: `int8_value = pixel_value - 128`

### Weights and Biases
- **Type**: `int8_t` (signed 8-bit integer)
- **Range**: [-128, 127]
- **Storage**: Stored in PROGMEM (flash memory) to save RAM

### Activations
- **Type**: `int8_t` for layer outputs
- **Accumulator**: `int32_t` during matrix multiplication (to prevent overflow)

## Quantization Scheme

All values use power-of-2 scales for efficient bit-shift operations:

```
real_value = int_value × 2^shift
```

Where `shift` is a signed integer (typically negative for values < 1).

### Scale Relationships

| Value | Scale Formula | Typical Shift |
|-------|---------------|---------------|
| Input | 2^input_shift | 0 (raw int8) |
| FC1 Weights | 2^fc1_weight_shift | ~ -7 |
| FC1 Bias | 2^fc1_bias_shift | ~ -7 |
| FC1 Output | 2^fc1_output_shift | ~ -4 |
| FC2 Weights | 2^fc2_weight_shift | ~ -7 |
| FC2 Bias | 2^fc2_bias_shift | ~ -7 |
| FC2 Output | 2^fc2_output_shift | ~ -4 |

## Inference Algorithm

### Helper Function: apply_shift

The shift values from the header use this convention:
- **Positive shift** = left shift (multiply by 2^shift)
- **Negative shift** = right shift with rounding (divide by 2^|shift|)

```cpp
// Apply shift with rounding (positive = left shift, negative = right shift)
static inline int32_t apply_shift(int32_t x, int shift) {
    if (shift >= 0) {
        return x << shift;
    } else {
        int right_shift = -shift;
        int32_t rounding = 1 << (right_shift - 1);
        return (x + rounding) >> right_shift;
    }
}
```

### Step 1: Load Input
```cpp
// Input is already int8, no conversion needed
int8_t input[36];  // Loaded from serial/memory
```

### Step 2: FC1 Layer (Input → Hidden)

```cpp
// For each hidden neuron
for (int n = 0; n < HIDDEN_SIZE; n++) {
    // Accumulate in int32 to prevent overflow
    int32_t acc = 0;

    // Matrix multiplication: input × weights (weights in RAM)
    for (int i = 0; i < INPUT_SIZE; i++) {
        acc += (int32_t)input[i] * (int32_t)fc1_weight[n * INPUT_SIZE + i];
    }

    // Add bias (aligned to accumulator scale)
    // FC1_BIAS_ALIGN_SHIFT = bias_shift - acc_shift
    // Positive = left shift, Negative = right shift
    int32_t bias_aligned = apply_shift((int32_t)fc1_bias[n], FC1_BIAS_ALIGN_SHIFT);
    acc += bias_aligned;

    // Requantize to output scale
    // FC1_REQUANT_SHIFT = acc_shift - output_shift
    // Positive = left shift (multiply), Negative = right shift (divide)
    int32_t result = apply_shift(acc, FC1_REQUANT_SHIFT);

    // Clamp to int8 range
    result = max(-128, min(127, result));

    // ReLU activation
    hidden[n] = (int8_t)max(0, result);
}
```

### Step 3: FC2 Layer (Hidden → Output)

```cpp
// For each output neuron
for (int n = 0; n < OUTPUT_SIZE; n++) {
    int32_t acc = 0;

    // Matrix multiplication
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        acc += (int32_t)hidden[i] * (int32_t)fc2_weight[n * HIDDEN_SIZE + i];
    }

    // Add aligned bias (same convention as FC1)
    int32_t bias_aligned = apply_shift((int32_t)fc2_bias[n], FC2_BIAS_ALIGN_SHIFT);
    acc += bias_aligned;

    // Requantize to output scale
    int32_t result = apply_shift(acc, FC2_REQUANT_SHIFT);

    // Clamp to int8 (NO ReLU on output layer!)
    output[n] = (int8_t)max(-128, min(127, result));
}
```

### Step 4: Classification

```cpp
// Find the neuron with highest activation (argmax)
int8_t max_val = output[0];
uint8_t prediction = 0;

for (int i = 1; i < OUTPUT_SIZE; i++) {
    if (output[i] > max_val) {
        max_val = output[i];
        prediction = i;
    }
}

// prediction is now the classified digit (0-9)
```

## Memory Layout

### RAM Usage
```
Weights and Biases (stored in RAM for fast access):
fc1_weight[720]  = 720 bytes  (36 × 20)
fc1_bias[20]     =  20 bytes
fc2_weight[200]  = 200 bytes  (20 × 10)
fc2_bias[10]     =  10 bytes
─────────────────────────────
Subtotal         = 950 bytes

Runtime buffers:
input[36]        =  36 bytes
hidden[20]       =  20 bytes
output[10]       =  10 bytes
accumulators     =   4 bytes (reused)
─────────────────────────────
Subtotal         ~  70 bytes

Total RAM        ~ 1020 bytes
```

Note: If RAM is constrained, weights can be moved to Flash using PROGMEM
and accessed with `pgm_read_byte()`. This saves ~950 bytes of RAM but
requires slightly more code.

## Shift Calculation Details

### Why Shifts Work

When we multiply two quantized values:
```
real_a = int_a × 2^shift_a
real_b = int_b × 2^shift_b
real_product = real_a × real_b = int_a × int_b × 2^(shift_a + shift_b)
```

The accumulator holds `int_a × int_b` with implicit scale `2^(shift_a + shift_b)`.

To requantize to a new scale:
```
new_int = old_int × 2^(old_shift - new_shift)
        = old_int >> (new_shift - old_shift)  // if new_shift > old_shift
        = old_int << (old_shift - new_shift)  // if new_shift < old_shift
```

### Bias Alignment

Bias must be scaled to match the accumulator before adding:
```
acc_shift = input_shift + weight_shift
bias_scale = 2^bias_shift
acc_scale = 2^acc_shift

To add bias to acc, both must be in the same "units":
  bias_float = bias_int × 2^bias_shift
             = (bias_int × 2^(bias_shift - acc_shift)) × 2^acc_shift

So: bias_aligned = bias_int × 2^(bias_shift - acc_shift)
                 = apply_shift(bias_int, bias_shift - acc_shift)

In the header: BIAS_ALIGN_SHIFT = bias_shift - acc_shift
```

## Complete Example Implementation

```cpp
#include "weights.h"  // Generated header with weights and shifts

#define INPUT_SIZE  36
#define HIDDEN_SIZE 20
#define OUTPUT_SIZE 10

// Clamp value to int8 range
static inline int8_t clamp_i8(int32_t x) {
    if (x < -128) return -128;
    if (x > 127) return 127;
    return (int8_t)x;
}

// Apply shift with rounding (positive = left shift, negative = right shift)
// Convention: shift > 0 means multiply by 2^shift (left shift)
//            shift < 0 means divide by 2^|shift| (right shift with rounding)
static inline int32_t apply_shift(int32_t x, int shift) {
    if (shift >= 0) {
        return x << shift;
    } else {
        int right_shift = -shift;
        int32_t rounding = 1 << (right_shift - 1);
        return (x + rounding) >> right_shift;
    }
}

uint8_t predict(const int8_t* input) {
    int8_t hidden[HIDDEN_SIZE];
    int8_t output[OUTPUT_SIZE];

    // Layer 1: FC + ReLU
    for (int n = 0; n < HIDDEN_SIZE; n++) {
        int32_t acc = 0;
        for (int i = 0; i < INPUT_SIZE; i++) {
            acc += (int32_t)input[i] * (int32_t)fc1_weight[n * INPUT_SIZE + i];
        }
        // Align bias to accumulator scale, then add
        int32_t bias_aligned = apply_shift((int32_t)fc1_bias[n], FC1_BIAS_ALIGN_SHIFT);
        acc += bias_aligned;
        // Requantize to output scale
        int32_t h = apply_shift(acc, FC1_REQUANT_SHIFT);
        hidden[n] = (h > 0) ? clamp_i8(h) : 0;  // ReLU + clamp
    }

    // Layer 2: FC (no ReLU)
    for (int n = 0; n < OUTPUT_SIZE; n++) {
        int32_t acc = 0;
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            acc += (int32_t)hidden[i] * (int32_t)fc2_weight[n * HIDDEN_SIZE + i];
        }
        int32_t bias_aligned = apply_shift((int32_t)fc2_bias[n], FC2_BIAS_ALIGN_SHIFT);
        acc += bias_aligned;
        output[n] = clamp_i8(apply_shift(acc, FC2_REQUANT_SHIFT));
    }

    // Argmax
    int8_t max_val = output[0];
    uint8_t prediction = 0;
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (output[i] > max_val) {
            max_val = output[i];
            prediction = i;
        }
    }

    return prediction;
}
```

## Data Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TRAINING (PC)                                │
├─────────────────────────────────────────────────────────────────────┤
│  MNIST Images (28×28)                                               │
│       ↓ downsample                                                  │
│  6×6 images, normalized to [-128, 127]                             │
│       ↓ train                                                       │
│  PyTorch model with quantization-aware training                     │
│       ↓ export                                                      │
│  C header file with int8 weights + shift amounts                    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        DEPLOYMENT (Arduino)                          │
├─────────────────────────────────────────────────────────────────────┤
│  PNG Image                                                          │
│       ↓ convert_data.py                                             │
│  36 int8 bytes [-128, 127]                                         │
│       ↓ serial transfer                                             │
│  Arduino receives int8 input (no conversion!)                       │
│       ↓ FC1 matmul (int32 acc)                                      │
│       ↓ >> shift + clamp + ReLU                                     │
│  int8 hidden[20]                                                    │
│       ↓ FC2 matmul (int32 acc)                                      │
│       ↓ >> shift + clamp                                            │
│  int8 output[10]                                                    │
│       ↓ argmax                                                      │
│  Predicted digit (0-9)                                              │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Implementation Notes

1. **No floating point**: All operations use int8/int32 and bit shifts
2. **Accumulator width**: Must use int32 for matmul to prevent overflow
3. **ReLU placement**: Only after hidden layer, NOT after output layer
4. **Bias addition**: Add bias AFTER matmul, not multiply
5. **Rounding**: For right shifts, add `1 << (shift-1)` before shifting for proper rounding
6. **Shift convention**: Positive shift = left shift (multiply), Negative shift = right shift (divide)
7. **Weight storage**: Weights stored in RAM (~950 bytes) for fast access
8. **Input format**: Inputs are already int8, no runtime quantization needed

## Verification

To verify your Arduino implementation matches the training:

1. Export test samples from Python with known labels
2. Run inference on Arduino
3. Compare predictions

Expected accuracy: ~85-92% on MNIST test set (depends on training)
