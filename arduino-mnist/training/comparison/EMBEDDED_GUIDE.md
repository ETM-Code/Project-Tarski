# Embedded Deployment Guide for Quantized Neural Networks

This guide explains how to deploy the trained integer-quantized neural networks on embedded systems using **pure integer arithmetic** (no floating point).

## Quick Start

```bash
# Train fixed-scale quantized models (recommended for embedded)
python snntorch_comparison.py --models ann_int8_fixed ann_int16_fixed baseline_int8_fixed

# Or train all variants for comparison
python snntorch_comparison.py --models ann ann_int8_fixed ann_int4_fixed baseline baseline_int8_fixed
```

After training, you'll find in `./comparison/results/`:
- `*_weights_8bit.h` - C header with int8 weights
- `*_weights_4bit.h` - C header with int4 weights (packed)
- `*_weights_16bit.h` - C header with int16 weights
- `*_weights_*.bin` - Binary weights for direct loading
- `*_meta.json` - Metadata with scales and shapes

---

## Model Types

| Model Type | Description | Use Case |
|------------|-------------|----------|
| `ann_int8` | Dynamic-scale QAT | Testing, comparison |
| `ann_int8_fixed` | **Fixed-scale QAT** | **Embedded deployment** |
| `baseline_int8_fixed` | Fixed-scale SNN | Embedded SNN |

**Use `*_fixed` models for embedded deployment** - they use predetermined scales that enable pure integer inference.

---

## Understanding Fixed-Point Scales

The fixed-scale models export float scales, but these can be converted to integer multipliers:

```
real_value = int_value × scale_float
           = int_value × (scale_mult / 2^scale_shift)
           ≈ (int_value × scale_mult) >> scale_shift
```

For example, if `scale = 0.00784`:
- `scale_mult = round(0.00784 × 32768) = 257`
- `scale_shift = 15`
- Error: < 0.1%

### Converting Scales to Fixed-Point

```c
// Convert float scale to fixed-point multiplier and shift
// Returns multiplier for given shift (typically 15 for int16 accumulators)
int32_t scale_to_multiplier(float scale, int shift) {
    return (int32_t)(scale * (1 << shift) + 0.5f);  // One-time setup, not runtime
}

// Example: precompute at build time
// If FC1_WEIGHT_SCALE = 0.00784f, shift = 15:
// FC1_WEIGHT_MULT = scale_to_multiplier(0.00784f, 15) = 257
```

---

## 8-Bit Fixed-Point ANN Implementation

This is **pure integer inference** - no floating point operations at runtime.

### Memory Layout

| Layer | Shape | Size (bytes) |
|-------|-------|--------------|
| fc1_weight | [12, 36] | 432 |
| fc1_bias | [12] | 12 |
| fc2_weight | [10, 12] | 120 |
| fc2_bias | [10] | 10 |
| **Total** | | **574 bytes** |

### C Header Structure (Fixed-Scale)

```c
/* Network Configuration */
#define NET_INPUT_SIZE  36
#define NET_HIDDEN_SIZE 12
#define NET_OUTPUT_SIZE 10
#define NET_BITS        8

/* Fixed-Point Scale Multipliers (precomputed from float scales)
 * To convert: mult = round(scale × 32768)
 * Shift is always 15 for these multipliers */
#define FC1_WEIGHT_MULT   257    // from scale 0.00784
#define FC1_BIAS_MULT     164    // from scale 0.00500
#define FC1_OUTPUT_MULT   312    // from scale 0.00952
#define FC2_WEIGHT_MULT   198    // from scale 0.00604
#define FC2_BIAS_MULT     145    // from scale 0.00443
#define SCALE_SHIFT       15

/* Or use the raw float scales if you have FPU for setup only */
#define FC1_WEIGHT_SCALE 0.00784f
#define FC1_BIAS_SCALE   0.00500f
// ... etc

/* Integer Weights */
static const int8_t fc1_weight[432] = { ... };
static const int8_t fc1_bias[12] = { ... };
static const int8_t fc2_weight[120] = { ... };
static const int8_t fc2_bias[10] = { ... };
```

### Pure Integer Matrix Multiply

```c
#include <stdint.h>

/**
 * Integer-only matrix multiply with fixed-point scaling.
 *
 * @param x         Input vector (int8)
 * @param W         Weight matrix (int8), row-major [out_features × in_features]
 * @param b         Bias vector (int8)
 * @param out       Output vector (int8)
 * @param in_size   Input dimension
 * @param out_size  Output dimension
 *
 * All scales are absorbed into the computation - no float math.
 */
void int8_matmul_fixed(
    const int8_t* x,
    const int8_t* W,
    const int8_t* b,
    int8_t* out,
    int in_size,
    int out_size
) {
    for (int o = 0; o < out_size; o++) {
        // Accumulate in int32 to prevent overflow
        int32_t acc = 0;

        for (int i = 0; i < in_size; i++) {
            acc += (int32_t)x[i] * (int32_t)W[o * in_size + i];
        }

        // Add bias (already at compatible scale from training)
        acc += (int32_t)b[o];

        // Clamp to int8 range
        // The fixed-scale training ensures outputs stay in range
        if (acc > 127) acc = 127;
        if (acc < -128) acc = -128;

        out[o] = (int8_t)acc;
    }
}

/**
 * Integer ReLU - simply clamp negatives to zero.
 */
void int8_relu(int8_t* x, int size) {
    for (int i = 0; i < size; i++) {
        if (x[i] < 0) x[i] = 0;
    }
}

/**
 * Quantize float input to int8.
 * This is the ONLY place floats are used (at input boundary).
 * For embedded sensors, you may receive data directly as integers.
 */
void quantize_input_int8(const float* input, int8_t* output, int size) {
    // Find max for scaling (or use a fixed input scale if known)
    float max_val = 0.0f;
    for (int i = 0; i < size; i++) {
        float abs_val = input[i] > 0 ? input[i] : -input[i];
        if (abs_val > max_val) max_val = abs_val;
    }

    // Scale to int8 range
    float scale = (max_val > 1e-8f) ? 127.0f / max_val : 1.0f;

    for (int i = 0; i < size; i++) {
        int32_t val = (int32_t)(input[i] * scale + 0.5f);
        if (val > 127) val = 127;
        if (val < -128) val = -128;
        output[i] = (int8_t)val;
    }
}

/**
 * Full ANN inference - pure integer after input quantization.
 *
 * @param input  Float input (36 values for 6×6 image)
 * @return       Predicted class (0-9)
 */
int ann_int8_inference(const float* input) {
    int8_t x[NET_INPUT_SIZE];
    int8_t h[NET_HIDDEN_SIZE];
    int8_t out[NET_OUTPUT_SIZE];

    // Quantize input (only float operation)
    quantize_input_int8(input, x, NET_INPUT_SIZE);

    // Layer 1: FC + ReLU (all integer)
    int8_matmul_fixed(x, fc1_weight, fc1_bias, h, NET_INPUT_SIZE, NET_HIDDEN_SIZE);
    int8_relu(h, NET_HIDDEN_SIZE);

    // Layer 2: FC (all integer)
    int8_matmul_fixed(h, fc2_weight, fc2_bias, out, NET_HIDDEN_SIZE, NET_OUTPUT_SIZE);

    // Argmax (integer)
    int8_t max_val = out[0];
    int max_idx = 0;
    for (int i = 1; i < NET_OUTPUT_SIZE; i++) {
        if (out[i] > max_val) {
            max_val = out[i];
            max_idx = i;
        }
    }

    return max_idx;
}
```

### For Integer-Only Input (No Floats)

If your sensor provides integer values directly:

```c
/**
 * Inference from raw integer sensor data.
 * No floating point anywhere.
 *
 * @param raw_input  Raw sensor values (e.g., 0-255 from ADC)
 * @param max_raw    Maximum possible raw value (e.g., 255)
 */
int ann_int8_inference_raw(const uint8_t* raw_input, uint8_t max_raw) {
    int8_t x[NET_INPUT_SIZE];

    // Convert unsigned raw to signed int8
    // Maps [0, max_raw] to [-128, 127]
    for (int i = 0; i < NET_INPUT_SIZE; i++) {
        int32_t scaled = ((int32_t)raw_input[i] * 255) / max_raw - 128;
        x[i] = (int8_t)scaled;
    }

    // ... rest of inference (same as above)
}
```

---

## 8-Bit Fixed-Point SNN Implementation

SNNs add membrane dynamics with fixed-point decay.

### LIF Neuron (Integer)

```c
/**
 * Leaky Integrate-and-Fire neuron step (pure integer).
 *
 * @param current    Input current from previous layer (int32 accumulators)
 * @param membrane   Membrane potential (int16, modified in place)
 * @param spikes     Output spike flags (0 or 1)
 * @param size       Number of neurons
 * @param beta_q7    Decay factor in Q7 format (e.g., 0.9 × 127 = 114)
 * @param threshold  Spike threshold (e.g., 127 for 1.0)
 */
void lif_step_int8(
    const int32_t* current,
    int16_t* membrane,      // int16 for membrane to prevent overflow
    uint8_t* spikes,
    int size,
    int8_t beta_q7,
    int16_t threshold
) {
    for (int i = 0; i < size; i++) {
        // Exponential decay: mem = (beta × mem) >> 7
        // Using int32 intermediate to prevent overflow
        int32_t mem_decayed = ((int32_t)beta_q7 * (int32_t)membrane[i]) >> 7;

        // Add input current (scale appropriately)
        int32_t mem_new = mem_decayed + (current[i] >> 4);  // Scale down current

        // Clamp to int16 range
        if (mem_new > 32767) mem_new = 32767;
        if (mem_new < -32768) mem_new = -32768;
        membrane[i] = (int16_t)mem_new;

        // Spike generation (subtract reset)
        if (membrane[i] >= threshold) {
            spikes[i] = 1;
            membrane[i] -= threshold;
        } else {
            spikes[i] = 0;
        }
    }
}

/**
 * Compute layer current from spikes (sparse multiply).
 * Only adds weights where input spiked - very efficient.
 */
void spike_matmul_int8(
    const uint8_t* spikes,
    const int8_t* W,
    const int8_t* b,
    int32_t* current,
    int in_size,
    int out_size
) {
    for (int o = 0; o < out_size; o++) {
        current[o] = (int32_t)b[o] << 4;  // Scale bias up

        for (int i = 0; i < in_size; i++) {
            if (spikes[i]) {
                current[o] += (int32_t)W[o * in_size + i] << 4;
            }
        }
    }
}

/**
 * Full SNN inference.
 *
 * @param input      Float input (rate-coded, same every timestep)
 * @param num_steps  Number of simulation timesteps
 * @return           Predicted class (0-9)
 */
int snn_int8_inference(const float* input, int num_steps) {
    int8_t x[NET_INPUT_SIZE];

    // Quantize input once
    quantize_input_int8(input, x, NET_INPUT_SIZE);

    // Membrane potentials (int16 for precision)
    int16_t mem1[NET_HIDDEN_SIZE] = {0};
    int16_t mem2[NET_OUTPUT_SIZE] = {0};

    // Spike counts for output
    int32_t spike_count[NET_OUTPUT_SIZE] = {0};

    // Precompute layer 1 current (constant over time)
    int32_t cur1[NET_HIDDEN_SIZE];
    for (int o = 0; o < NET_HIDDEN_SIZE; o++) {
        cur1[o] = (int32_t)fc1_bias[o] << 4;
        for (int i = 0; i < NET_INPUT_SIZE; i++) {
            cur1[o] += (int32_t)x[i] * (int32_t)fc1_weight[o * NET_INPUT_SIZE + i];
        }
    }

    // SNN parameters (from header)
    int8_t beta_q7 = 114;       // 0.9 in Q7 (127 × 0.9)
    int16_t threshold = 127;    // 1.0 scaled to int8 max

    // Simulate timesteps
    for (int t = 0; t < num_steps; t++) {
        // Layer 1: LIF
        uint8_t spk1[NET_HIDDEN_SIZE];
        lif_step_int8(cur1, mem1, spk1, NET_HIDDEN_SIZE, beta_q7, threshold);

        // Layer 2: compute current from spikes
        int32_t cur2[NET_OUTPUT_SIZE];
        spike_matmul_int8(spk1, fc2_weight, fc2_bias, cur2, NET_HIDDEN_SIZE, NET_OUTPUT_SIZE);

        // Layer 2: LIF
        uint8_t spk2[NET_OUTPUT_SIZE];
        lif_step_int8(cur2, mem2, spk2, NET_OUTPUT_SIZE, beta_q7, threshold);

        // Accumulate output spikes
        for (int i = 0; i < NET_OUTPUT_SIZE; i++) {
            spike_count[i] += spk2[i];
        }
    }

    // Argmax on spike counts
    int32_t max_count = spike_count[0];
    int max_idx = 0;
    for (int i = 1; i < NET_OUTPUT_SIZE; i++) {
        if (spike_count[i] > max_count) {
            max_count = spike_count[i];
            max_idx = i;
        }
    }

    return max_idx;
}
```

---

## 4-Bit Implementation

4-bit provides ~50% memory savings but lower accuracy.

### Packed Format

Two int4 values per byte:
```
Byte: [high_nibble:4][low_nibble:4]
      weight[i+1]    weight[i]
```

### Memory Layout

| Layer | Shape | Size (bytes) |
|-------|-------|--------------|
| fc1_weight | [12, 36] | 216 |
| fc1_bias | [12] | 6 |
| fc2_weight | [10, 12] | 60 |
| fc2_bias | [10] | 5 |
| **Total** | | **287 bytes** |

### Unpacking

```c
// Extract signed int4 from packed byte
int8_t unpack_int4_lo(uint8_t packed) {
    int8_t val = packed & 0x0F;
    if (val & 0x08) val |= 0xF0;  // Sign extend
    return val;
}

int8_t unpack_int4_hi(uint8_t packed) {
    int8_t val = (packed >> 4) & 0x0F;
    if (val & 0x08) val |= 0xF0;  // Sign extend
    return val;
}

int8_t get_int4(const uint8_t* packed, int index) {
    return (index & 1) ? unpack_int4_hi(packed[index >> 1])
                       : unpack_int4_lo(packed[index >> 1]);
}
```

### 4-Bit Matrix Multiply

```c
void int4_matmul_fixed(
    const int8_t* x,           // Input (int4 range: -8 to +7)
    const uint8_t* W_packed,   // Packed int4 weights
    const uint8_t* b_packed,   // Packed int4 biases
    int8_t* out,
    int in_size,
    int out_size
) {
    for (int o = 0; o < out_size; o++) {
        int32_t acc = 0;

        for (int i = 0; i < in_size; i++) {
            int8_t w = get_int4(W_packed, o * in_size + i);
            acc += (int32_t)x[i] * (int32_t)w;
        }

        acc += (int32_t)get_int4(b_packed, o);

        // Clamp to int4 range
        if (acc > 7) acc = 7;
        if (acc < -8) acc = -8;

        out[o] = (int8_t)acc;
    }
}
```

---

## 16-Bit Implementation

16-bit provides best accuracy with 2× memory cost.

### Memory Layout

| Layer | Shape | Size (bytes) |
|-------|-------|--------------|
| fc1_weight | [12, 36] | 864 |
| fc1_bias | [12] | 24 |
| fc2_weight | [10, 12] | 240 |
| fc2_bias | [10] | 20 |
| **Total** | | **1148 bytes** |

### 16-Bit Matrix Multiply

```c
void int16_matmul_fixed(
    const int16_t* x,
    const int16_t* W,
    const int16_t* b,
    int16_t* out,
    int in_size,
    int out_size
) {
    for (int o = 0; o < out_size; o++) {
        int64_t acc = 0;  // 64-bit accumulator for int16 products

        for (int i = 0; i < in_size; i++) {
            acc += (int64_t)x[i] * (int64_t)W[o * in_size + i];
        }

        acc += (int64_t)b[o];

        // Scale down and clamp
        acc >>= 8;  // Adjust based on your scale factors
        if (acc > 32767) acc = 32767;
        if (acc < -32768) acc = -32768;

        out[o] = (int16_t)acc;
    }
}
```

---

## Platform-Specific Optimizations

### ARM Cortex-M (STM32)

```c
// Use CMSIS-NN for hardware-accelerated int8 ops
#include "arm_nnfunctions.h"

// CMSIS-NN provides:
// - arm_fully_connected_s8() - optimized FC layer
// - arm_relu_q7() - optimized ReLU
// - Leverages DSP instructions on Cortex-M4/M7
```

### AVR (Arduino)

```c
// AVR is 8-bit native, int8 is efficient
// Store weights in flash (PROGMEM) to save RAM
#include <avr/pgmspace.h>

const int8_t fc1_weight[] PROGMEM = { ... };

// Read from flash:
int8_t w = pgm_read_byte(&fc1_weight[idx]);
```

### ESP32

```c
// ESP32 has hardware MAC unit
// Use DRAM_ATTR for fast weight access
const int8_t fc1_weight[] DRAM_ATTR = { ... };

// Consider using ESP-NN library for optimized ops
```

### FPGA/ASIC

For hardware implementation:
- **int8**: 8×8 multipliers → 16-bit product → 32-bit accumulator
- **int4**: 4×4 multipliers → 8-bit product → 16-bit accumulator
- **Membrane decay**: Single multiply + barrel shift (no division)
- **Threshold**: Simple comparator

---

## Expected Accuracy

Results from fixed-scale quantization-aware training on 6×6 MNIST:

| Model | Bits | Typical Accuracy | Memory (weights) |
|-------|------|------------------|------------------|
| StandardANN | 32 (float) | ~88% | 2296 bytes |
| StandardANN_Int16_Fixed | 16 | ~88% | 1148 bytes |
| StandardANN_Int8_Fixed | 8 | ~87% | 574 bytes |
| StandardANN_Int4_Fixed | 4 | ~80% | 287 bytes |
| GilgameshSNN | 32 (float) | ~85% | 2296 bytes |
| GilgameshSNN_Int8_Fixed | 8 | ~83% | 574 bytes |
| GilgameshSNN_Int4_Fixed | 4 | ~75% | 287 bytes |

---

## Verification

The training script verifies integer inference matches training:

```
Verifying integer inference for ann_int8_fixed (8-bit)...
    PyTorch (fixed quant) accuracy: 87.97%
    Integer-only accuracy:          87.95%
    Difference:                     +0.02%
```

A small difference (< 0.5%) is expected due to rounding. Large differences indicate a bug.

---

## Troubleshooting

### Model stuck at ~10% accuracy during training
- Check that scales are being applied correctly
- Ensure not double-applying scales in forward pass

### Integer overflow
- Use int32/int64 accumulators for matmul
- Clamp membrane potentials after each step

### Accuracy mismatch between training and integer inference
- Verify all quantization operations match between training and C code
- Check bias alignment and scaling

### Memory alignment issues
- Some platforms require aligned access (4-byte boundaries)
- Use `__attribute__((aligned(4)))` if needed

---

## Files Reference

```
comparison/results/
├── StandardANN_Int8_Fixed_weights.pt       # PyTorch checkpoint
├── StandardANN_Int8_Fixed_weights.json     # JSON weights (float)
├── StandardANN_Int8_Fixed_weights_8bit.h   # C header (int8)
├── StandardANN_Int8_Fixed_weights_8bit.bin # Binary (int8)
├── StandardANN_Int8_Fixed_weights_8bit_meta.json
├── GilgameshSNN_Int8_Fixed_weights_8bit.h  # SNN int8
└── ...
```
