# Training and Weight Generation

## Overview

The MNIST model is split into two parts:
- **FC1 (Input → Hidden)**: 36 inputs → 14 hidden activations, computed on the laptop
- **FC2 (Hidden → Output)**: 14 hidden inputs → 10 output classes, computed on the Arduino

## Training Pipeline

The training script `training/comparison/snntorch_comparison.py` trains various ANN and SNN models and generates quantized weights for embedded deployment.

### Running Training

```bash
cd arduino-mnist
python3 training/comparison/snntorch_comparison.py --models baseline ann --bits 8
```

This will:
1. Train the models on MNIST (6×6 downsampled)
2. Export weights as quantized integer values
3. **Generate separate C headers for FC1 and FC2** to: `training/comparison/resultsnew/`
   - `StandardANN_Int8_Shift_fc1_weights_8bit.h` — FC1 weights (36×14), for laptop preprocessing
   - `StandardANN_Int8_Shift_fc2_weights_8bit.h` — FC2 weights (14×10), for Arduino

### Weight Files

Each generated header contains:
- **Quantized weight arrays** (int8, int16, or int4 depending on `--bits`)
- **Bias vectors**
- **Shift amounts** (for integer-only inference)
- **Configuration macros** (input size, hidden size, output size, etc.)

#### FC1 Header Example
```c
#define STANDARDANN_INT8_SHIFT_INPUT_SIZE 36
#define STANDARDANN_INT8_SHIFT_HIDDEN_SIZE 14

static const int8_t StandardANN_Int8_Shift_fc1_weight[504] = { ... };
static const int8_t StandardANN_Int8_Shift_fc1_bias[14] = { ... };

#define STANDARDANN_INT8_SHIFT_FC1_REQUANT_SHIFT (-9)
#define STANDARDANN_INT8_SHIFT_FC1_BIAS_ALIGN_SHIFT (-1)
```

#### FC2 Header Example
```c
#define STANDARDANN_INT8_SHIFT_HIDDEN_SIZE 14
#define STANDARDANN_INT8_SHIFT_OUTPUT_SIZE 10

static const int8_t StandardANN_Int8_Shift_fc2_weight[140] = { ... };
static const int8_t StandardANN_Int8_Shift_fc2_bias[10] = { ... };

#define STANDARDANN_INT8_SHIFT_FC2_REQUANT_SHIFT (-6)
#define STANDARDANN_INT8_SHIFT_FC2_BIAS_ALIGN_SHIFT (1)
```

### Integration with Build System

To use the latest trained weights:

1. **For laptop software** (`arduino-mnist/software/`):
   ```bash
   cp training/comparison/resultsnew/StandardANN_Int8_Shift_fc1_weights_8bit.h software/include/fc1_weights.h
   ```

2. **For Arduino firmware** (`arduino-mnist/src/`):
   ```bash
   cp training/comparison/resultsnew/StandardANN_Int8_Shift_fc2_weights_8bit.h src/weights_fc2.hpp
   ```

Alternatively, update the build system to reference the training output directory directly.

## Notes

- The `--bits` parameter controls quantization: 4, 8, or 16 bits
- Shift-based quantization (recommended for integer-only hardware) generates shift parameters instead of floating-point scales
- Both FC1 and FC2 use the same quantization bit-width for consistency
