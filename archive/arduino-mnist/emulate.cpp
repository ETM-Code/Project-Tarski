/*
 * Arduino MNIST Emulator
 *
 * Compiles the exact integer inference logic from model.cpp
 * with the trained weights from weights.hpp, targeting desktop.
 *
 * Protocol: reads 36 signed bytes (int8) from stdin per sample,
 *           writes 1 byte (prediction 0-9) to stdout.
 *           Loops until EOF.
 *
 * Build:  g++ -O2 -o emulate emulate.cpp
 * Usage:  python test_emulate.py
 */

#include <cstdint>
#include <cstdio>
#include <cmath>

// --- Desktop-compatible type aliases (matching Arduino datatypes.hpp) ---
using u8    = uint8_t;
using u16   = uint16_t;
using u32   = uint32_t;
using usize = uint16_t;

using i8    = int8_t;
using i16   = int16_t;
using i32   = int32_t;
using isize = int16_t;

using f32   = float;

using param_t = i8;

// --- View template (from datastructs.hpp) ---
template<typename T>
struct View
{
    const usize len;
    T* data;
};

// --- Network configuration and weights (from weights.hpp) ---
#define ANN_INPUT_SIZE  36
#define ANN_HIDDEN_SIZE 14
#define ANN_OUTPUT_SIZE 10

#define FC1_REQUANT_SHIFT     (-9)
#define FC1_BIAS_ALIGN_SHIFT  (-1)
#define FC2_REQUANT_SHIFT     (-6)
#define FC2_BIAS_ALIGN_SHIFT  (1)

static const i8 fc1_weight[504] = {
      33,   35,  -10,   39,   -9,    9,  -21,   25,   38,  -31,   37,    8,   32,    6,   21,   -6,
      33,    6,  -20,   11,  -20,   -5,  -17,   28,  -34,  -20,  -12,  -26,    4,  -42,   39,  -36,
      33,    7,  -14,   26,  -10,   19,   13,  -11,  -26,  -34,    8,   49,   36,   31,   15,  -24,
       1,  -83,  -66,   13,  -19,    1,    8,   -2,    4,    2,   23,  -19,   18,   36,  -12,   11,
      26,    8,   24,  -12,   -1,  -39,   12,   -1,   56,  -20,  -26,  -17,  -14,   32,   30,   47,
      -8,   -7,   38,   -2,   39,  -13,   13,  -16,  -29,   26,   23,  -13,   34,   10,  -35,  -10,
      54,    6,    4,  -38,   -9,   26,    9,  -16,   14,  -15,  -21,   -9,  -20,  -11,   54,   18,
      30,  -22,  -11,    3,   18,  -29,    5,   -5,   18,    5,   18,  -26,   -7,   -5,    3,  -13,
      32,  -15,   47,   30,  -20,   46,   -4,   -8,  -24,  -26,   29,    9,   38,  -14,   49,  -14,
     -15,   -2,   -3,   -6,    9,  -43,   13,   -9,   49,   67,   -2,  -41,   30,   17,  -53,    7,
      87,    1,    3,  -21,    9,   11,   -9,   22,    3,   -6,   -4,  -45,  -33,  -27,  -33,   -9,
      18,   49,   24,  -48,   13,   25,   11,  -12,   26,   26,   19,   -4,   -8,    8,   -4,   -9,
      31,   33,   -1,   -6,    4,    1,   13,   15,  -36,  -13,   33,   20,    1,  -22,  -23,  -35,
     -11,   18,   30,   23,  -34,  -22,   -3,   45,   18,    1,   30,   -5,  -16,   43,   -8,    4,
     -29,   38,   42,   27,  -18,  -30,  -24,   26,   30,   47,   34,   21,    7,  -23,  -25,  -28,
      29,  -11,   -1,  -12,    4,  -18,  -16,   26,   -1,   39,   -2,  -20,   -2,  -17,   -4,    5,
      36,  -22,   21,  -10,   11,  -21,    4,  -14,   15,    7,  -39,  -21,  -12,   -5,   26,  -56,
      61,    5,  -34,   26,  -28,   30,   37,   16,   35,   -7,    2,  -26,  -39,   -8,  -38,   40,
      29,  -12,  -12,  -15,  -15,  -41,    6,  -37,   -2,   19,    1,  -31,  -11,  -21,  -21,  -83,
      23,   33,  -25,   49,   52,   -9,   54,    6,  -39,   19,   51,   37,   21,   18,  -13,   -1,
     -36,  -21,   15,   13,  -12,  -25,  -29,  -42,  -23,    5,   38,   -8,   -1,    5,   36,  -11,
       0,   31,   28,  -38,   16,    8,   -6,   -1,  -74,  -45,  -19,  -22,  -14,   14,   28,   15,
       7,   28,  -29,   -1,   28,   24,   -7,   55,  -36,  -40,  -23,    3,  -22,  -16,  -10,  -28,
      18,   22,   24,   28,  -37,    9,    0,  -52,   16,   39,   23,  -26,   38,  -22,    8,   19,
     -20,   64,   17,   19,   49,  -13,  -29,  -49,   27,   17,   12,  -40,  -27,    2,  -11,  -14,
      10,   38,    5,  -24,    0,   -6,   27,  -26,  -25,   53,    8,  -44,   49,  -19,   25,   19,
      28,   64,  -10,  -23,   -7,  -53,  -20,  -39,  -15,    4,   -1,  -27,   13,   15,   19,    0,
      -7,   33,  -25,   36,   -7,   10,    1,   14,   -6,   35,    7,   43,   51,  -10,  -23,   25,
      40,  -23,    6,  -32,  -25,   15,   18,   18,   -3,   46,   12,  -19,    6,   20,   24,  -21,
      25,  -12,  -26,  -14,   41,   36,   38,   25,   32,   -6,  -23,   21,  -22,  -29,  -14,    9,
      22,  -17,  -25,    6,  -25,  -28,   22,   -7,   39,   42,   13,   15,   10,    1,   -3,    1,
      16,   40,  -11,  -18,  -10,  -21,    7,   32,
};

static const int8_t fc1_bias[14] = {
      67,   73,  -92,  -80,   -2,  -22,  -16,  -91,   38,  -96,  -15,  -87,   55,  -81,
};

static const int8_t fc2_weight[140] = {
      52,   -8,  -26,   30,  -21,   40,   36,  -27,   31,    1,    2,  -17,    9,  -33,   -9,   -9,
      39,  -16,  -24,    1,  -46,   71,  -59,  -18,  -19,  -37,   55,    7,   40,   53,   49,  -24,
     -32,   30,  -19,    5,   12,   -5,   -3,   18,   36,   68,   47,   60,   23,  -38,  -32,  -44,
     -49,  -38,  -25,  -32,   33,    9,  -29,   26,   57,    7,  -16,   33,  -54,  -29,   32,  -12,
       4,  -39,  -20,   66,  -48,   -1,    4,   21,   18,   51,  -60,   -1,  -37,  -22,  -20,  -14,
      47,   18,  -33,   36,  -65,  -70,   -8,  -41,  -42,  -14,   42,   26,   69,  -24,  -50,  -15,
      15,    2,    9,   39,   -8,   72,   -1,   -7,  -47,  -11,    1,   17,  -41,   28,  -58,   20,
     -45,    0,   28,   53,  -21,  -29,  -45,   -6,  -26,  -55,   47,   10,   55,   43,  -67,    2,
      11,  -28,  -14,  -34,   44,  -43,  -10,  -42,    2,   39,  -19,  -39,
};

static const int8_t fc2_bias[10] = {
     -32,   62,  -78,   60,  -37,   25,   -9,   -9,  -52,   46,
};

// --- Neuron / layer structures (from model.cpp) ---

struct neuron_t
{
    const param_t* weights;
    i32 bias;
};

struct dense_layer_t
{
    usize n_neurons;
    neuron_t* neurons;
};

// --- Static state ---

static neuron_t hidden_neurons[ANN_HIDDEN_SIZE];
static neuron_t output_neurons[ANN_OUTPUT_SIZE];

static const dense_layer_t hidden_layer = { ANN_HIDDEN_SIZE, hidden_neurons };
static const dense_layer_t output_layer = { ANN_OUTPUT_SIZE, output_neurons };

static i32 hidden_layer_output_data[ANN_HIDDEN_SIZE];
static i8  output_layer_input_data[ANN_HIDDEN_SIZE];
static i32 output_data[ANN_OUTPUT_SIZE];

static View<i32> hidden_layer_output = { ANN_HIDDEN_SIZE, hidden_layer_output_data };
static View<i8>  output_layer_input  = { ANN_HIDDEN_SIZE, output_layer_input_data };
static View<i32> output              = { ANN_OUTPUT_SIZE,  output_data };

// --- Model functions (exact copies from model.cpp) ---

namespace Model
{
    static void ClampOutput(View<i8>& out, const View<i32>& input)
    {
        for(usize i = 0; i < input.len; i++)
        {
            if (input.data[i] < -128) { out.data[i] = -128; continue; }
            if (input.data[i] >  127) { out.data[i] =  127; continue; }
            out.data[i] = input.data[i];
        }
    }

    static i32 ApplyShift(i32 value, i16 shift)
    {
        if (shift >= 0) return value << shift;
        else
        {
            i16 right_shift = -shift;
            i32 rounding = 1 << (right_shift - 1);
            return (value + rounding) >> right_shift;
        }
    }

    static usize ArgMax(const View<i32>& data)
    {
        usize pos = 0;
        i32 max = data.data[pos];

        for(usize i = 1; i < data.len; i++)
        {
            if(data.data[i] > max)
            {
                max = data.data[i];
                pos = i;
            }
        }

        return pos;
    }

    static void ReLUActivation(View<i32>& data)
    {
        for(usize i = 0; i < data.len; i++)
        {
            data.data[i] = data.data[i] < 0 ? 0 : data.data[i];
        }
    }

    static i32 ForwardPass(const neuron_t& neuron, const View<param_t>& input, i16 bias_shift, i16 requant_shift)
    {
        i32 accumulator = 0;

        for(usize i = 0; i < input.len; i++)
            accumulator += (i32)input.data[i] * neuron.weights[i];

        i32 bias = Model::ApplyShift(neuron.bias, bias_shift);
        accumulator += bias;

        return Model::ApplyShift(accumulator, requant_shift);
    }

    static void ProcessDenseLayer(const dense_layer_t& dense_layer, const View<param_t>& input, View<i32>& out, i16 bias_shift, i16 requant_shift)
    {
        for(usize i = 0; i < dense_layer.n_neurons; i++)
            out.data[i] = ForwardPass(dense_layer.neurons[i], input, bias_shift, requant_shift);
    }

    void Init(void)
    {
        for(usize i = 0; i < ANN_HIDDEN_SIZE; i++)
        {
            hidden_neurons[i].weights = fc1_weight + (i * ANN_INPUT_SIZE);
            hidden_neurons[i].bias = fc1_bias[i];
        }

        for(usize i = 0; i < ANN_OUTPUT_SIZE; i++)
        {
            output_neurons[i].weights = fc2_weight + (i * ANN_HIDDEN_SIZE);
            output_neurons[i].bias = fc2_bias[i];
        }
    }

    u8 PredictClass(const View<param_t>& input)
    {
        // Layer 1
        Model::ProcessDenseLayer(hidden_layer, input, hidden_layer_output, FC1_BIAS_ALIGN_SHIFT, FC1_REQUANT_SHIFT);
        Model::ReLUActivation(hidden_layer_output);

        // Clamp for Layer 2
        Model::ClampOutput(output_layer_input, hidden_layer_output);

        // Layer 2
        Model::ProcessDenseLayer(output_layer, output_layer_input, output, FC2_BIAS_ALIGN_SHIFT, FC2_REQUANT_SHIFT);

        return Model::ArgMax(output);
    }
}

// --- Main: read samples from stdin, write predictions to stdout ---

int main()
{
    Model::Init();

    param_t sample[ANN_INPUT_SIZE];
    View<param_t> input = { ANN_INPUT_SIZE, sample };

    while (true)
    {
        // Read exactly 36 signed bytes
        size_t n = fread(sample, 1, ANN_INPUT_SIZE, stdin);
        if (n == 0) break;           // EOF
        if (n != ANN_INPUT_SIZE)
        {
            fprintf(stderr, "Error: incomplete sample (%zu bytes)\n", n);
            return 1;
        }

        u8 prediction = Model::PredictClass(input);

        // Write single byte prediction
        fwrite(&prediction, 1, 1, stdout);
        fflush(stdout);
    }

    return 0;
}
