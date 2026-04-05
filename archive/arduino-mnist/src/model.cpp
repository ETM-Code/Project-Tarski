#include "model.hpp"
#include "math.h"
#include "weights.hpp"

//  --- Data Structures ---

/*
 * Structure that models a neuron.
 * Specifies the weights of the neuron as a vector (param_t* weights) and the bias (param_t bias).
 */
struct neuron_t
{
    const param_t* weights;
    i32 bias;
};

/*
 * Structure that models a dense layer.
 * Specifies the number of neurons (usize n_neurons) and a vector of neurons (neuron_t * neurons). 
 */
struct dense_layer_t
{
    usize n_neurons;
    neuron_t* neurons;
};

//  --- Static Variables ---

// Neuron arrays
static neuron_t hidden_neurons[ANN_HIDDEN_SIZE];
static neuron_t output_neurons[ANN_OUTPUT_SIZE];

// Neuron layers
static const dense_layer_t hidden_layer = { ANN_HIDDEN_SIZE, hidden_neurons };
static const dense_layer_t output_layer = { ANN_OUTPUT_SIZE, output_neurons };

// Output data storage
static i32 hidden_layer_output_data[ANN_HIDDEN_SIZE];
static i8 output_layer_input_data[ANN_HIDDEN_SIZE];
static i32 output_data[ANN_OUTPUT_SIZE];
static f32 confidence_data[ANN_OUTPUT_SIZE];

static View<i32> hidden_layer_output = { ANN_HIDDEN_SIZE, hidden_layer_output_data };
static View<i8> output_layer_input = { ANN_HIDDEN_SIZE, output_layer_input_data };
static View<i32> output = { ANN_OUTPUT_SIZE, output_data };
static View<f32> confidence = { ANN_OUTPUT_SIZE, confidence_data };

//  --- Private Functions ---

namespace Model
{
    /**
     * Bro got clamped for speeding.
     */
    static void ClampOutput(View<i8>& output, const View<i32>& input)
    {
        for(usize i = 0; i < input.len; i++)
        {
            if (input.data[i] < -128) { output.data[i] = -128; continue; }
            if (input.data[i] >  127) { output.data[i] =  127; continue; }
            output.data[i] = input.data[i];
        }
    }

    /**
     * Shifting data around at the speed of sound.
     */
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

    /**
     * Returns the index of the largest value in the provided data view.
     */
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

    /**
     * 
     */
    static void ReLUActivation(View<i32>& data)
    {
        for(usize i = 0; i < data.len; i++)
        {
            data.data[i] = data.data[i] < 0 ? 0 : data.data[i];
        }
    }

    /**
     * 
     */
    static void SoftMaxActivation(const View<i32>& data)
    {
        // Get maximum value
        f32 max = -INFINITY;
        for(usize i = 0; i < data.len; i++)
        {
            if(data.data[i] > max)
                max = data.data[i];
        }

        // Sum all outputs
        f32 sum = 0;
        for(usize i = 0; i < data.len; i++)
        {
            sum += exp(data.data[i] - max);
        }

        // Calculate offset
        f32 offset = max + log(sum);
        for(usize i = 0; i < data.len; i++)
        {
            confidence.data[i] = exp(data.data[i] - offset);
        }
    }

    /**
     * TODO
     */
    static i32 ForwardPass(const neuron_t& neuron, const View<param_t>& input, i16 bias_shift, i16 requant_shift)
    {
        i32 accumulator = 0;

        for(usize i = 0; i < input.len; i++)
            accumulator += (i32)input.data[i] * neuron.weights[i];

        i32 bias = Model::ApplyShift(neuron.bias, bias_shift);
        accumulator += bias;

        return Model::ApplyShift(accumulator, requant_shift);
    }

    /**
     * TODO
     */
    static void ProcessDenseLayer(const dense_layer_t& dense_layer, const View<param_t>& input, View<i32>& output, i16 bias_shift, i16 requant_shift)
    {
        for(usize i = 0; i < dense_layer.n_neurons; i++)
            output.data[i] = ForwardPass(dense_layer.neurons[i], input, bias_shift, requant_shift);
    }

    static void Requantise(const View<i32>& input, View<i8>& output, f32 scale)
    {
        for(usize i = 0; i < input.len; i++)
        {
            float scaled = input.data[i] * scale;
            i32 rounded = lrintf(scaled);

            if (rounded >  127) rounded =  127;
            if (rounded < -128) rounded = -128;

            output.data[i] = (i8)rounded;
        }
    }
}

//  --- Public Functions ---

void Model::Init(void)
{
    // Load weight pointers to required places
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

u8 Model::PredictClass(const View<param_t>& input)
{
    //  --- Layer 1 ---
    Model::ProcessDenseLayer(hidden_layer, input, hidden_layer_output, FC1_BIAS_ALIGN_SHIFT, FC1_REQUANT_SHIFT);
    Model::ReLUActivation(hidden_layer_output);
    
    //  --- Clamp data for Layer 2 ---
    Model::ClampOutput(output_layer_input, hidden_layer_output);
    
    //  --- Layer 2 ---
    Model::ProcessDenseLayer(output_layer, output_layer_input, output, FC2_BIAS_ALIGN_SHIFT, FC2_REQUANT_SHIFT);

    //  --- We no longer perform a SoftMax Activation ---
    // Model::SoftMaxActivation(output);

    return Model::ArgMax(output);
}

f32 Model::GetConfidence(u8 classification)
{
    return 0;
}

const View<i32>& Model::getL1Output(void)
{
    return hidden_layer_output;
}

const View<i8>& Model::getL1QuantOutput(void)
{
    return output_layer_input;
}

const View<i32>& Model::getL2Output(void)
{
    return output;
}