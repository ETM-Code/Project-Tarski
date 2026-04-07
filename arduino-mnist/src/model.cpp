#include "model.hpp"
#include "weights_fc2.hpp"

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

static neuron_t output_neurons[ANN_OUTPUT_SIZE];
static const dense_layer_t output_layer = { ANN_OUTPUT_SIZE, output_neurons };
static i32 output_data[ANN_OUTPUT_SIZE];
static View<i32> output = { ANN_OUTPUT_SIZE, output_data };

namespace Model
{
    static i32 ApplyShift(i32 value, i16 shift)
    {
        if(shift >= 0) return value << shift;

        const i16 right_shift = -shift;
        const i32 rounding = 1 << (right_shift - 1);
        return (value + rounding) >> right_shift;
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

    static i32 ForwardPass(const neuron_t& neuron, const View<param_t>& input, i16 bias_shift, i16 requant_shift)
    {
        i32 accumulator = 0;

        for(usize i = 0; i < input.len; i++)
            accumulator += static_cast<i32>(input.data[i]) * neuron.weights[i];

        const i32 bias = Model::ApplyShift(neuron.bias, bias_shift);
        accumulator += bias;

        return Model::ApplyShift(accumulator, requant_shift);
    }

    static void ProcessDenseLayer(const dense_layer_t& dense_layer, const View<param_t>& input, View<i32>& layer_output, i16 bias_shift, i16 requant_shift)
    {
        for(usize i = 0; i < dense_layer.n_neurons; i++)
            layer_output.data[i] = ForwardPass(dense_layer.neurons[i], input, bias_shift, requant_shift);
    }
}

void Model::Init(void)
{
    for(usize i = 0; i < ANN_OUTPUT_SIZE; i++)
    {
        output_neurons[i].weights = fc2_weight + (i * ANN_HIDDEN_SIZE);
        output_neurons[i].bias = fc2_bias[i];
    }
}

u8 Model::PredictClassFromHidden(const View<i8>& hidden_input)
{
    Model::ProcessDenseLayer(output_layer, hidden_input, output, FC2_BIAS_ALIGN_SHIFT, FC2_REQUANT_SHIFT);
    return Model::ArgMax(output);
}

f32 Model::GetConfidence(u8 classification)
{
    (void)classification;
    return 0;
}

const View<i32>& Model::getL2Output(void)
{
    return output;
}
