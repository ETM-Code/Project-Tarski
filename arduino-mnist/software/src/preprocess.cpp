#include <algorithm>
#include "preprocess.hpp"
#include "fc1_weights.h"

namespace Preprocess
{
    static i32 apply_shift(i32 value, i16 shift)
    {
        if(shift >= 0) return value << shift;
        const i16 right_shift = -shift;
        const i32 rounding = 1 << (right_shift - 1);
        return (value + rounding) >> right_shift;
    }
}

bool Preprocess::ToHiddenActivations(const std::vector<u8>& sample_data, std::array<i8, HIDDEN_SIZE>& hidden_out)
{
    if(sample_data.size() < INPUT_SIZE) return false;

    for(std::size_t h = 0; h < HIDDEN_SIZE; h++)
    {
        i32 acc = 0;
        const std::size_t w_offset = h * INPUT_SIZE;

        for(std::size_t i = 0; i < INPUT_SIZE; i++)
        {
            const i8 x = static_cast<i8>(sample_data[i]);
            const i8 w = StandardANN_Int8_Shift_fc1_fc1_weight[w_offset + i];
            acc += static_cast<i32>(x) * static_cast<i32>(w);
        }

        const i32 bias = apply_shift(StandardANN_Int8_Shift_fc1_fc1_bias[h], STANDARDANN_INT8_SHIFT_FC1_FC1_BIAS_ALIGN_SHIFT);
        acc += bias;
        acc = apply_shift(acc, STANDARDANN_INT8_SHIFT_FC1_FC1_REQUANT_SHIFT);
        if(acc < 0) acc = 0; // ReLU
        acc = std::clamp<i32>(acc, -128, 127);

        hidden_out[h] = static_cast<i8>(acc);
    }

    return true;
}
