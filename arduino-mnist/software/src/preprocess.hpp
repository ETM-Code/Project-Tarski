#pragma once

#include <array>
#include <vector>
#include "datatypes.hpp"

namespace Preprocess
{
    constexpr std::size_t INPUT_SIZE = 36;
    constexpr std::size_t HIDDEN_SIZE = 14;

    bool ToHiddenActivations(const std::vector<u8>& sample_data, std::array<i8, HIDDEN_SIZE>& hidden_out);
}
