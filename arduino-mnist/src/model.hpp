#pragma once

#include "datastructs.hpp"

using param_t = i8;

namespace Model
{
    constexpr usize HIDDEN_FEATURE_SIZE = 14;

    /**
     * Initialise all required internal data for the MNIST inference model.
     */
    void Init(void);

    /**
     * Predict from host-preprocessed hidden activations.
     * Input must be ReLU'd/clamped int8 activations with length HIDDEN_FEATURE_SIZE.
     */
    u8 PredictClass(const View<i8>& input);

    //  --- DEBUG METHODS ---

    /**
     * Returns a read-only reference to the models Layer 2 (output layer) raw values
     */
    const View<i32>& getL2Output(void);
}
