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
    u8 PredictClassFromHidden(const View<i8>& hidden_input);

    /**
     * Returns the confidence of the provided classification.
     * @warning `classification` must be a valid return value from `PredictClass`.
     * @param classification The classification for which you would like to know the confidence of.
     * @return The percentage confidence of the classification.
     */
    f32 GetConfidence(u8 classification);

    //  --- DEBUG METHODS ---

    /**
     * Returns a read-only reference to the models Layer 2 (output layer) raw values
     */
    const View<i32>& getL2Output(void);
}
