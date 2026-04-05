#pragma once

#include "datastructs.hpp"

using param_t = i8;

#define SAMPLE_DATA_SIZE        36

namespace Model
{
    /**
     * Initialise all required internal data for the MNIST inference model.
     */
    void Init(void);

    /**
     * Predict what number the provided data corresponts to.
     * Results stored internally to the Model.
     */
    u8 PredictClass(const View<param_t>& input);

    /**
     * Returns the confidence of the provided classification.
     * @warning `classification` must be a valid return value from `PredictClass`.
     * @param classification The classification for which you would like to know the confidence of.
     * @return The percentage confidence of the classification.
     */
    f32 GetConfidence(u8 classification);

    //  --- DEBUG METHODS ---

    /**
     * Returns a read-only reference to the models internal Layer 1 output
     */
    const View<i32>& getL1Output(void);

    /**
     * Returns a read-only reference to the models internal Layer 1 quantised output
     */
    const View<i8>& getL1QuantOutput(void);

    /**
     * Dumps the contents of the second layers output to the serial interface in CSV format
     */
    const View<i32>& getL2Output(void);
}