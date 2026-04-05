#pragma once

#include "arduino.hpp"
#include "cli.hpp"

/**
 * Runtime operation dispatcher helpers for each CLI mode.
 * These functions execute validated mode workflows against an initialized Arduino device.
 */

namespace Runner
{
    /**
     * Execute load mode operation.
     * Loads sample data and optionally repeats load requests for the configured duration.
     * @param device Initialized Arduino communication interface.
     * @param args Parsed and validated CLI arguments.
     */
    bool DataLoad(Arduino& device, const CLI::Arguments& args);

    /**
     * Execute run mode operation.
     * Runs inference once or repeatedly for the configured duration.
     * @param device Initialized Arduino communication interface.
     * @param args Parsed and validated CLI arguments.
     */
    bool Inference(Arduino& device, const CLI::Arguments& args);

    /**
     * Execute predict mode operation.
     * Runs single-file or batch prediction depending on `args.batch_mode`.
     * @param device Initialized Arduino communication interface.
     * @param args Parsed and validated CLI arguments.
     */
    bool Prediction(Arduino& device, const CLI::Arguments& args);
}
