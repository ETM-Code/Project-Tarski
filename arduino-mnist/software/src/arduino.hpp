#pragma once

#include <span>
#include "serial.hpp"

/**
 * Thin Arduino-specific protocol wrapper over `Serial`.
 * Provides version query, sample load, and classification commands.
 */

// Non-standard Serial Signals
#define PORT_SIG        0x05            // Used to request for device signature (version number) 
#define PORT_LOAD       'L'             // Used to load a data sample into the arduino's memory
#define PORT_LOAD_HIDDEN 'H'            // Used to load host-preprocessed hidden activations
#define PORT_INFER      'I'             // Signals to the arduino to run the inference model

class Arduino : public Serial
{
    public:
    
    // This is the size of the image data to load onto the arduino in bytes
    constexpr static std::size_t DATA_SIZE = 36;
    constexpr static std::size_t HIDDEN_SIZE = 14;

    Arduino(const char* path, Baudrate baudrate, u32 timeout_ms);

    /**
     * Prints the arduino firmware version to the console.
     * @return `true` on successful communication and version display, `false` otherwise.
     */
    bool printVersion(void);

    /**
     * Sends a sample payload to the Arduino and validates the echoed response.
     * @param data Sample bytes to transfer. Must contain at least `DATA_SIZE` bytes.
     * @return `true` if transfer and echo verification succeeded, `false` otherwise.
     */
    bool loadData(std::span<u8> data);

    /**
     * Sends host-preprocessed hidden activations to the Arduino.
     * @param hidden L1 post-ReLU int8 activations, length HIDDEN_SIZE.
     * @return `true` if transfer succeeded, `false` otherwise.
     */
    bool loadHiddenData(std::span<i8> hidden);

    /**
     * Runs classification on the currently loaded sample and writes device output.
     * @param output_sink Stream used to write classification/message output.
     * @param correct_classification Expected class label, or sentinel value to skip comparison.
     * @return `true` if inference transaction completed successfully, `false` otherwise.
     */
    bool runClassification(FILE* output_sink, u8 correct_classification);

    private:

};
