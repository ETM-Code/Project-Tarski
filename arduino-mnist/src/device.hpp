#pragma once

/**
 * This file defines the functionality of the arduino nano.
**/

#include "datatypes.hpp"

namespace Device
{
    /**
     * Initialise the device with the provided parameters.
     * @param timeout_ms The amount of miliseconds to wait for serial communication.
     */
    void Init(time_t timeout_ms);

    /**
     * Send the devices signature over the serial bus.
     */
    void SendSignature(void);

    /**
     * Sends a NAK and message informing the sender of an unknown command over the serial bus.
     */
    void SendUnknownCommand(void);

    /**
     * Load host-preprocessed hidden activations (L1 output) over serial.
     */
    void LoadHiddenData(void);

    /**
     * Run's the inference model on the sample data and returns results onto the serial bus.
     */
    void RunInference(void);

    /**
     * Await for any data on the serial bus.
     * The time it awaits is based on the value passed into `Device::Init()`
     * @return `true` if data was recieved, `false` otherwise.
     */
    bool AwaitData(void);

    /**
     * Await for an ACK or NAK signal on the serial bus.
     * The time it awaits is based on the value passed into `Device::Init()`
     * @return `true` if an ACK or NAK was recieved, `false` otherwise.
     */
    bool AwaitResponse(void);
};  
