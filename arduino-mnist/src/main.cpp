#include <Arduino.h>
#include "device.hpp"
#include "signals.hpp"
// #include "mnist_model.hpp"
#include "model.hpp"

// Probably want to use FIXED point numbers

void setup()
{
    // Initialise all used pins for I/O
    // UNUSED

    // Begin serial communication
    Serial.begin(9600);
    // Serial.begin(115200);

    // Initialise internal device data
    Device::Init(1000);

    // Initialise the model
    Model::Init();
}

void loop()
{
    // Await serial communication from the user
	while(!Serial.available()) continue;
	
    // Get the command character from the serial interface
    char command_type = Serial.read();

    switch(command_type)
    {
        case PORT_SIG:                          // Get Device Signature
            Device::SendSignature();
            break;

        case PORT_LOAD_HIDDEN:                  // Load host-preprocessed hidden activations
            Device::LoadHiddenData();
            break;

        case PORT_INFER:                        // Infer output from sample data and return results
            Device::RunInference();
            break;

        default:                                // Unknown Command
            Device::SendUnknownCommand();
            break;
    }
}
