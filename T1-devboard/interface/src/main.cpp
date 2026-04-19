#include <Arduino.h>
#include "device.hpp"
#include "signals.hpp"

void setup()
{
    // Initialise the device
    Device::Init(Device::Timeout::TO1S);

    // Begin serial communication
    Serial.begin(115200);
}

void loop()
{
    // Wait until data is available on the serial bus.
    while(!Serial.available()) Device::Idle();

    // Read the command identifier sent by the host.
    char command_type = Serial.read();

    // Route the command to the corresponding device handler.
    switch(command_type)
    {
        case PORT_SIG:                          // Get Device Signature
            Device::SendSignature();
            break;

        case PORT_READ_OUT:                     // Read latched accelerator output
            Device::ReadOutput();
            break;

        case PORT_LOAD_SYN:                     // Load synapse weights
            Device::LoadWeights();
            break;

        case PORT_LOAD_DAC:                     // Set one or more DAC values
            Device::LoadDACs();
            break;

        case PORT_READ_MEAS:                    // Read L1/L2 voltage measurement
            Device::ReadMeasurement();
            break;

        case PORT_SET_FLAG:                     // Enable a feature flag
            Device::SetFlag();
            break;

        case PORT_UNSET_FLAG:                   // Disable a feature flag
            Device::UnSetFlag();
            break;

        case PORT_TGL_FLAG:                     // Toggle a feature flag
            Device::ToggleFlag();
            break;

        case PORT_RUN_INF:                      // Run inference with rapid spike sampling
            Device::RunInference();
            break;

        case PORT_CALIB_L1:                     // L1 calibration: time to spike
            Device::CalibL1();
            break;

        case PORT_CALIB_PULSE:                  // Pulse duration: fast ADC burst
            Device::CalibPulse();
            break;

        case PORT_PROG_DAC:                     // Program MCP4728 I2C address
            Device::ProgramDACAddress();
            break;

        default:                                // Unknown command
            Device::SendUnknownCommand();
            break;
    }
}
