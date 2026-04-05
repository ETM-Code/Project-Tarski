#include <Arduino.h>
#include "device.hpp"
#include "signals.hpp"
#include "version.hpp"
#include "datastructs.hpp"
#include "model.hpp"

namespace Device
{
    static time_t _timeout_ms;
    static u8 _response;
    static param_t _sample_data[SAMPLE_DATA_SIZE];
    
    static View<param_t> _input = { SAMPLE_DATA_SIZE, _sample_data };
}

void Device::Init(time_t timeout_ms)
{
    // Set up the timeout value to the provided one
    _timeout_ms = timeout_ms;
}

void Device::SendSignature(void)
{
    Serial.write(PORT_ACK);	            // Acknowledge request

    // Version numbers shifter up 0x30 to avoid collisions with control characters
    Serial.write(FIRM_VER_MJR + 0x30);  // Firmware major version
    Serial.write(FIRM_VER_MNR + 0x30);  // Firmware minor version
    Serial.write(FIRM_VER_PCH + 0x30);  // Firmware patch version

    Serial.write(PORT_TRN_END);         // EOT
}

void Device::SendUnknownCommand(void)
{
    Serial.write(PORT_NAK);             // Acknowledge data recieved but command unknown

    // Serial.write(PORT_MSG);             // Signal that the device wants to send a message
    // Send message
    // Serial.write(PORT_MSG_END);         // Signal end of message

    Serial.write(PORT_TRN_END);         // Signal end of transmission
}

void Device::LoadSampleData(void)
{
    // Acknowledge command recieved
    Serial.write(PORT_ACK);

    // Load sample data from the serial bus.
    // Image size will always be 72 bytes.
    for(u8 i = 0; i < SAMPLE_DATA_SIZE; i++)
    {
        if(!Device::AwaitData()) { break; }     // TODO - Handle failed data transfer
        _sample_data[i] = Serial.read();        // Load recieved data into sample data buffer
    }

    // Acknowledge all data received
    Serial.write(PORT_ACK);

    // Echo data to ensure correct read
    for(u8 i = 0; i < SAMPLE_DATA_SIZE; i++)
    {
        Serial.write(_sample_data[i]);
    }

    Device::AwaitResponse();                    // Handle timeout event

    // The computer will transmit an ACK if the data matches, and a NAK if the data differs.
    if(_response == PORT_NAK)
    {
        // Handle this error
    }

    // Successful load of sample data
}

void Device::RunInference(void)
{
    // Acknowledge command recieved
    Serial.write(PORT_ACK);

    // Run the inference model and time it
    time_t start = millis();
    u8 prediction = Model::PredictClass(_input);
    time_t prediction_time = millis() - start;

    // Return the results over the serial bus.
    // Results are printed in a CSV format.
    // prediction_time, prediction

    // Send the raw prediction data over the serial port
    Serial.write(prediction);

    // Inform computer of the beginning of a message
    Serial.write(PORT_MSG);

    // Print data to the serial bus.
    Serial.print(prediction_time);
    Serial.write(',');
    Serial.print(prediction);

    // Inform computer of the end of a message and transmission
    Serial.write(PORT_MSG_END);
    Serial.write(PORT_TRN_END);
}

bool Device::AwaitData(void)
{
    // Setup timer and end time
    time_t current_time = millis();
    time_t timeout_time = current_time + _timeout_ms;

    // Loop until communication achieved or timeout
    while(current_time < timeout_time)
    {
        if(Serial.available()) return true;
        current_time = millis();
    }

    // If we timed-out return false
    return false;
}

bool Device::AwaitResponse(void)
{
    // Setup timer and end time
    time_t current_time = millis();
    time_t timeout_time = current_time + _timeout_ms;

    // Set response to known value
    _response = 0x00;

    // Loop until communication achieved or timeout
    while(current_time < timeout_time)
    {
        if(Serial.available()){ _response = Serial.read(); break; }
        current_time = millis();
    }

    // Return false if we had no response
    if(!_response) return false;

    // Return false if the response is not an ACK or NAK
    if(_response != PORT_ACK && _response != PORT_NAK) return false;

    // Otherwise return true
    return true;
}