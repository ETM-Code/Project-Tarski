#include <Arduino.h>
#include "device.hpp"
#include "signals.hpp"
#include "version.hpp"
#include "datastructs.hpp"
#include "model.hpp"

namespace Device
{
    static u8 _response;
    static time_t _timeout_ms;
    static i8 _hidden_data[Model::HIDDEN_FEATURE_SIZE];
    static View<i8> _hidden_input = { Model::HIDDEN_FEATURE_SIZE, _hidden_data };
}

void Device::Init(time_t timeout_ms)
{
    _timeout_ms = timeout_ms;
}

void Device::SendSignature(void)
{
    Serial.write(PORT_ACK);
    Serial.write(FIRM_VER_MJR + 0x30);
    Serial.write(FIRM_VER_MNR + 0x30);
    Serial.write(FIRM_VER_PCH + 0x30);
    Serial.write(PORT_TRN_END);
}

void Device::SendUnknownCommand(void)
{
    Serial.write(PORT_NAK);
    Serial.write(PORT_TRN_END);
}

void Device::LoadData(void)
{
    // Acknowledge command recieved
    Serial.write(PORT_ACK);

    // Load sample data from the serial bus.
    for(u8 i = 0; i < Model::HIDDEN_FEATURE_SIZE; i++)
    {
        if(!Device::AwaitData())
        {
            Serial.write(PORT_NAK);
            break;
        }
        _hidden_data[i] = static_cast<i8>(Serial.read());
    }

    // Acknowledge all data received
    Serial.write(PORT_ACK);

    // Echo data to ensure correct read
    for(u8 i = 0; i < Model::HIDDEN_FEATURE_SIZE; i++)
    {
        Serial.write(_hidden_data[i]);
    }

    if(!Device::AwaitResponse())
    {
        Serial.write(PORT_NAK);
        return;
    }

    // Successful load of sample data
}

void Device::RunInference(void)
{
    // Acknowledge command recieved
    Serial.write(PORT_ACK);

    // Run the inference model and time it
    time_t start = millis();
    const u8 prediction = Model::PredictClass(_hidden_input);
    time_t prediction_time = millis() - start;

    // Send the raw prediction data over the serial port
    Serial.write(prediction);

    // Inform computer of the beginning of a message
    Serial.write(PORT_MSG);

    // Return the results over the serial bus.
    // Results are printed in a CSV format.
    // prediction_time, prediction

    // Print data to the serial bus.
    Serial.print(prediction_time);
    Serial.write(',');
    Serial.print(prediction);

    const View<i32>& outputs = Model::getL2Output();

    for(usize i = 0; i < outputs.len; i++)
    {
        Serial.write(',');
        Serial.print(outputs.data[i]);
    }

    // Inform computer of the end of a message and transmission
    Serial.write(PORT_MSG_END);
    Serial.write(PORT_TRN_END);
}

bool Device::AwaitData(void)
{
    const time_t timeout_time = millis() + _timeout_ms;
    while(millis() < timeout_time)
    {
        if(Serial.available()) return true;
    }
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
