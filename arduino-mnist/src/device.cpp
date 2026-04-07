#include <Arduino.h>
#include "device.hpp"
#include "signals.hpp"
#include "version.hpp"
#include "datastructs.hpp"
#include "model.hpp"

namespace Device
{
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

void Device::LoadHiddenData(void)
{
    Serial.write(PORT_ACK);

    for(u8 i = 0; i < Model::HIDDEN_FEATURE_SIZE; i++)
    {
        if(!Device::AwaitData()) break;
        _hidden_data[i] = static_cast<i8>(Serial.read());
    }

    Serial.write(PORT_ACK);
    Serial.write(PORT_TRN_END);
}

void Device::RunInference(void)
{
    Serial.write(PORT_ACK);

    const u8 prediction = Model::PredictClassFromHidden(_hidden_input);
    Serial.write(prediction);
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
    return false;
}
