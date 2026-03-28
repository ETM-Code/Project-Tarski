#include <Arduino.h>
#include <Wire.h>
#include "device.hpp"
#include "signals.hpp"
#include "version.hpp"
#include "datastructs.hpp"
#include "config.hpp"
#include "pinout.hpp"

namespace Device
{
    static constexpr usize _DACS_PER_MCP4728 = 4;
    static constexpr u8 _FLAG_MEAS_ENABLE = 0;
    static constexpr u8 _MEAS_SOURCE_L1 = 0;
    static constexpr u8 _MEAS_SOURCE_L2 = 1;

    static time_t _timeout_ms;
    static u8 _response;
    static bool _command_impl_approved = false;
    
    static u8 _weight_data[CONF_SYNAPSE_COUNT];
    static View<u8> _weights = { CONF_SYNAPSE_COUNT, _weight_data };

    static u16 _output;
    static u16 _L1_meas;
    static u16 _L2_meas;

    static u16 _dac_data[CONF_DAC_COUNT];
    static View<u16> _dacs = { CONF_DAC_COUNT, _dac_data };

    static void _SendFailure(void)
    {
        Serial.write(PORT_NAK);
        Serial.write(PORT_TRN_END);
    }

    static void _SendSuccess(void)
    {
        Serial.write(PORT_ACK);
        Serial.write(PORT_TRN_END);
    }

    static bool _ReadIntoBuffer(u8* data, usize length)
    {
        for(usize i = 0; i < length; i++)
        {
            if(!Device::ReadU8(data[i])) return false;
        }
        return true;
    }

    static void _PulsePin(u8 pin)
    {
        digitalWrite(pin, HIGH);
        digitalWrite(pin, LOW);
    }

    static u16 _ReadShiftRegisterWord(void)
    {
        u16 value = 0;

        for(u8 bit = 0; bit < 16; bit++)
        {
            value <<= 1;
            value |= static_cast<u16>(digitalRead(PIN_MISO) & 0x01);
            _PulsePin(PIN_SCLK);
        }

        return value;
    }

    static bool _WriteDACData(usize dac_count)
    {

        usize written = 0;
        usize device_index = 0;

        while(written < dac_count)
        {
            if(device_index >= CONF_MCP4728_COUNT) return false;

            const u8 address = CONF_MCP4728_ADDRS[device_index];
            const usize device_count = (dac_count - written > _DACS_PER_MCP4728) ? _DACS_PER_MCP4728 : (dac_count - written);

            // MCP4728 Fast Write:
            // Byte 1 after address has [X X PD1 PD0 D11 D10 D9 D8], with PD=00 (normal mode).
            Wire.beginTransmission(address);
            for(usize ch = 0; ch < device_count; ch++)
            {
                const u16 value = _dacs.data[written + ch] & ((1U << CONF_DAC_BITS) - 1U);
                const u8 byte_1 = static_cast<u8>((value >> 8) & 0x0F);
                const u8 byte_2 = static_cast<u8>(value & 0xFF);
                Wire.write(byte_1);
                Wire.write(byte_2);
            }

            if(Wire.endTransmission() != 0) return false;

            written += device_count;
            device_index++;
        }

        return true;
    }

    static bool _SetSignalFlag(u8 flag, bool enabled)
    {
        const u8 level = enabled ? HIGH : LOW;

        switch(flag)
        {
            case _FLAG_MEAS_ENABLE:
                digitalWrite(PIN_L1_EN_MEAS, level);
                digitalWrite(PIN_L2_EN_MEAS, level);
                return true;

            default:
                return false;
        }
    }

    static bool _ToggleSignalFlag(u8 flag)
    {
        switch(flag)
        {
            case _FLAG_MEAS_ENABLE:
            {
                const u8 next_level = digitalRead(PIN_L1_EN_MEAS) ? LOW : HIGH;
                digitalWrite(PIN_L1_EN_MEAS, next_level);
                digitalWrite(PIN_L2_EN_MEAS, next_level);
                return true;
            }

            default:
                return false;
        }
    }
}

void Device::Init(time_t timeout_ms)
{
    // Initialise GPIO used for serial-style peripheral control.
    pinMode(PIN_LATCH_DAC, OUTPUT);
    pinMode(PIN_PARALLEL_LD, OUTPUT);
    pinMode(PIN_R_CLK, OUTPUT);
    pinMode(PIN_RESET_SR, OUTPUT);
    pinMode(PIN_MOSI, OUTPUT);
    pinMode(PIN_MISO, INPUT);
    pinMode(PIN_SCLK, OUTPUT);
    pinMode(PIN_L2_EN_MEAS, OUTPUT);
    pinMode(PIN_L1_EN_MEAS, OUTPUT);
    pinMode(PIN_OE_S, OUTPUT);
    pinMode(PIN_SRCLR_S, OUTPUT);
    pinMode(PIN_L1_MEAS_OUT, INPUT);
    pinMode(PIN_L2_MEAS_OUT, INPUT);

    // Safe default states.
    digitalWrite(PIN_LATCH_DAC, LOW);    // LDAC active: DACs update immediately on I2C write
    digitalWrite(PIN_PARALLEL_LD, HIGH); // 74HC165: shift mode (not loading)
    digitalWrite(PIN_R_CLK, LOW);        // STCP: no latch
    digitalWrite(PIN_RESET_SR, LOW);     // SR latch reset: inactive
    digitalWrite(PIN_MOSI, LOW);
    digitalWrite(PIN_SCLK, LOW);
    digitalWrite(PIN_L2_EN_MEAS, LOW);   // Measurement paths disabled
    digitalWrite(PIN_L1_EN_MEAS, LOW);
    digitalWrite(PIN_OE_S, HIGH);        // Synapse SR outputs disabled during init

    // Clear synapse shift registers to known state on startup.
    // SRCLR is active-LOW on 74HC595: pulse LOW to clear, then release HIGH.
    digitalWrite(PIN_SRCLR_S, LOW);
    delayMicroseconds(1);
    digitalWrite(PIN_SRCLR_S, HIGH);
    // Latch the cleared state to outputs
    _PulsePin(PIN_R_CLK);

    // Initialize I2C peripheral for MCP4728 DAC communication.
    Wire.begin();

    // Set up the timeout value to the provided one
    _timeout_ms = timeout_ms;
}

void Device::SendSignature(void)
{
    Serial.write(PORT_ACK);	            // Acknowledge request

    // Version numbers shifter up 0x30 to avoid collisions with control characters
    SendU8(FIRM_VER_MJR + 0x30);        // Firmware major version
    SendU8(FIRM_VER_MNR + 0x30);        // Firmware minor version
    SendU8(FIRM_VER_PCH + 0x30);        // Firmware patch version

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

bool Device::ReadU8(u8& value)
{
    if(!Device::AwaitData()) return false;
    value = static_cast<u8>(Serial.read());
    return true;
}

void Device::SendU8(u8 value)
{
    Serial.write(value);
}

void Device::SendU16(u16 value)
{
    SendU8(static_cast<u8>(value & 0xFF));
    SendU8(static_cast<u8>((value >> 8) & 0xFF));
}

void Device::ReadOutput(void)
{
    // Acknowledge command receipt.
    Serial.write(PORT_ACK);

    // Latch external SR outputs, shift data in, then reset latches after read.
    digitalWrite(PIN_PARALLEL_LD, LOW);
    digitalWrite(PIN_PARALLEL_LD, HIGH);
    _output = _ReadShiftRegisterWord();
    _PulsePin(PIN_RESET_SR);


    SendU16(_output);
    Serial.write(PORT_TRN_END);
}

void Device::LoadWeights(void)
{
    Serial.write(PORT_ACK);

    if(!_ReadIntoBuffer(_weights.data, _weights.len))
    {
        _SendFailure();
        return;
    }

    digitalWrite(PIN_OE_S, HIGH);    // Disable outputs during shift
    digitalWrite(PIN_SRCLR_S, HIGH); // Release shift register clear

    for(usize i = 0; i < _weights.len; i++)
    {
        shiftOut(PIN_MOSI, PIN_SCLK, MSBFIRST, _weights.data[i]);
    }

    _PulsePin(PIN_R_CLK);           // Latch shift register → storage register
    digitalWrite(PIN_OE_S, LOW);     // Re-enable outputs
    _SendSuccess();
}

void Device::LoadDACs(void)
{
    Serial.write(PORT_ACK);

    // Host sends count first so a single DAC or a batch can be updated.
    u8 write_count = 0;
    if(!ReadU8(write_count))
    {
        _SendFailure();
        return;
    }

    if(write_count == 0 || write_count > CONF_DAC_COUNT)
    {
        _SendFailure();
        return;
    }

    for(u8 i = 0; i < write_count; i++)
    {
        u8 lsb = 0;
        u8 msb = 0;

        if(!ReadU8(lsb) || !ReadU8(msb))
        {
            _SendFailure();
            return;
        }

        const u16 raw = static_cast<u16>(lsb) | (static_cast<u16>(msb) << 8);
        _dacs.data[i] = raw & ((1U << CONF_DAC_BITS) - 1U);
    }

    if(!_WriteDACData(write_count))
    {
        _SendFailure();
        return;
    }

    _SendSuccess();
}

void Device::ReadMeasurement(void)
{
    Serial.write(PORT_ACK);

    // Host selects channel: 0 = L1, 1 = L2.
    u8 measurement_source = 0;
    if(!ReadU8(measurement_source))
    {
        _SendFailure();
        return;
    }

    if(measurement_source == _MEAS_SOURCE_L1)
    {
        digitalWrite(PIN_L1_EN_MEAS, HIGH);
        digitalWrite(PIN_L2_EN_MEAS, LOW);
        _L1_meas = static_cast<u16>(analogRead(PIN_L1_MEAS_OUT));
    }
    else if(measurement_source == _MEAS_SOURCE_L2)
    {
        digitalWrite(PIN_L1_EN_MEAS, LOW);
        digitalWrite(PIN_L2_EN_MEAS, HIGH);
        _L2_meas = static_cast<u16>(analogRead(PIN_L2_MEAS_OUT));
    }
    else
    {
        _SendFailure();
        return;
    }

    digitalWrite(PIN_L1_EN_MEAS, LOW);
    digitalWrite(PIN_L2_EN_MEAS, LOW);

    const u16 measurement = (measurement_source == _MEAS_SOURCE_L1) ? _L1_meas : _L2_meas;
    SendU16(measurement);
    Serial.write(PORT_TRN_END);
}

void Device::SetFlag(void)
{
    Serial.write(PORT_ACK);

    u8 flag = 0;
    if(!ReadU8(flag))
    {
        _SendFailure();
        return;
    }

    if(!_SetSignalFlag(flag, true))
    {
        _SendFailure();
        return;
    }

    _SendSuccess();
}

void Device::UnSetFlag(void)
{
    Serial.write(PORT_ACK);

    u8 flag = 0;
    if(!ReadU8(flag))
    {
        _SendFailure();
        return;
    }

    if(!_SetSignalFlag(flag, false))
    {
        _SendFailure();
        return;
    }

    _SendSuccess();
}

void Device::ToggleFlag(void)
{
    Serial.write(PORT_ACK);

    u8 flag = 0;
    if(!ReadU8(flag))
    {
        _SendFailure();
        return;
    }

    if(!_ToggleSignalFlag(flag))
    {
        _SendFailure();
        return;
    }

    _SendSuccess();
}

void Device::ProgramDACAddress(void)
{
    Serial.write(PORT_ACK);

    // Read old and new addresses from host
    u8 old_addr = 0;
    u8 new_addr = 0;
    if(!ReadU8(old_addr) || !ReadU8(new_addr))
    {
        _SendFailure();
        return;
    }

    // Validate addresses (MCP4728 uses 0x60-0x67, 7-bit)
    if(old_addr < 0x60 || old_addr > 0x67 || new_addr < 0x60 || new_addr > 0x67)
    {
        _SendFailure();
        return;
    }

    // MCP4728 address programming sequence (from DS20002532 Section 7.3):
    //
    // 1. LDAC must be HIGH initially
    // 2. General Call Reset: write 0x06 to address 0x00
    // 3. Wait >1ms
    // 4. Send "Read Address" command to current address
    // 5. Pull LDAC LOW during the ACK bit of the second byte
    // 6. Send new address bits
    //
    // Simplified sequence using the "General Call" method:
    // Step 1: Set LDAC HIGH
    digitalWrite(PIN_LATCH_DAC, HIGH);
    delayMicroseconds(100);

    // Step 2: General Call Reset (optional, ensures clean state)
    Wire.beginTransmission(0x00); // General call address
    Wire.write(0x06);             // General call reset
    Wire.endTransmission();
    delay(1);

    // Step 3: Write new address using the address bits command
    // Command byte: [1 1 0 0 0 A2 A1 A0] where A2:A0 is the new address bits
    // The MCP4728 7-bit address is 0b110_0xxx where xxx = A2:A0
    u8 old_bits = old_addr & 0x07;
    u8 new_bits = new_addr & 0x07;

    // Address programming I2C frame:
    // Byte 1 (to old address): 0b0110_0001 | (old_bits << 2) = address command
    // Byte 2: 0b0110_0010 | (new_bits << 2) = new address bits
    // Byte 3: 0b0110_0011 | (new_bits << 2) = new address confirmation
    Wire.beginTransmission(old_addr);
    Wire.write(0x61 | (old_bits << 2)); // Current address + command
    Wire.write(0x62 | (new_bits << 2)); // New address bits
    Wire.write(0x63 | (new_bits << 2)); // Confirm new address

    // Pull LDAC LOW during transmission (timing-critical)
    digitalWrite(PIN_LATCH_DAC, LOW);
    delayMicroseconds(1);
    u8 result = Wire.endTransmission();
    digitalWrite(PIN_LATCH_DAC, LOW); // Keep LDAC low for normal operation

    if(result != 0)
    {
        _SendFailure();
        return;
    }

    delay(50); // EEPROM write time

    // Verify: try to communicate with the new address
    Wire.beginTransmission(new_addr);
    if(Wire.endTransmission() == 0)
    {
        _SendSuccess();
    }
    else
    {
        _SendFailure();
    }
}
