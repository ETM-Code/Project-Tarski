#include <Arduino.h>
#include <avr/sleep.h>
#include <avr/interrupt.h>
#include <avr/power.h>
#include <avr/wdt.h>
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
    static constexpr u8 _FLAG_ADC_ENABLE = 1;
    static constexpr u8 _MEAS_SOURCE_L1 = 0;
    static constexpr u8 _MEAS_SOURCE_L2 = 1;

    static volatile bool _wdt_elapsed = false;
    static u8 _wdt_setting = 0;

    static u8 _response;
    
    static u8 _weight_data[CONF_SYNAPSE_COUNT];
    static View<u8> _weights = { CONF_SYNAPSE_COUNT, _weight_data };

    static u16 _output;
    static u16 _L1_meas;
    static u16 _L2_meas;

    static u16 _dac_data[CONF_DAC_COUNT];
    static View<u16> _dacs = { CONF_DAC_COUNT, _dac_data };

    ISR(WDT_vect) { _wdt_elapsed = true; }

    static u8 _WDTFromTimeout(Timeout timeout)
    {
        switch(timeout)
        {
            case Timeout::TO16MS:   return 0;
            case Timeout::TO32MS:   return _BV(WDP0);
            case Timeout::TO64MS:   return _BV(WDP1);
            case Timeout::TO125MS:  return _BV(WDP1) | _BV(WDP0);
            case Timeout::TO250MS:  return _BV(WDP2);
            case Timeout::TO500MS:  return _BV(WDP2) | _BV(WDP0);
            case Timeout::TO1S:     return _BV(WDP2) | _BV(WDP1);
            case Timeout::TO2S:     return _BV(WDP2) | _BV(WDP1) | _BV(WDP0);
            case Timeout::TO4S:     return _BV(WDP3);
            case Timeout::TO8S:     return _BV(WDP3) | _BV(WDP0);
            default:                return _BV(WDP2) | _BV(WDP1);
        }
    }

    static void _WDTStart(void)
    {
        _wdt_elapsed = false;
        MCUSR &= ~_BV(WDRF);

        cli();
        wdt_reset();
        WDTCSR = _BV(WDCE) | _BV(WDE);
        WDTCSR = _BV(WDIE) | _wdt_setting;
        sei();
    }

    static void _WDTStop(void)
    {
        cli();
        wdt_disable();
        sei();
    }

    static void _EnableADC(void)
    {
        ADCSRA |= _BV(ADEN);
    }

    static void _DisableADC(void)
    {
        ADCSRA &= ~_BV(ADEN);
    }

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

            case _FLAG_ADC_ENABLE:
                if(enabled)
                    _EnableADC();
                else
                    _DisableADC();
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

void Device::Init(Timeout timeout)
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
    _wdt_setting = _WDTFromTimeout(timeout);

    // Disable peripherals that are never used by this firmware.
    power_spi_disable();
    power_timer1_disable();
    power_timer2_disable();
    ACSR |= _BV(ACD); // Disable analog comparator
    TIMSK0 &= ~_BV(TOIE0); // Disable Timer0 overflow interrupt

    // Disable ADC to reduce power consumption
    _DisableADC();

    // Enable interrupts
    sei();
}

void Device::Idle(void)
{
    if(!Serial.available())
    {
        set_sleep_mode(SLEEP_MODE_IDLE);
        cli();
        sleep_enable();
        sei();
        sleep_cpu();            // Device sleeps here until an interrupt arrives
        sleep_disable();
    }
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
    if(Serial.available()) return true;

    // Start timeout timer
    _WDTStart();

    // Keep sleeping until timer expires or we recieve data
    while(!Serial.available() && !_wdt_elapsed)
    {
        set_sleep_mode(SLEEP_MODE_IDLE);
        cli();
        sleep_enable();
        sei();
        sleep_cpu();
        sleep_disable();
    }

    _WDTStop();

    return Serial.available();
}

bool Device::AwaitResponse(void)
{
    // Set response to known value
    _response = 0x00;
    
    if(!Serial.available())
    {
        // Start timeout timer
        _WDTStart();
    
        // Keep sleeping until timer expires or we recieve data
        while(!Serial.available() && !_wdt_elapsed)
        {
            set_sleep_mode(SLEEP_MODE_IDLE);
            cli();
            sleep_enable();
            sei();
            sleep_cpu();
            sleep_disable();
        }
    
        _WDTStop();
    }    

    if(Serial.available()) _response = Serial.read();

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
        analogRead(PIN_L1_MEAS_OUT);            // Dummy read to settle ADC
        _L1_meas = static_cast<u16>(analogRead(PIN_L1_MEAS_OUT));
    }
    else if(measurement_source == _MEAS_SOURCE_L2)
    {
        digitalWrite(PIN_L1_EN_MEAS, LOW);
        digitalWrite(PIN_L2_EN_MEAS, HIGH);
        analogRead(PIN_L2_MEAS_OUT);            // Dummy read to settle ADC
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

void Device::RunInference(void)
{
    Serial.write(PORT_ACK);

    // Read parameters: num_samples, interval_us (u16 little-endian)
    u8 num_samples = 0;
    u8 interval_lo = 0;
    u8 interval_hi = 0;
    if(!ReadU8(num_samples) || !ReadU8(interval_lo) || !ReadU8(interval_hi))
    {
        _SendFailure();
        return;
    }

    const u16 interval_us = static_cast<u16>(interval_lo) | (static_cast<u16>(interval_hi) << 8);

    // Cap at 250 samples (500 bytes) to stay within Arduino RAM
    if(num_samples > 250 || num_samples == 0)
    {
        _SendFailure();
        return;
    }

    // Buffer for spike snapshots (stored on stack, max 500 bytes)
    u16 samples[250];

    // Reset SR latches before starting
    _PulsePin(PIN_RESET_SR);

    // Sample loop: read latches, shift in data, reset latches, wait
    for(u8 i = 0; i < num_samples; i++)
    {
        // Wait the specified interval (neurons are computing during this time)
        if(i > 0) delayMicroseconds(interval_us);

        // Parallel-load: capture current latch state into 74HC165
        digitalWrite(PIN_PARALLEL_LD, LOW);
        digitalWrite(PIN_PARALLEL_LD, HIGH);

        // Shift in 16 bits from the 74HC165 chain
        samples[i] = _ReadShiftRegisterWord();

        // Reset SR latches so next sample only captures NEW spikes
        _PulsePin(PIN_RESET_SR);
    }

    // Send all samples back
    for(u8 i = 0; i < num_samples; i++)
    {
        SendU16(samples[i]);
    }
    Serial.write(PORT_TRN_END);
}

void Device::CalibL1(void)
{
    Serial.write(PORT_ACK);

    // Read parameters
    u8 dac_channel = 0;
    u8 code_lo = 0, code_hi = 0;
    u8 wait_lo = 0, wait_hi = 0;
    u8 meas_channel = 0;

    if(!ReadU8(dac_channel) || !ReadU8(code_lo) || !ReadU8(code_hi) ||
       !ReadU8(wait_lo) || !ReadU8(wait_hi) || !ReadU8(meas_channel))
    {
        _SendFailure();
        return;
    }

    const u16 dac_code = static_cast<u16>(code_lo) | (static_cast<u16>(code_hi) << 8);
    const u16 max_wait_ms = static_cast<u16>(wait_lo) | (static_cast<u16>(wait_hi) << 8);

    // Validate
    if(dac_channel >= CONF_DAC_COUNT || meas_channel > 1)
    {
        _SendFailure();
        return;
    }

    // Select measurement pin
    const u8 meas_pin = (meas_channel == 0) ? PIN_L1_MEAS_OUT : PIN_L2_MEAS_OUT;

    // Enable measurement path
    if(meas_channel == 0)
    {
        digitalWrite(PIN_L1_EN_MEAS, HIGH);
        digitalWrite(PIN_L2_EN_MEAS, LOW);
    }
    else
    {
        digitalWrite(PIN_L1_EN_MEAS, LOW);
        digitalWrite(PIN_L2_EN_MEAS, HIGH);
    }

    // Zero all DACs first (let neuron membrane reset)
    for(usize i = 0; i < CONF_DAC_COUNT; i++) _dacs.data[i] = 0;
    _WriteDACData(CONF_DAC_COUNT);
    delay(10); // Wait for membrane to decay to resting potential

    // Set the target DAC channel
    _dacs.data[dac_channel] = dac_code & ((1U << CONF_DAC_BITS) - 1U);
    _WriteDACData(CONF_DAC_COUNT);

    // Start timing and poll for spike
    const unsigned long start_us = micros();
    const unsigned long timeout_us = static_cast<unsigned long>(max_wait_ms) * 1000UL;
    u32 elapsed_us = 0xFFFFFFFF; // Default: timeout

    // Use analogRead with a threshold: neuron comparator output
    // swings from ~0V (no spike) to ~4.5V (spike). ADC returns 0-1023.
    // Threshold at ~2.5V = ADC value ~512.
    const u16 spike_threshold = 512;

    while((micros() - start_us) < timeout_us)
    {
        if(static_cast<u16>(analogRead(meas_pin)) > spike_threshold)
        {
            elapsed_us = static_cast<u32>(micros() - start_us);
            break;
        }
    }

    // Zero the DAC to reset neuron
    _dacs.data[dac_channel] = 0;
    _WriteDACData(CONF_DAC_COUNT);

    // Disable measurement path
    digitalWrite(PIN_L1_EN_MEAS, LOW);
    digitalWrite(PIN_L2_EN_MEAS, LOW);

    // Send result: 4 bytes (u32 little-endian)
    SendU8(static_cast<u8>(elapsed_us & 0xFF));
    SendU8(static_cast<u8>((elapsed_us >> 8) & 0xFF));
    SendU8(static_cast<u8>((elapsed_us >> 16) & 0xFF));
    SendU8(static_cast<u8>((elapsed_us >> 24) & 0xFF));
    Serial.write(PORT_TRN_END);
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

    // MCP4728 address programming (DS20002532 Section 7.3).
    //
    // This requires bit-banging I2C because LDAC must be toggled at a
    // precise point DURING the I2C transaction — between the 8th and 9th
    // SCL clock of byte 2. The Wire library can't do this.
    //
    // Protocol:
    //   1. LDAC HIGH
    //   2. START condition
    //   3. Send current address byte (write): [1100_A2A1A0_0]
    //   4. Wait for ACK
    //   5. Send command byte 1: [0110_0001] | (old_bits << 2)
    //   6. Wait for ACK
    //   7. Send command byte 2: [0110_0010] | (new_bits << 2)
    //   8. After 8th SCL clock of byte 2, pull LDAC LOW before 9th clock (ACK)
    //   9. Wait for ACK
    //   10. Send command byte 3: [0110_0011] | (new_bits << 2)
    //   11. Wait for ACK
    //   12. STOP condition

    // Disable Wire library to take manual control of SDA/SCL
    Wire.end();

    const u8 sda_pin = PIN_I2C_SDA;
    const u8 scl_pin = PIN_I2C_SCL;

    pinMode(sda_pin, OUTPUT);
    pinMode(scl_pin, OUTPUT);
    digitalWrite(sda_pin, HIGH);
    digitalWrite(scl_pin, HIGH);

    u8 old_bits = old_addr & 0x07;
    u8 new_bits = new_addr & 0x07;

    // Helper: clock out one bit on I2C
    auto i2c_bit = [&](bool bit) {
        digitalWrite(sda_pin, bit ? HIGH : LOW);
        delayMicroseconds(5);
        digitalWrite(scl_pin, HIGH);
        delayMicroseconds(5);
        digitalWrite(scl_pin, LOW);
        delayMicroseconds(5);
    };

    // Helper: clock out one byte, return ACK
    auto i2c_byte = [&](u8 byte) -> bool {
        for(int b = 7; b >= 0; b--) {
            i2c_bit((byte >> b) & 1);
        }
        // ACK: release SDA, clock SCL, read SDA
        pinMode(sda_pin, INPUT_PULLUP);
        delayMicroseconds(5);
        digitalWrite(scl_pin, HIGH);
        delayMicroseconds(5);
        bool ack = (digitalRead(sda_pin) == LOW);
        digitalWrite(scl_pin, LOW);
        pinMode(sda_pin, OUTPUT);
        delayMicroseconds(5);
        return ack;
    };

    // Helper: clock out byte with LDAC toggle before ACK clock
    auto i2c_byte_with_ldac = [&](u8 byte) -> bool {
        for(int b = 7; b >= 0; b--) {
            i2c_bit((byte >> b) & 1);
        }
        // CRITICAL: pull LDAC LOW before the ACK clock
        digitalWrite(PIN_LATCH_DAC, LOW);
        delayMicroseconds(2);
        // ACK clock
        pinMode(sda_pin, INPUT_PULLUP);
        delayMicroseconds(5);
        digitalWrite(scl_pin, HIGH);
        delayMicroseconds(5);
        bool ack = (digitalRead(sda_pin) == LOW);
        digitalWrite(scl_pin, LOW);
        pinMode(sda_pin, OUTPUT);
        delayMicroseconds(5);
        return ack;
    };

    // 1. LDAC HIGH
    digitalWrite(PIN_LATCH_DAC, HIGH);
    delayMicroseconds(100);

    // 2. START condition: SDA goes LOW while SCL is HIGH
    digitalWrite(sda_pin, LOW);
    delayMicroseconds(5);
    digitalWrite(scl_pin, LOW);
    delayMicroseconds(5);

    // 3. Address byte (write): [1100_A2A1A0_0]
    u8 addr_byte = (old_addr << 1) & 0xFE; // 7-bit address + write bit
    bool ack1 = i2c_byte(addr_byte);

    // 4. Command byte 1: current address bits
    u8 cmd1 = 0x61 | (old_bits << 2);
    bool ack2 = i2c_byte(cmd1);

    // 5. Command byte 2 WITH LDAC toggle: new address bits
    u8 cmd2 = 0x62 | (new_bits << 2);
    bool ack3 = i2c_byte_with_ldac(cmd2);

    // 6. Command byte 3: confirm new address
    u8 cmd3 = 0x63 | (new_bits << 2);
    bool ack4 = i2c_byte(cmd3);

    // 7. STOP condition: SDA goes HIGH while SCL is HIGH
    digitalWrite(sda_pin, LOW);
    delayMicroseconds(5);
    digitalWrite(scl_pin, HIGH);
    delayMicroseconds(5);
    digitalWrite(sda_pin, HIGH);
    delayMicroseconds(5);

    // Restore LDAC to normal (LOW = immediate update)
    digitalWrite(PIN_LATCH_DAC, LOW);

    // Re-enable Wire library
    Wire.begin();

    delay(50); // EEPROM write time

    // Verify: try to communicate with the new address
    Wire.beginTransmission(new_addr);
    bool verified = (Wire.endTransmission() == 0);

    if(verified && ack1 && ack2 && ack3 && ack4)
    {
        _SendSuccess();
        return;
    }

    // Retry: power cycle can sometimes be needed.
    // Try again with a fresh general call reset.
    Wire.end();
    delay(10);
    Wire.begin();

    // General call reset
    Wire.beginTransmission(0x00);
    Wire.write(0x06);
    Wire.endTransmission();
    delay(5);

    // Re-verify with the new address
    Wire.beginTransmission(new_addr);
    if(Wire.endTransmission() == 0)
    {
        _SendSuccess();
        return;
    }

    // Also check if device is still at old address (programming failed)
    Wire.beginTransmission(old_addr);
    bool still_at_old = (Wire.endTransmission() == 0);

    // Report detailed status: byte 1 = result code
    //   0x01 = success (verified at new address)
    //   0x02 = failed, device still at old address
    //   0x03 = failed, device not responding at either address
    //   0x04 = partial ACK failure (I2C framing issue)
    u8 status = still_at_old ? 0x02 : 0x03;
    if(!ack1 || !ack2 || !ack3 || !ack4) status = 0x04;

    Serial.write(PORT_NAK);
    SendU8(status);
    Serial.write(PORT_TRN_END);
}
