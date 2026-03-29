#pragma once

/**
 * This file defines the device-side serial command handlers.
**/

#include "datatypes.hpp"

namespace Device
{
    /**
     * Initialise module state used by the serial command interface.
     * @param timeout_ms Timeout (in milliseconds) used by blocking wait helpers.
     */
    void Init(time_t timeout_ms);

    /**
     * Send the device firmware signature over serial.
     * The response is framed as ACK + version bytes + transmission end marker.
     */
    void SendSignature(void);

    /**
     * Notify the host that the received command is not recognized.
     * Sends a NAK followed by a transmission end marker.
     */
    void SendUnknownCommand(void);

    /**
     * Wait for any incoming serial data until timeout.
     * Uses the timeout configured by `Device::Init()`.
     * @return `true` if data was received, otherwise `false`.
     */
    bool AwaitData(void);

    /**
     * Wait for a host response byte (ACK or NAK) until timeout.
     * Uses the timeout configured by `Device::Init()`.
     * @return `true` if a valid ACK/NAK response was received, otherwise `false`.
     */
    bool AwaitResponse(void);

    /**
     * Read a single 8-bit value from serial.
     * Uses the timeout configured by `Device::Init()`.
     * @param value Output parameter for the received byte.
     * @return `true` on success, otherwise `false`.
     */
    bool ReadU8(u8& value);

    /**
     * Send a single 8-bit value over serial.
     * @param value The value to transmit.
     */
    void SendU8(u8 value);

    /**
     * Send a 16-bit value over serial in little-endian byte order.
     * @param value The value to transmit.
     */
    void SendU16(u16 value);

    /**
     * Read output data from SR latches connected to the Arduino GPIO pins.
     * These latches store data produced by the connected accelerator.
     * The read data is transmitted over serial, then the latches are reset.
     */
    void ReadOutput(void);

    /**
     * Read weight data from serial and output it to the synapses.
     */
    void LoadWeights(void);

    /**
     * Read DAC data from serial and set it on one DAC or multiple DACs.
     */
    void LoadDACs(void);

    /**
     * Read either the L1 or L2 voltage output and transmit it over serial.
     */
    void ReadMeasurement(void);

    /**
     * Set a single device flag to enable a feature.
     */
    void SetFlag(void);

    /**
     * Clear a single device flag to disable a feature.
     */
    void UnSetFlag(void);

    /**
     * Toggle a single device flag to enable or disable a feature.
     */
    void ToggleFlag(void);

    /**
     * Run inference with rapid spike sampling.
     *
     * Instead of a single ReadOutput(), this samples the spike latches
     * repeatedly during the inference window, giving spike count data.
     *
     * Protocol: Host sends [num_samples(u8), interval_us_lo(u8), interval_us_hi(u8)].
     * The Arduino resets the SR latches, then loops num_samples times:
     *   - Parallel-load 74HC165 (capture latch state)
     *   - Shift in 16 bits
     *   - Reset SR latches (so next sample captures new spikes only)
     *   - Wait interval_us microseconds
     * Then sends all samples back: num_samples × 2 bytes (little-endian u16).
     *
     * DAC values must be loaded BEFORE calling this command.
     * Weights must be loaded BEFORE calling this command.
     * The neurons compute continuously from when the DACs are set.
     */
    void RunInference(void);

    /**
     * Layer 1 calibration: measure time to first spike for one hidden neuron.
     *
     * Protocol: Host sends [dac_channel(u8), dac_code_lo(u8), dac_code_hi(u8),
     *                        max_wait_ms_lo(u8), max_wait_ms_hi(u8),
     *                        meas_channel(u8)].
     *
     * The Arduino:
     *   1. Zeros all DAC channels
     *   2. Sets the specified channel to the given code
     *   3. Starts a microsecond timer
     *   4. Polls the specified MEAS_OUT pin until it goes HIGH or timeout
     *   5. Returns: elapsed_us as u32 (little-endian), or 0xFFFFFFFF for timeout
     *
     * The host must reset the neuron before calling this (e.g., by briefly
     * zeroing the DAC and waiting for membrane to decay).
     */
    void CalibL1(void);

    /**
     * Program a MCP4728 DAC I2C address.
     *
     * All MCP4728s ship with factory address 0x60. To use multiple DACs on the
     * same I2C bus, each must be programmed to a unique address (0x60-0x67).
     *
     * Procedure (requires jumpers to isolate each DAC):
     *   1. Disconnect all DACs except the one being programmed
     *   2. Send PORT_PROG_DAC command with old_address and new_address
     *   3. The Arduino handles the LDAC-based address programming sequence
     *   4. Reconnect all DACs and repeat for the next one
     *
     * Protocol: Host sends [old_address, new_address] after ACK.
     * The MCP4728 stores the new address in its EEPROM (persistent).
     */
    void ProgramDACAddress(void);
};  
