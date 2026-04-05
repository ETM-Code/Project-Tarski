#include <cstdio>
#include "arduino.hpp"
#include "logger.hpp"

/**
 * Helper functions for common Arduino protocol errors.
 */
namespace Error
{
    /**
     * These helper functions always return false as a convenience
     * for compact early-return error handling.
     */

    static bool DeviceTimeout(void)
    {
        Logger::Error("Device did not respond");
        return false;
    }

    static bool DeviceNAK(const char* action)
    {
        Logger::Error("Device did not acknowledge %s", action);
        return false;
    }
}

Arduino::Arduino(const char* path, Baudrate baudrate, u32 timeout_ms)
    : Serial(path, baudrate, timeout_ms)
{}

bool Arduino::printVersion(void)
{
    Logger::Info("Awaiting device signature");

    this->writeByte(PORT_SIG);                  // Request device signature
    
    // Await for the device to respond
    if(!this->awaitData()) return ::Error::DeviceTimeout();

    u8 response = this->readByte();

    if(response != PORT_ACK)                    // Device has not acknowledged
        return ::Error::DeviceNAK("signature request");

    u8 version[3];

    if(!this->awaitData(sizeof(version)))       // Device didn't transmit enough data in time
    {
        Logger::Error("Device has not transmitted requested data in time");
        return false;
    }

    this->readBytes(version, 3);                // Read version data from the serial interface

    for(u8 i = 0; i < sizeof(version); i++)     // Transmitted version numbers are shifted up by 0x30
    {
        version[i] -= 0x30;                     // They must now be decremented
    }

    // Print device signature
    Logger::Info("Device firmware version: %d.%d.%d", version[0], version[1], version[2]);

    if(!this->awaitData())
    {
        Logger::Error("Device has not sent End Of Transmission");
        return false;
    }

    response = this->readByte();

    // Ensure transmission was ended with an EOT
    if(response != PORT_TRN_END)
    {
        Logger::Error("Device did not terminate communication appropriately");
        return false;
    }

    return true;
}

bool Arduino::loadData(std::span<u8> data)
{
    //  --- Allocate a buffer for the echoed data ---
    u8 data_echo[DATA_SIZE];

    this->writeByte(PORT_LOAD);
    Logger::Info("Sent sample data load request to device");

    // Await for the device to respond
    if(!this->awaitData()) return ::Error::DeviceTimeout();
    if(this->readByte() != PORT_ACK)                            // Device has not acknowledged
        return ::Error::DeviceNAK("load request");

    Logger::Info("Device acknowledged load request");
    this->writeBytes(data.data(), DATA_SIZE);                   // Write the sample data to the device.

    // Await for the device to respond
    if(!this->awaitData()) return ::Error::DeviceTimeout();
    if(this->readByte() != PORT_ACK)                            // Device has not acknowledged
        return ::Error::DeviceNAK("data received");

    Logger::Info("Device acknowledged data received");

    // Await data echo
    if(!this->awaitData(DATA_SIZE))
    {
        Logger::Error("Device has not echoed image data in time");
        return false;
    }
    this->readBytes(data_echo, DATA_SIZE);                      // Read echoed sample data

    // Compare TX'd and RX'd data
    bool valid = true;
    for(u32 i = 0; i < DATA_SIZE; i++)
    {
        if(data[i] != data_echo[i]) { valid = false; break; }
    }

    // Confirm if data was transmitted successfully
    if(!valid)
    {
        this->writeByte(PORT_NAK);
        Logger::Error("Device did not echo image data successfully");
        return false;
    }

    this->writeByte(PORT_ACK);
    Logger::Info("Device echoed sample data successfully");
    Logger::Info("Data load successful");

    return true;
}

bool Arduino::runClassification(FILE* output_sink, u8 correct_classification)
{
    this->writeByte(PORT_INFER);
    Logger::Info("Sent infer data request to device");

    // Await for the device to respond
    if(!this->awaitData()) return ::Error::DeviceTimeout();

    if(this->readByte() != PORT_ACK)           // Device has not acknowledged
        return ::Error::DeviceNAK("infer request");

    Logger::Info("Device acknowledged infer request");

    // Await for the device to respond
    if(!this->awaitData(2)) return ::Error::DeviceTimeout();
    u8 prediction = this->readByte();
    if(this->readByte() == PORT_MSG)           // Device has sent a message
    {
        Logger::Info("Device transmitting message");
        while(this->awaitData())
        {
            u8 data = this->readByte();
            if(data == PORT_MSG_END) break;

            std::fputc(data, output_sink);
        }
        if(correct_classification != 0xF0)
            std::fprintf(output_sink, ",%hhu,%s", correct_classification, prediction == correct_classification ? "true" : "false");
        std::fputc('\n', output_sink);
        Logger::Info("Device ended message transmission");
    }
    else
    {
        // Device did not follow the expected `prediction + PORT_MSG` response format.
        // Keep flow unchanged for now; protocol error handling can be added here later.
    }

    // Await for the device to end transmission
    if(!this->awaitData()) return ::Error::DeviceTimeout();
    if(this->readByte() != PORT_TRN_END)                // Device has not ended transmission successfully
    {
        Logger::Error("Device has not ended transmission successfully");
        return false;
    }

    return true;
}
