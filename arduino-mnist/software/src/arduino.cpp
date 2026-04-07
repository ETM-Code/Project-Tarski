#include <cstdio>
#include "arduino.hpp"
#include "logger.hpp"

namespace Error
{
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
    this->writeByte(PORT_SIG);

    if(!this->awaitData()) return ::Error::DeviceTimeout();
    if(this->readByte() != PORT_ACK) return ::Error::DeviceNAK("signature request");

    u8 version[3];
    if(!this->awaitData(sizeof(version)))
    {
        Logger::Error("Device has not transmitted requested data in time");
        return false;
    }
    this->readBytes(version, 3);
    for(u8& v : version) v -= 0x30;

    Logger::Info("Device firmware version: %d.%d.%d", version[0], version[1], version[2]);

    if(!this->awaitData()) return ::Error::DeviceTimeout();
    if(this->readByte() != PORT_TRN_END)
    {
        Logger::Error("Device did not terminate communication appropriately");
        return false;
    }
    return true;
}

bool Arduino::loadData(const std::span<i8> data)
{
    // Validate input data size
    if(data.size() < HIDDEN_SIZE)
    {
        Logger::Error("Data size too small (expected %zu bytes)", HIDDEN_SIZE);
        return false;
    }

    // Allocate a buffer for the echoed data
    u8 data_echo[HIDDEN_SIZE];

    // Send command to arduino
    this->writeByte(PORT_LOAD);
    Logger::Info("Sent load request to device");

    // Await for the device to respond
    if(!this->awaitData()) return ::Error::DeviceTimeout();
    if(this->readByte() != PORT_ACK) return ::Error::DeviceNAK("load request");
    Logger::Info("Device acknowledged load request");

    // Send sample data to the arduino
    this->writeBytes(reinterpret_cast<const u8*>(data.data()), HIDDEN_SIZE);

    // Wait for the device to acknowledge data recieved
    if(!this->awaitData()) return ::Error::DeviceTimeout();
    if(this->readByte() != PORT_ACK) return ::Error::DeviceNAK("data received");
    Logger::Info("Device acknowledged data received");

    // Await for the arduino to echo the data for validation
    if(!this->awaitData(HIDDEN_SIZE))
    {
        Logger::Error("Device has not echoed image data in time");
        return false;
    }
    this->readBytes(data_echo, HIDDEN_SIZE);

    // Compare TX'd and RX'd data
    bool valid = true;
    for(u32 i = 0; i < HIDDEN_SIZE; i++)
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

    // Acknowledge TX'd and RX'd data match
    this->writeByte(PORT_ACK);
    Logger::Info("Device echoed sample data successfully");
    Logger::Info("Data load successful");

    return true;
}

bool Arduino::runClassification(FILE* output_sink, u8 correct_classification)
{
    // Send command to arduino
    this->writeByte(PORT_INFER);
    Logger::Info("Sent infer data request to device");

    // Wait for the device to respond and acknowledge
    if(!this->awaitData())           return ::Error::DeviceTimeout();
    if(this->readByte() != PORT_ACK) return ::Error::DeviceNAK("infer request");
    Logger::Info("Device acknowledged infer request");

    // Wait for the device to respond
    if(!this->awaitData(2)) return ::Error::DeviceTimeout();
    const u8 prediction = this->readByte();
    const u8 response = this->readByte();

    if(response == PORT_MSG) // The deivce is transmitting a message that is to be logged
    {
        Logger::Info("Device transmitting message");
        
        while(this->awaitData())
        {
            u8 data = this->readByte();
            if(data == PORT_MSG_END) break;

            std::fputc(data, output_sink);
        }
        if(correct_classification != NO_CHECK)
            std::fprintf(output_sink, ",%hhu,%s", correct_classification, prediction == correct_classification ? "true" : "false");
        std::fputc('\n', output_sink);
        Logger::Info("Device ended message transmission");
    }
    else if(response == PORT_TRN_END) // The device had no data to send and ended transmission
    {
        // Do nothing in this case
    }
    else // Unexpected device response
    {
        Logger::Error("Device responded with unexpected data: 0x%02X", response);
        return false;
    }

    // Wait for the device to end transmission
    if(!this->awaitData()) return ::Error::DeviceTimeout();
    if(this->readByte() != PORT_TRN_END)
    {
        Logger::Error("Device has not ended transmission successfully");
        return false;
    }

    return true;
}
