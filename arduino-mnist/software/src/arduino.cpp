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

bool Arduino::loadData(std::span<u8>)
{
    Logger::Error("Raw input load is no longer supported");
    return false;
}

bool Arduino::loadHiddenData(std::span<i8> hidden)
{
    if(hidden.size() < HIDDEN_SIZE)
    {
        Logger::Error("Hidden vector too small (expected %zu bytes)", HIDDEN_SIZE);
        return false;
    }

    this->writeByte(PORT_LOAD_HIDDEN);
    Logger::Info("Sent hidden-activation load request to device");

    if(!this->awaitData()) return ::Error::DeviceTimeout();
    if(this->readByte() != PORT_ACK) return ::Error::DeviceNAK("hidden load request");

    this->writeBytes(reinterpret_cast<const u8*>(hidden.data()), HIDDEN_SIZE);

    if(!this->awaitData()) return ::Error::DeviceTimeout();
    if(this->readByte() != PORT_ACK) return ::Error::DeviceNAK("hidden data received");

    if(!this->awaitData()) return ::Error::DeviceTimeout();
    if(this->readByte() != PORT_TRN_END)
    {
        Logger::Error("Hidden load did not terminate communication appropriately");
        return false;
    }

    return true;
}

bool Arduino::runClassification(FILE* output_sink, u8 correct_classification)
{
    this->writeByte(PORT_INFER);
    Logger::Info("Sent infer data request to device");

    if(!this->awaitData()) return ::Error::DeviceTimeout();
    if(this->readByte() != PORT_ACK) return ::Error::DeviceNAK("infer request");

    if(!this->awaitData()) return ::Error::DeviceTimeout();
    const u8 prediction = this->readByte();

    if(!this->awaitData()) return ::Error::DeviceTimeout();
    if(this->readByte() != PORT_TRN_END)
    {
        Logger::Error("Device has not ended transmission successfully");
        return false;
    }

    if(correct_classification != 0xF0)
        std::fprintf(output_sink, "-1,%hhu,%hhu,%s\n", prediction, correct_classification, prediction == correct_classification ? "true" : "false");
    else
        std::fprintf(output_sink, "-1,%hhu\n", prediction);

    return true;
}
