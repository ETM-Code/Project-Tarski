#include <chrono>
#include <vector>
#include "runner.hpp"
#include "file_handler.hpp"
#include "logger.hpp"

namespace Runner
{
    static bool ReadSample(std::vector<u8>& sample_data, const char* path, std::size_t minimum_size)
    {
        if(!FileHandler::ReadBinaryImage(sample_data, path))
            return false;

        if(sample_data.size() < minimum_size)
        {
            Logger::Error("Binary image file is too small.");
            return false;
        }

        return true;
    }
}

bool Runner::DataLoad(Arduino& device, const CLI::Arguments& args)
{
    //  --- Allocate a buffer for the sample data ---
    std::vector<u8> sample_data;

    if(!ReadSample(sample_data, args.input, Arduino::DATA_SIZE))
        return false;

    if(!args.repeat_mode)
    {
        //  --- Send the image data to the device ---
        if(!device.loadData(sample_data))
        {
            Logger::Error("Failed to load sample data to device");
            return false;
        }
        return true;
    }

    const auto end_time = std::chrono::steady_clock::now() + std::chrono::seconds(args.duration_seconds);
    unsigned int send_count = 0;

    Logger::Info("Sending sample repeatedly for %u second(s)", args.duration_seconds);
    while(std::chrono::steady_clock::now() < end_time)
    {
        if(!device.loadData(sample_data))
        {
            Logger::Error("Failed to load sample data during repeated load");
            return false;
        }
        send_count++;
    }

    Logger::Info("Repeated send completed, sent %u sample(s)", send_count);
    return true;
}

bool Runner::Inference(Arduino& device, const CLI::Arguments& args)
{
    FileHandler::File data_sink = FileHandler::OpenDataSink(args.output);
    if(!data_sink)
        return false;

    if(!args.repeat_mode)
    {
        //  --- Set correct_prediction to 0xF0 to disable checking ---
        if(!device.runClassification(data_sink, 0xF0))
        {
            Logger::Error("Failed to run inference");
            return false;
        }
        return true;
    }

    const auto end_time = std::chrono::steady_clock::now() + std::chrono::seconds(args.duration_seconds);
    unsigned int infer_count = 0;

    Logger::Info("Running inference repeatedly for %u second(s)", args.duration_seconds);
    while(std::chrono::steady_clock::now() < end_time)
    {
        if(!device.runClassification(data_sink, 0xF0))
        {
            Logger::Error("Failed during repeated inference");
            return false;
        }
        infer_count++;
    }

    Logger::Info("Repeated inference completed, ran %u inference(s)", infer_count);
    return true;
}

bool Runner::Prediction(Arduino& device, const CLI::Arguments& args)
{
    //  --- Allocate a buffer for the sample and echoed data ---
    std::vector<u8> sample_data;

    FileHandler::File data_sink = FileHandler::OpenDataSink(args.output);
    if(!data_sink)
        return false;

    if(!args.batch_mode)
    {
        if(!ReadSample(sample_data, args.input, Arduino::DATA_SIZE + 1))
            return false;

        const u8 classification = sample_data[Arduino::DATA_SIZE];
        if(!device.loadData(sample_data))
        {
            Logger::Error("Failed to load sample data for prediction");
            return false;
        }
        if(!device.runClassification(data_sink, classification))
        {
            Logger::Error("Failed to run prediction");
            return false;
        }
        return true;
    }

    if(!FileHandler::ForEachBinFile(args.input, [&](const char* path) -> bool
        {
            Logger::Info("Processing: %s", path);

            if(!ReadSample(sample_data, path, Arduino::DATA_SIZE + 1))
                return false;

            const u8 classification = sample_data[Arduino::DATA_SIZE];
            if(!device.loadData(sample_data))
            {
                Logger::Error("Failed to load sample '%s'", path);
                return false;
            }
            if(!device.runClassification(data_sink, classification))
            {
                Logger::Error("Failed to classify sample '%s'", path);
                return false;
            }
            return true;
        }))
            return false;

    return true;
}
