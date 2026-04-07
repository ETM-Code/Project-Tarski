#include <chrono>
#include <vector>
#include <array>
#include "runner.hpp"
#include "file_handler.hpp"
#include "logger.hpp"
#include "preprocess.hpp"

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
    std::vector<u8> sample_data;
    std::array<i8, Preprocess::HIDDEN_SIZE> hidden_data{};

    if(!ReadSample(sample_data, args.input, Preprocess::INPUT_SIZE))
        return false;
    if(!Preprocess::ToHiddenActivations(sample_data, hidden_data))
    {
        Logger::Error("Failed to preprocess sample into hidden activations");
        return false;
    }

    if(!args.repeat_mode)
    {
        if(!device.loadData(hidden_data))
        {
            Logger::Error("Failed to load hidden activations to device");
            return false;
        }
        return true;
    }

    const auto end_time = std::chrono::steady_clock::now() + std::chrono::seconds(args.duration_seconds);
    unsigned int send_count = 0;

    Logger::Info("Sending sample repeatedly for %u second(s)", args.duration_seconds);
    while(std::chrono::steady_clock::now() < end_time)
    {
        if(!device.loadData(hidden_data))
        {
            Logger::Error("Failed to load hidden activations during repeated load");
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
        if(!device.runClassification(data_sink, Arduino::NO_CHECK))
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
        if(!device.runClassification(data_sink, Arduino::NO_CHECK))
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
    std::vector<u8> sample_data;
    std::array<i8, Preprocess::HIDDEN_SIZE> hidden_data{};

    FileHandler::File data_sink = FileHandler::OpenDataSink(args.output);
    if(!data_sink)
        return false;

    if(!args.batch_mode)
    {
        if(!ReadSample(sample_data, args.input, Preprocess::INPUT_SIZE + 1))
            return false;

        const u8 classification = sample_data[Preprocess::INPUT_SIZE];
        if(!Preprocess::ToHiddenActivations(sample_data, hidden_data))
        {
            Logger::Error("Failed to preprocess sample into hidden activations");
            return false;
        }
        if(!device.loadData(hidden_data))
        {
            Logger::Error("Failed to load hidden activations for prediction");
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

            if(!ReadSample(sample_data, path, Preprocess::INPUT_SIZE + 1))
                return false;

            const u8 classification = sample_data[Preprocess::INPUT_SIZE];
            if(!Preprocess::ToHiddenActivations(sample_data, hidden_data))
            {
                Logger::Error("Failed to preprocess sample '%s'", path);
                return false;
            }
            if(!device.loadData(hidden_data))
            {
                Logger::Error("Failed to load hidden activations for sample '%s'", path);
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
