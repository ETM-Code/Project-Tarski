#include <cstdlib>
#include "arduino.hpp"
#include "cli.hpp"
#include "logger.hpp"
#include "runner.hpp"

/**
 * Display a usage message to the console.
 * Then exit the program with an error code.
 */
[[noreturn]]
static void ExitUsage(void)
{
    Logger::DisplayUsage();
    exit(EXIT_FAILURE);
}

int main(int argc, char** argv)
{
    // Parse the provided command line arguments
    CLI::Arguments args;
    if(!CLI::ParseArguments(args, argc, argv)) ExitUsage();

    // Exit program if no mode argument was provided
    if(args.op == CLI::Operation::NONE) ExitUsage();

    // Open the arduino's serial port
    Arduino arduino(args.dev_path, Serial::Baudrate::BR9600, 5000);

    if(!arduino.available())
    {
        switch(arduino.getError())
        {
            case Serial::Error::INVALID_PATH:
                Logger::Error("Unable to open serial port path");
                return EXIT_FAILURE;

            case Serial::Error::INVALID_CONFIG:
                Logger::Error("Unable to set the requested configuration");
                return EXIT_FAILURE;

            default: break;
        }
    }

    /* Port is now ready for serial communication */

    // Obtain device signature to ensure we are communicating with the correct device
    if(!arduino.printVersion()) return EXIT_FAILURE;

    switch(args.op)
    {
        case CLI::Operation::LOAD:
            if(!CLI::Validate(args, CLI::Valid::OPT_LOAD)) ExitUsage();
            if(!Runner::DataLoad(arduino, args)) return EXIT_FAILURE;
            break;

        case CLI::Operation::RUN:
            if(!CLI::Validate(args, CLI::Valid::OPT_RUN)) ExitUsage();
            if(!Runner::Inference(arduino, args)) return EXIT_FAILURE;
            break;

        case CLI::Operation::PREDICT:
            if(!CLI::Validate(args, CLI::Valid::OPT_PREDICT)) ExitUsage();
            if(!Runner::Prediction(arduino, args)) return EXIT_FAILURE;
            break;

        case CLI::Operation::NONE:
            ExitUsage();
    }

    return 0;
}
