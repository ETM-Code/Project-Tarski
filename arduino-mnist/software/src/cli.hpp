#pragma once

/**
 * Command-line argument parser for arduino-interface.
 * Validates flags and fills `CLI::Arguments` with selected mode, paths,
 * and optional timed repeat settings used by runtime execution.
 */

namespace CLI
{
    enum class Valid
    {
        OPT_LOAD,
        OPT_RUN,
        OPT_PREDICT
    };

    /**
     * List of valid operations that the program can perform on the Arduino.
     */
    enum class Operation : char
    {
        NONE    =  0 ,
        LOAD    = 'l',
        RUN     = 'r',
        PREDICT = 'p',
    };

    /**
     * Stores the arguments parsed from the command line.
     */
    struct Arguments
    {
        const char* input               = nullptr;
        const char* output              = nullptr;
        const char* dev_path            = nullptr;
        unsigned int duration_seconds   = 0;
        Operation op                    = Operation::NONE;
        bool batch_mode                 = false;
        bool repeat_mode                = false;
    };

    /**
     * Parse the provided arguments from `argv` and store the results into `args_dest`.
     * @param args_dest Memory to store the parsed arguments into.
     * @param argc The number of arguments to parse.
     * @param argv The array of arguments to parse.
     * @return `true` if parsing was successful, `false` otherwise.
     */
    bool ParseArguments(Arguments& args_dest, int argc, char** argv);

    /**
     * Validate parsed arguments for a specific operation target.
     * Logs validation errors and returns false when invalid.
     * @param args The parsed arguments to validate.
     * @param target The operation context to validate for.
     * @return `true` if arguments are valid for the target, `false` otherwise.
     */
    bool Validate(const Arguments& args, Valid target);
};
