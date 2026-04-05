#include <cstdlib>
#include <climits>
#include <string_view>
#include "cli.hpp"
#include "logger.hpp"

/**
 * Helper functions for common CLI parsing errors.
 */
namespace Error
{
    /**
     * `std::string_view::data()` is safe here because every argument
     * originates from `argv`, which provides null-terminated C strings.
     */

    /**
     * These helper functions always return false as a convenience
     * for one-line parse failure returns.
     */

    static bool UnknownArgument(std::string_view argument)
    {
        Logger::Error("Unknown argument '%s'", argument.data());
        return false;
    }

    static bool MissingArgument(std::string_view src_argument)
    {
        Logger::Error("Missing argument after '%s'", src_argument.data());
        return false;
    }

    static bool DuplicateArgument(std::string_view argument)
    {
        Logger::Error("Duplicate argument '%s' provided", argument.data());
        return false;
    }
}

bool CLI::ParseArguments(Arguments& args_dest, int argc, char** argv)
{
    //  --- Set the logger executable name to the running program ---
    Logger::SetExecutableName(argv[0]);

    //  --- Look for device path argument ---
    if(argc < 2)
    {
        Logger::Error("No device path provided");
        return false;
    }

    args_dest.dev_path = argv[1];

    //  --- Loop over the remaining arguments for parsing ---
    for(int i = 2; i < argc; i++)
    {
        std::string_view argument = argv[i];

        // Single character arguments are not permitted
        if(argument.length() == 1)
            return Error::UnknownArgument(argument);

        // Only allow arguments that start with '-'
        if(!argument.starts_with('-'))
            return Error::UnknownArgument(argument);

        // Only accept arguments of the format '-x'.
        if(argument.length() > 2)
            return Error::UnknownArgument(argument);

        char option = argument.at(1);

        switch(option)
        {
            // Operations
            case 'l':
            case 'r':
            case 'p':
                if(args_dest.op != Operation::NONE)
                {
                    Logger::Error("Mode set more than once.");
                    return false;
                }
                args_dest.op = static_cast<Operation>(option);
                break;

            // Input file set
            case 'i':
                if(args_dest.input) return Error::DuplicateArgument(argument);
                if(i + 1 >= argc)   return Error::MissingArgument(argument);

                args_dest.input = argv[++i];
                break;

            // Output file set
            case 'o':
                if(args_dest.output) return Error::DuplicateArgument(argument);
                if(i + 1 >= argc)    return Error::MissingArgument(argument);

                args_dest.output = argv[++i];
                break;

            // Set batch processing mode
            case 'b':
                if(args_dest.batch_mode) return Error::DuplicateArgument(argument);

                args_dest.batch_mode = true;
                break;

            // Repeatedly run the selected mode operation for a duration
            case 'f':
                if(args_dest.repeat_mode) return Error::DuplicateArgument(argument);

                args_dest.repeat_mode = true;
                break;

            // Duration of repeated send mode in seconds
            case 't':
            {
                if(args_dest.duration_seconds != 0) return Error::DuplicateArgument(argument);
                if(i + 1 >= argc)                  return Error::MissingArgument(argument);

                char* end_ptr = nullptr;
                long duration = strtol(argv[i + 1], &end_ptr, 10);

                if(end_ptr == argv[i + 1] || *end_ptr != '\0')
                {
                    Logger::Error("Invalid duration '%s'. Expected an integer number of seconds", argv[i + 1]);
                    return false;
                }

                if(duration <= 0)
                {
                    Logger::Error("Duration must be greater than zero");
                    return false;
                }
                if(static_cast<unsigned long>(duration) > UINT_MAX)
                {
                    Logger::Error("Duration is too large");
                    return false;
                }

                args_dest.duration_seconds = static_cast<unsigned int>(duration);
                i++;
                break;
            }

            default: return Error::UnknownArgument(argument);
        }
    }
    return true;
}

bool CLI::Validate(const Arguments& args, Valid target)
{
    const bool has_duration = (args.duration_seconds != 0);
    const bool repeat_valid = (!has_duration || args.repeat_mode) && (!args.repeat_mode || has_duration);

    if(!repeat_valid)
    {
        if(has_duration && !args.repeat_mode)
            Logger::Error("Duration (-t) requires repeat mode (-f)");
        else
            Logger::Error("Repeat mode (-f) requires duration (-t <seconds>)");
        return false;
    }

    switch(target)
    {
        case Valid::OPT_LOAD:
            if(!args.input)
            {
                Logger::Error("No image filename provided");
                return false;
            }
            if(args.batch_mode)
            {
                Logger::Error("Cannot perform load operation in batch mode");
                return false;
            }
            return true;

        case Valid::OPT_RUN:
            if(args.batch_mode)
            {
                Logger::Error("Cannot perform run operation in batch mode");
                return false;
            }
            return true;

        case Valid::OPT_PREDICT:
            if(!args.input)
            {
                if(args.batch_mode)
                    Logger::Error("No image directory provided");
                else
                    Logger::Error("No image filename provided");
                return false;
            }
            if(args.repeat_mode)
            {
                Logger::Error("Repeat mode (-f) is only available with load mode (-l) or run mode (-r)");
                return false;
            }
            return true;
    }

    return false;
}
