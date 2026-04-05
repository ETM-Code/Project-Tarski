#include <cstdio>
#include <cstdarg>
#include "logger.hpp"

namespace Logger
{
    static const char* _exec_name = nullptr;
    static FILE* _log_file = stdout;
    static FILE* _err_file = stderr;
}

void Logger::SetExecutableName(const char* name)
{
    Logger::_exec_name = name;
}

void Logger::DisplayUsage(void)
{
    //  --- Always log to stdout ---
    printf("Usage: %s <port> <-l|-r|-p> [<options>]\n", Logger::_exec_name);
    printf("  <port>: Serial port path. e.g. /dev/ttyUSB0\n");
    printf("  modes:\n");
    printf("    -l              Load binary sample data to the Arduino (requires -i <path>)\n");
    printf("    -r              Run inference on already-loaded data and read output\n");
    printf("    -p              Load data and then run inference (requires -i <path>)\n");
    printf("  options:\n");
    printf("    -b              Run the program in batch mode. Runs inference on multiple data sets.\n");
    printf("    -f              Repeat selected mode action for a fixed duration (-l or -r only).\n");
    printf("    -i <path>       Path to the input file\n");
    printf("    -o <path>       Path to the output file\n");
    printf("    -t <seconds>    Duration of repeat mode (requires -f).\n");
}

void Logger::Info(const char* fmt, ...)
{
    // Start variadic arguments list
    va_list arguments;
    va_start(arguments, fmt);

    // Log message
    fprintf(Logger::_log_file, "INFO: ");
    vfprintf(Logger::_log_file, fmt, arguments);
    fprintf(Logger::_log_file, ".\n");    

    // End the variadic arguments list
    va_end(arguments);
}

void Logger::Error(const char* fmt, ...)
{
    // Start variadic arguments list
    va_list arguments;
    va_start(arguments, fmt);

    // Log message
    fprintf(Logger::_err_file, "ERROR: ");
    vfprintf(Logger::_err_file, fmt, arguments);
    fprintf(Logger::_err_file, ".\n");    

    // End the variadic arguments list
    va_end(arguments);
}
