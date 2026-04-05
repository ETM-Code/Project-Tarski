#pragma once

namespace Logger
{
    /**
     * Set the internal executable name value to the passed in value.
     * @param name The name / path of the currently executed program.
     */
    void SetExecutableName(const char* name);

    /**
     * Display the usage of the program to the console.
     */
    void DisplayUsage(void);

    /**
     * Display an information message to the console.
     * @param fmt The message to display or a formatted string.
     */
    void Info(const char* fmt, ...);

    /**
     * Display an error message to the console.
     * @param fmt The message to display or a formatted string.
     */
    void Error(const char* fmt, ...);
}