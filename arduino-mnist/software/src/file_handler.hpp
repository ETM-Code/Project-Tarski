#pragma once

#include <cstdio>
#include <functional>
#include <vector>
#include "datatypes.hpp"

namespace FileHandler
{
    /**
     * Error types for file operation failures.
     */
    enum class Error : u8
    {
        NO_ERROR = 0,
        INVALID_FILE_TYPE,
        INVALID_PATH,
    };

    class File
    {
        public:

        /**
         * Construct an invalid file wrapper.
         */
        File() = default;

        /**
         * Construct a file wrapper from an existing `FILE*`.
         */
        explicit File(FILE* file);
        File(const File&) = delete;
        File& operator=(const File&) = delete;

        /**
         * Move ownership of the wrapped file handle.
         */
        File(File&& other) noexcept;
        File& operator=(File&& other) noexcept;

        /**
         * Close the wrapped file handle if needed.
         */
        ~File();

        /**
         * Implicit conversion to raw `FILE*` for C APIs.
         */
        operator FILE*() const;

        /**
         * Check whether the wrapper currently owns a valid handle.
         */
        explicit operator bool() const;
        
        /**
         * Close the handle. `stdin`, `stdout`, and `stderr` are never closed.
         */
        void close(void);

        private:
            FILE* _FILE = nullptr;

    };

    /**
     * Open a data sink for writing.
     * If `path` is null, sink is set to `stdout`.
     * @param path Optional file path for output; null selects `stdout`.
     * @return A valid sink wrapper on success, otherwise an invalid wrapper.
     */
    File OpenDataSink(const char* path);

    /**
     * Iterate over all `.bin` files in a directory and invoke a callback for each.
     * @param dir_path Path to the input directory.
     * @param callback Callback invoked for each discovered `.bin` file path.
     * Returning `true` continues iteration, `false` stops iteration early.
     * @return `true` on success, `false` on failure.
     */
    bool ForEachBinFile(const char* dir_path, const std::function<bool(const char*)>& callback);

    /**
     * Load the contents of a binary file into a buffer.
     * @param dest The buffer to put the contents into.
     * @param path The path to the file to read.
     * @return `true` if loading succeeded, `false` otherwise.
     */
    bool ReadBinaryImage(std::vector<u8>& dest, const char* path);
}
