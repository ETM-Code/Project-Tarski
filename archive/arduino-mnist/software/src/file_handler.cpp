#include <fstream>
#include <filesystem>
#include "file_handler.hpp"
#include "logger.hpp"

FileHandler::File::File(FILE* file)
    : _FILE(file)
{}

FileHandler::File::File(File&& other) noexcept
    : _FILE(other._FILE)
{
    other._FILE = nullptr;
}

FileHandler::File& FileHandler::File::operator=(File&& other) noexcept
{
    if(this != &other)
    {
        this->close();
        _FILE = other._FILE;
        other._FILE = nullptr;
    }
    return *this;
}

FileHandler::File::~File() { this->close(); }

FileHandler::File::operator FILE*() const { return this->_FILE; }

FileHandler::File::operator bool() const { return this->_FILE != nullptr; }

void FileHandler::File::close(void)
{
    if(_FILE && _FILE != stdin && _FILE != stdout && _FILE != stderr)
        std::fclose(_FILE);
    _FILE = nullptr;
}

FileHandler::File FileHandler::OpenDataSink(const char* path)
{
    if(!path)
    {
        Logger::Info("Using stdout as output data sink");
        return File(stdout);
    }

    //  --- Check if a file at the path exists ---
    bool exists = std::filesystem::exists(path);

    //  --- Check that the path is a file ---
    if(exists && !std::filesystem::is_regular_file(path))
    {
        Logger::Error("The provided file is not a standard file");
        return File();
    }

    //  --- Open file output sink ---
    FILE* file = fopen(path, "wb");
    if(!file)
    {
        Logger::Error("The provided file could not be opened for writing");
        return File();
    }

    Logger::Info("Opened output data sink '%s'", path);
    return File(file);
}

bool FileHandler::ForEachBinFile(const char* dir_path, const std::function<bool(const char*)>& callback)
{
    if(!dir_path || !callback)
    {
        Logger::Error("Invalid input directory or callback provided");
        return false;
    }

    std::filesystem::path path(dir_path);

    if(!std::filesystem::exists(path))
    {
        Logger::Error("Input directory does not exist");
        return false;
    }

    if(!std::filesystem::is_directory(path))
    {
        Logger::Error("Input path does not point to a directory");
        return false;
    }

    std::size_t count = 0;
    for(const auto& entry : std::filesystem::directory_iterator(path))
    {
        if(entry.path().extension() != ".bin")
            continue;

        const std::string file_path = entry.path().string();
        if(!callback(file_path.c_str()))
        {
            Logger::Info("Stopped .bin iteration early after %zu file(s) in '%s'", count, dir_path);
            return false;
        }
        count++;
    }

    Logger::Info("Discovered %zu .bin file(s) in '%s'", count, dir_path);
    return true;
}

bool FileHandler::ReadBinaryImage(std::vector<u8>& dest, const char* path)
{
    if(!path)
    {
        Logger::Error("No input file path provided");
        return false;
    }

    if(!std::filesystem::exists(path))
    {
        Logger::Error("The provided file could not be opened");
        return false;
    }

    //  --- Check that the path is a file ---
    if(!std::filesystem::is_regular_file(path))
    {
        Logger::Error("The provided file is not a standard file");
        return false;
    }

    //  --- Get the size of the file ---
    const auto size = std::filesystem::file_size(path);

    //  --- Open the file for reading ---
    std::ifstream file(path, std::ios::binary);

    //  --- Ensure the file opened correctly ---
    if(!file)
    {
        Logger::Error("The provided file could not be opened");
        return false;
    }

    //  --- Read data into buffer ---
    dest.resize(size);
    file.read(reinterpret_cast<char*>(dest.data()), size);
    Logger::Info("Read %zu byte(s) from '%s'", static_cast<std::size_t>(size), path);

    return true;
}
