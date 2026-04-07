#include <chrono>
#include <thread>
#include "serial.hpp"

// This should change based on the platform
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <termios.h>

//  --- Private handle struct implementation ---
struct Serial::Impl
{
    int fd = -1;

    ~Impl()
    {
        if(fd != -1) ::close(fd);
    }
};

Serial::Serial(const char* path, Baudrate baudrate, u32 timeout_ms)
{
    // Set internal variables to passed in values
    _timeout_ms = timeout_ms;
    _error = Error::NO_ERROR;
 
    // Set baudrate
    switch(baudrate)
    {
        case Baudrate::BR9600:
            _baudrate = B9600;
            break;
    }

    // Allocate memory for the serial port interface handle
    _handle = new Impl;
    
    // Open the serial bus interface
    _handle->fd = open(path, O_RDWR | O_NDELAY | O_NOCTTY);
    if(_handle->fd < 0)
    {
        delete _handle;
        _handle = nullptr;
        _error = Error::INVALID_PATH;
        return;
    }

    // Wait for the port to be opened
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Set config to default values
    struct termios options {};
    options.c_cflag = CS8 | CLOCAL | CREAD;
    options.c_iflag = IGNPAR;
    options.c_oflag = 0;
    options.c_lflag = 0;

    // Configure input/output speed in a portable way (Linux/macOS).
    cfsetispeed(&options, _baudrate);
    cfsetospeed(&options, _baudrate);

    tcflush(_handle->fd, TCIFLUSH);
    int err = tcsetattr(_handle->fd, TCSANOW, &options);

    if(err)
    {
        delete _handle;
        _handle = nullptr;
        _error = Error::INVALID_CONFIG;
        return;
    }
}

Serial::~Serial()
{
    if(_handle) delete _handle;
}

Serial::Serial(Serial&& other) noexcept
    : _handle(other._handle), _timeout_ms(other._timeout_ms), _baudrate(other._baudrate), _error(other._error)
{
    other._handle = nullptr;
}

Serial& Serial::operator=(Serial&& other) noexcept
{
    if(this != &other)
    {
        if(_handle) delete _handle;
        _handle = other._handle;
        _timeout_ms = other._timeout_ms;
        _baudrate = other._baudrate;
        _error = other._error;
        other._handle = nullptr;
    }
    return *this;
}

bool Serial::available(void)
{
    return _error == Error::NO_ERROR;
}

Serial::Error Serial::getError(void) { return _error; }

u32 Serial::dataAvailable(void)
{
    int bytes_present = 0;
    if(ioctl(_handle->fd, FIONREAD, &bytes_present) == -1)
        _error = Error::CTRL_ERROR;

    // Ensure bytes available are not negative
    if(bytes_present < 0)
    {
        _error = Error::CTRL_ERROR;
        bytes_present = 0;
    }

    return static_cast<u32>(bytes_present);
}

[[nodiscard]] bool Serial::awaitData(void) { return awaitData(1); }

[[nodiscard]] bool Serial::awaitData(u32 count)
{
    using clock = std::chrono::steady_clock;
    using timepoint = clock::time_point;
    
    // Setup timeout variable
    const timepoint timeout_time = clock::now() + std::chrono::milliseconds(_timeout_ms);

    // Await data until timeout
    while(clock::now() < timeout_time)
    {
        if(this->dataAvailable() >= count) return true;

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    return false;                                       // Connection timed out
}

void Serial::writeByte(u8 data)
{
    if(write(_handle->fd, &data, 1) == -1)
        _error = Error::WRITE_ERROR;
}

void Serial::writeBytes(const u8* src, u32 count)
{
    if(write(_handle->fd, src, count) == -1)
        _error = Error::WRITE_ERROR;
}

u8 Serial::readByte(void)
{
    u8 ret;
    this->readBytes(&ret, 1);
    return ret;
}

u32 Serial::readBytes(u8* dest, u32 count)
{
    int res = read(_handle->fd, dest, count);

    // Ensure bytes read is non-negative
    if(res < 0)
    {
        _error = Error::READ_ERROR;
        return 0;
    }

    return static_cast<u32>(res);
}
