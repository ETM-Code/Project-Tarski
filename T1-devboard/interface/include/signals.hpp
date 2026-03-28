#pragma once

/**
 * This file defines the different signals that can be sent and received over the serial interface.
 * These are 1 byte ASCII (control & printable) characters used to interact with the Arduino.
**/

#define PORT_TIMEOUT    -1              // Used when communication on the port times out
#define PORT_MSG        0x02            // Used when the device wants to send a printable message
#define PORT_MSG_END    0x03            // Used when the device wants to stop sending a printable message
#define PORT_TRN_END    0x04            // Used to signal end of transmission
#define PORT_SIG        0x05            // Used to request the device signature (version number)
#define PORT_ACK        0x06            // Used for acknowledging data received
#define PORT_NAK        0x15            // Used for acknowledging data not received or error
#define PORT_READ_OUT   'O'             // Used to request the current output state from the device
#define PORT_LOAD_SYN   'S'             // Used to load synapse weight data into the device
#define PORT_LOAD_DAC   'D'             // Used to load DAC values into the device
#define PORT_READ_MEAS  'M'             // Used to request measurement data from the device
#define PORT_SET_FLAG   'F'             // Used to set a single device flag
#define PORT_UNSET_FLAG 'U'             // Used to unset a single device flag
#define PORT_TGL_FLAG   'T'             // Used to toggle a single device flag
#define PORT_PROG_DAC   'P'             // Used to program MCP4728 I2C address
