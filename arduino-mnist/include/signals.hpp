#pragma once

/**
 * This file defines the different signals that can be sent and received over the serial interface.
 * These are 1 byte ASCII (control & printable) characters used to interact with the the arduino.
**/

#define PORT_TIMEOUT    -1              // Used when communication on the port times-out
#define PORT_MSG        0x02            // Used when the device want to send a printable message
#define PORT_MSG_END    0x03            // Used when the device want to stop sending a printable message
#define PORT_TRN_END    0x04            // Used to signal end of transmission
#define PORT_SIG        0x05            // Used to request for device signature (version number) 
#define PORT_ACK        0x06            // Used for acknowledging data received
#define PORT_NAK        0x15            // Used for acknowledging data not recieved or error
#define PORT_LOAD       'L'             // Used to load a data sample into the arduino's memory
#define PORT_INFER      'I'             // Signals to the arduino to run the inference model
// #define PORT_RDY        'R'
// #define PORT_ERR        'E'
