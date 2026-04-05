#pragma once

#include "datatypes.hpp"

/**
 * This file stores all of the custom data structure definitions for easy access later
**/

template<typename T>
struct View
{
    const usize len;
    T* data;
};

//  --- Ensure data structures are of expected size ---