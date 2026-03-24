#pragma once

#include "datatypes.hpp"

/**
 * This file stores custom data structure definitions used across the project.
**/

template<typename T>
struct View
{
    // Number of elements available in `data`.
    const usize len;
    // Non-owning pointer to contiguous data.
    T* data;
};
