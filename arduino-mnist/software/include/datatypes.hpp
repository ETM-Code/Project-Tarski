#pragma once

#include <cstdint>

/**
 * Central aliases for fixed-width integer types used across the project.
 */

using u8  = unsigned char;
using u16 = unsigned short;
using u32 = unsigned int;

using i8  = signed char;
using i16 = signed short;
using i32 = signed int;

using an_time_t = unsigned int;

//  --- Ensure data types are of expected size ---

static_assert(sizeof(u8)  == 1);
static_assert(sizeof(u16) == 2);
static_assert(sizeof(u32) == 4);

static_assert(sizeof(i8)  == 1);
static_assert(sizeof(i16) == 2);
static_assert(sizeof(i32) == 4);

static_assert(sizeof(an_time_t) == 4);
