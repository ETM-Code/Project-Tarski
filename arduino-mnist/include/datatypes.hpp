#pragma once

#include <stdint.h>

/**
 * This file stores all of the custom data type definitions for easy access later
**/

using u8 = unsigned char;
using u16 = unsigned short;
using u32 = unsigned long;
using usize = unsigned short;

using i8 = signed char;
using i16 = signed short;
using i32 = signed long;
using isize = signed short;

using f32 = float;

using time_t = unsigned long;

//  --- Ensure data types are of expected size ---

static_assert(sizeof(u8) == 1);
static_assert(sizeof(u16) == 2);
static_assert(sizeof(u32) == 4);
static_assert(sizeof(usize) == 2);

static_assert(sizeof(i8) == 1);
static_assert(sizeof(i16) == 2);
static_assert(sizeof(i32) == 4);
static_assert(sizeof(isize) == 2);

static_assert(sizeof(f32) == 4);

static_assert(sizeof(time_t) == 4);