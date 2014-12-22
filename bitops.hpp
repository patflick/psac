/**
 * @file    bitops.hpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements common bit operations in a platform independent manner.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 *
 * TODO add Licence
 */

#ifndef BITOPS_HPP
#define BITOPS_HPP

#include <cstdint>
#include <assert.h>

/**
 * bitScanForward
 * @author Martin Läuter (1997)
 *         Charles E. Leiserson
 *         Harald Prokop
 *         Keith H. Randall
 * "Using de Bruijn Sequences to Index a 1 in a Computer Word"
 * @param x     Input value (64 bit integer).
 * @precondition x != 0
 * @return      The number of trailing zeros in the integer.
 */
inline unsigned int trailing_zeros(uint64_t x) {
    static const unsigned int index64[64] = {
        0,  1,  48,  2, 57, 49, 28,  3,
        61, 58, 50, 42, 38, 29, 17,  4,
        62, 55, 59, 36, 53, 51, 43, 22,
        45, 39, 33, 30, 24, 18, 12,  5,
        63, 47, 56, 27, 60, 41, 37, 16,
        54, 35, 52, 21, 44, 32, 23, 11,
        46, 26, 40, 15, 34, 20, 31, 10,
        25, 14, 19,  9, 13,  8,  7,  6
    };
    const uint64_t debruijn64 = 0x03f79d71b4cb0a89ull;
    assert (x != 0);
    return index64[((x & -x) * debruijn64) >> 58];
}


/**
 * Fast integer log base 2 for 64 bit integers.
 * @param x         Input value.
 * @precondition    x != 0
 * @return  The log base 2.
 */
// source: https://stackoverflow.com/questions/11376288/fast-computing-of-log2-for-64-bit-integers
inline unsigned int log2_64(uint64_t value)
{
    static const unsigned int tab64[64] = {
        63,  0, 58,  1, 59, 47, 53,  2,
        60, 39, 48, 27, 54, 33, 42,  3,
        61, 51, 37, 40, 49, 18, 28, 20,
        55, 30, 34, 11, 43, 14, 22,  4,
        62, 57, 46, 52, 38, 26, 32, 41,
        50, 36, 17, 19, 29, 10, 13, 21,
        56, 45, 25, 31, 35, 16,  9, 12,
        44, 24, 15,  8, 23,  7,  6,  5
    };
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    value |= value >> 32;
    return tab64[((uint64_t)((value - (value >> 1))*0x07EDD5E59A4E28C2ull)) >> 58];
}

inline unsigned int leading_zeros(uint64_t x)
{
    if (x == 0)
        return 64;
    unsigned int log2 = log2_64(x);
    return 64 - log2 - 1;
}

unsigned int reference_trailing_zeros(uint64_t x)
{
    unsigned int n = 0;
    while (!(x & 0x1)) {
        x >>= 1;
        ++n;
    }
    return n;
}

unsigned int reference_leading_zeros(uint64_t x)
{
    unsigned int n = 0;
    while (!(x & (static_cast<uint64_t>(0x1) << (sizeof(x)*8-1))))
    {
        ++n;
        x <<= 1;
    }
    return n;
}

// TODO: make this faster!
unsigned int ceillog2(unsigned int x)
{
    unsigned int log_floor = 0;
    unsigned int n = x;
    for (;n != 0; n >>= 1)
    {
        ++log_floor;
    }
    --log_floor;
    // add one if not power of 2
    return log_floor + (((x&(x-1)) != 0) ? 1 : 0);
}

#endif // BITOPS_HPP

