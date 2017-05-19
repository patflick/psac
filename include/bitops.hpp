/*
 * Copyright 2015 Georgia Institute of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file    bitops.hpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements common bit operations in a platform independent manner.
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
inline unsigned int log2_64(uint64_t value) {
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

inline unsigned int leading_zeros_64(uint64_t x) {
    if (x == 0)
        return 64;
    unsigned int log2 = log2_64(x);
    return 64 - log2 - 1;
}

inline unsigned int leading_zeros_32(uint32_t x) {
    // TODO implement faster version for 32 bits
    return leading_zeros_64(static_cast<uint64_t>(x)) - 32;
}

template<typename T>
inline unsigned int leading_zeros(T x) {
    if (sizeof(T)*8 == 64)
        return leading_zeros_64(x);
    else if (sizeof(T)*8 == 32)
        return leading_zeros_32(x);
    else
    {
        // TODO: implement other sizes
        assert(false);
        return -1;
    }
}

unsigned int reference_trailing_zeros(uint64_t x) {
    unsigned int n = 0;
    while (!(x & 0x1)) {
        x >>= 1;
        ++n;
    }
    return n;
}

unsigned int reference_leading_zeros(uint64_t x) {
    unsigned int n = 0;
    while (!(x & (static_cast<uint64_t>(0x1) << (sizeof(x)*8-1))))
    {
        ++n;
        x <<= 1;
    }
    return n;
}


unsigned int reference_ceillog2(unsigned int x) {
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


template <typename IntType>
inline unsigned int floorlog2(IntType n) {
    //return log2_64(n);
    return ((unsigned) (8*sizeof(unsigned long long) - __builtin_clzll((n)) - 1));
}

template <typename IntType>
inline unsigned int ceillog2(IntType n) {
    unsigned int log_floor = floorlog2(n);
    // add one if not power of 2
    return log_floor + (((n&(n-1)) != 0) ? 1 : 0);
}

/**
 * @brief   Returns the number identical characters of two strings in k-mer
 *          compressed bit representation with `bits_per_char` bits per
 *          character in the word of type `T`.
 *
 * @tparam T                The type of the values (an integer type).
 * @param x                 The first value to compare.
 * @param y                 The second value to compare.
 * @param k                 The total number of characters stored in
 *                          one word of type `T`.
 * @param bits_per_char     The number of bits per character in the k-mer
 *                          representation of `x` and `y`.
 * @return  The longest common prefix, i.e., the number of sequential characters
 *          equal in the two values `x` and `y`.
 */
template <typename T>
unsigned int lcp_bitwise(T x, T y, unsigned int k, unsigned int bits_per_char) {
    if (x == y)
        return k;
    // XOR the two values and then find the MSB that isn't zero (since
    // the k-mer strings start (have first character) at MSB)
    T z = x ^ y;
    // get leading zeros
    unsigned int lz = leading_zeros(z);

    // get leading zeros in the k-mer representation
    unsigned int kmer_lz = lz - (sizeof(T)*8 - k*bits_per_char);
    unsigned int lcp = kmer_lz / bits_per_char;
    return lcp;
}

/**
 * @brief   Returns the number identical characters of two strings in k-mer
 *          compressed bit representation with `bits_per_char` bits per
 *          character in the word of type `T`. This version does not count
 *          trailing `0`s towards the LCP.
 *
 * @tparam T                The type of the values (an integer type).
 * @param x                 The first value to compare.
 * @param y                 The second value to compare.
 * @param k                 The total number of characters stored in
 *                          one word of type `T`.
 * @param bits_per_char     The number of bits per character in the k-mer
 *                          representation of `x` and `y`.
 * @return  The longest common prefix, i.e., the number of sequential characters
 *          equal in the two values `x` and `y`, ignoring the trailing 0s.
 */
template <typename T>
unsigned int lcp_bitwise_no0(T x, T y, unsigned int k, unsigned int bits_per_char) {
    if (x == y)
        return k;
    // XOR the two values and then find the MSB that isn't zero (since
    // the k-mer strings start (have first character) at MSB)
    T z = x ^ y;
    unsigned int lz = leading_zeros(z);
    // if x==y, then return the trailing zeroes of both combined, else 0
    unsigned int tz = trailing_zeros((!(x == y)) & (x | y));

    // get leading zeros in the k-mer representation
    unsigned int kmer_lz = lz - (sizeof(T)*8 - k*bits_per_char);
    unsigned int lcp = kmer_lz/bits_per_char - tz/bits_per_char;
    return lcp;
}


#endif // BITOPS_HPP

