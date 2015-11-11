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
 * @file    difsufsort_wrapper.hpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Wraps the C function calls of libdivsufsort with C++ templated
 *          calls and into a namespace. Requires linking against libdivsufsort.
 */
#ifndef DSS_WRAPPER_HPP
#define DSS_WRAPPER_HPP


#include <string>
#include <vector>
#include <iterator>
#include <limits>
#include <stdexcept>

// include divsufsort files (both 32 and 64 bits)
#include <divsufsort.h>
#include <divsufsort64.h>


/// C++ interface for libdivsufsort
namespace dss
{

/**
 * @brief   Constuct the suffix array for the given character range using
 *          libdivsufsort.
 *
 * @tparam InputIterator    An input iterator with value type char
 * @tparam T                An index type for the suffix array. Either a 32bit
 *                          or 64 bit integer.
 * @param begin             The `begin` iterator for the input string.
 * @param end               The `end` iterator for the input string.
 * @param SA                The suffix array (as std::vector).
 */
template <typename InputIterator, typename T>
void construct(InputIterator begin, InputIterator end, std::vector<T>& SA)
{
    typedef typename std::iterator_traits<InputIterator>::value_type char_t;
    if (sizeof(char_t) != 1)
        throw std::runtime_error("Input must be a char type");
    std::size_t n = std::distance(begin, end);
    if (SA.size() != n)
        SA.resize(n);
    if (sizeof(T) == sizeof(saidx_t)) {
        if (n >= std::numeric_limits<saidx_t>::max())
            throw std::runtime_error("Input size is too large for 32bit indexing.");
        divsufsort(reinterpret_cast<const sauchar_t*>(&(*begin)), reinterpret_cast<saidx_t*>(&SA[0]), n);
    } else if (sizeof(T) == sizeof(saidx64_t)) {
        divsufsort64(reinterpret_cast<const sauchar_t*>(&(*begin)), reinterpret_cast<saidx64_t*>(&SA[0]), n);
    } else {
        throw std::runtime_error("Unsupported datatype of Suffix Array.");
    }
}

/**
 * @brief   Checks whether the given suffix array is correct given the string.
 *
 *  This uses libdivsufsort's `sufcheck()` function to check the suffix array
 *  sequentially.
 *
 * @tparam InputIterator    An input iterator with value type char
 * @tparam T                An index type for the suffix array. Either a 32bit
 *                          or 64 bit integer.
 * @param begin             The `begin` iterator for the input string.
 * @param end               The `end` iterator for the input string.
 * @param SA                The suffix array (as std::vector).
 *
 * @return  True, if the given suffix array is correct given the string. False
 *          otherwise.
 */
template <typename InputIterator, typename T>
bool check(InputIterator begin, InputIterator end, const std::vector<T>& SA)
{
    typedef typename std::iterator_traits<InputIterator>::value_type char_t;
    if (sizeof(char_t) != 1)
        throw std::runtime_error("Input must be a char type");
    std::size_t n = std::distance(begin, end);
    if (SA.size() != n)
        return false;
    if (sizeof(T) == sizeof(saidx_t)) {
        if (n >= std::numeric_limits<saidx_t>::max())
            throw std::runtime_error("Input size is too large for 32bit indexing.");
        return sufcheck(reinterpret_cast<const sauchar_t*>(&(*begin)),
                        reinterpret_cast<const saidx_t*>(&SA[0]), n, 1) == 0;
    } else if (sizeof(T) == sizeof(saidx64_t)) {
        return sufcheck64(reinterpret_cast<const sauchar_t*>(&(*begin)),
                          reinterpret_cast<const saidx64_t*>(&SA[0]), n, 1) == 0;
    } else {
        throw std::runtime_error("Unsupported datatype of Suffix Array.");
    }
}

} // namespace divsufsort (dss)

#endif // DSS_WRAPPER_HPP
