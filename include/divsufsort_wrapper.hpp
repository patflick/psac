/**
 * @file    difsufsort_wrapper.hpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Wraps the C function calls of libdivsufsort with templates calls
 *          and namespaces. This allows clean usage for comparisons.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 *
 * TODO add Licence
 */
#ifndef DSS_WRAPPER_HPP
#define DSS_WRAPPER_HPP


#include <string>
#include <vector>
#include <iterator>
#include <limits>
#include <stdexcept>

// include divsufsort files
#include <divsufsort.h>
#include <divsufsort64.h>


// C++ interface for libdivsufsort
namespace dss
{

#if 0
// construction
void sa_construction(const std::string& str, std::vector<saidx_t>& SA)
{
    saidx_t n = str.size();
    SA.resize(n);
    divsufsort(reinterpret_cast<const sauchar_t*>(str.data()), &SA[0], n);
}
// construction 64 bits
void sa_construction(const std::string& str, std::vector<saidx64_t>& SA)
{
    saidx64_t n = str.size();
    SA.resize(n);
    divsufsort64(reinterpret_cast<const sauchar_t*>(str.data()), &SA[0], n);
}

// check correctness
bool sa_check(const std::string& str, const std::vector<saidx_t>& SA)
{
    saidx_t n = str.size();
    const sauchar_t* T = reinterpret_cast<const sauchar_t*>(str.data());
    return sufcheck(T, &SA[0], n, 1) == 0;
}

// check correctness 64 bits
bool sa_check(const std::string& str, const std::vector<saidx64_t>& SA)
{
    saidx64_t n = str.size();
    const sauchar_t* T = reinterpret_cast<const sauchar_t*>(str.data());
    return sufcheck64(T, &SA[0], n, 1) == 0;
}
#endif

template <typename InputIterator, typename T>
void construct(InputIterator begin, InputIterator end, std::vector<T>& SA)
{
    typedef typename std::iterator_traits<InputIterator>::value_type char_t;
    if (sizeof(char_t) != 1)
        throw std::runtime_error("Input must be a char type");
    std::size_t n = std::distance(begin, end);
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

template <typename InputIterator, typename T>
bool check(InputIterator begin, InputIterator end, const std::vector<T>& SA)
{
    typedef typename std::iterator_traits<InputIterator>::value_type char_t;
    if (sizeof(char_t) != 1)
        throw std::runtime_error("Input must be a char type");
    std::size_t n = std::distance(begin, end);
    if (sizeof(T) == sizeof(saidx_t)) {
        if (n >= std::numeric_limits<saidx_t>::max())
            throw std::runtime_error("Input size is too large for 32bit indexing.");
        return sufcheck(reinterpret_cast<const sauchar_t*>(str.data()), reinterpret_cast<saidx_t*>(&SA[0]), n, 1) == 0;
    } else if (sizeof(T) == sizeof(saidx64_t)) {
        return sufcheck64(reinterpret_cast<const sauchar_t*>(str.data()), reinterpret_cast<saidx64_t*>(&SA[0]), n, 1) == 0;
    } else {
        throw std::runtime_error("Unsupported datatype of Suffix Array.");
    }
}

} // namespace divsufsort (dss)

#endif // DSS_WRAPPER_HPP
