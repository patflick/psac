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
#ifndef ALPHABET_HPP
#define ALPHABET_HPP

#include <vector>
#include <string>
#include <algorithm>

#include "bitops.hpp"

/*****************************
 *  create random DNA input  *
 *****************************/
inline char rand_dna_char() {
    const static char DNA[4] = {'A', 'C', 'G', 'T'};
    return DNA[rand() % 4];
}

std::string rand_dna(std::size_t size, int seed) {
    srand(1337*seed);
    std::string str;
    str.resize(size, ' ');
    for (std::size_t i = 0; i < size; ++i) {
        str[i] = rand_dna_char();
    }
    return str;
}

template<typename T, typename Iterator>
std::vector<T> get_histogram(Iterator begin, Iterator end, std::size_t size = 0) {
    if (size == 0)
        size = static_cast<std::size_t>(*std::max_element(begin, end)) + 1;
    std::vector<T> hist(size);

    while (begin != end) {
        char c = *begin;
        std::size_t s = (unsigned char)c;
        ++hist[s];
        ++begin;
    }

    return hist;
}

template <typename index_t>
std::vector<uint16_t> alphabet_mapping_tbl(const std::vector<index_t>& global_hist) {
    std::vector<uint16_t> mapping(256, 0);

    uint16_t next = static_cast<uint16_t>(1);
    for (std::size_t c = 0; c < 256; ++c) {
        if (global_hist[c] != 0) {
            mapping[c] = next;
            ++next;
        }
    }
    return mapping;
}

template <typename index_t>
unsigned int alphabet_unique_chars(const std::vector<index_t>& global_hist) {
    unsigned int unique_count = 0;
    for (std::size_t c = 0; c < 256; ++c) {
        if (global_hist[c] != 0) {
            ++unique_count;
        }
    }
    return unique_count;
}

template <typename index_t>
std::vector<char> alphabet_unique_char_vec(const std::vector<index_t>& global_hist) {
    std::vector<char> v;
    for (std::size_t c = 0; c < 256; ++c) {
        if (global_hist[c] != 0) {
            v.push_back(c);
        }
    }
    return v;
}


unsigned int alphabet_bits_per_char(unsigned int sigma) {
    // since we have to account for the `0` character, we use ceil(log(unique_chars + 1))
    return ceillog2(sigma+1);
}

template<typename word_t>
unsigned int alphabet_chars_per_word(unsigned int bits_per_char) {
    unsigned int bits_per_word = sizeof(word_t)*8;
    // TODO: this is currently a "work-around": if the type is signed, we
    //       can't use the msb, thus we need to subtract one
    if (std::is_signed<word_t>::value)
        --bits_per_word;
    return bits_per_word/bits_per_char;
}

#endif // ALPHABET_HPP
