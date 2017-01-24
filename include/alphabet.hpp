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
#include <mxx/comm.hpp>

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


template <typename index_t, typename InputIterator>
std::vector<index_t> alphabet_histogram(InputIterator begin, InputIterator end, const mxx::comm& comm) {
    static_assert(std::is_same<typename std::iterator_traits<InputIterator>::value_type, char>::value, "Iterator must be of value type `char`.");
    // get local histogram of alphabet characters
    std::vector<index_t> hist = get_histogram<index_t>(begin, end, 256);
    std::vector<index_t> out_hist = mxx::allreduce(hist, comm);
    return out_hist;
}

template<typename CharType>
class alphabet {
    static_assert(sizeof(CharType) == 1, "Dynamic alphabet supports only `char` (1-byte) alphabets");
public:

    using char_type = CharType;
    using uchar_type = typename std::make_unsigned<char_type>::type;
    static constexpr uchar_type max_uchar = std::numeric_limits<uchar_type>::max();

    /// executes the given function on every char in the alphabet
    /// (eg, for iterating through characters used within the alphabet)
    template <typename Func>
    inline void for_each_char(Func f) {
        for (uchar_type c = 0; ; ++c) {
            if (chars_used[c]) {
                f(c);
            }
            if (c == max_uchar)
                break;
        }
    }

private:


    std::vector<bool> chars_used;

    /// maps each unsigned char to a new integer using at most log(sigma+1) bits
    std::vector<uint16_t> mapping_table;

    unsigned int m_sigma;
    unsigned int m_bits_per_char;


    inline void init_mapping_table() {
        mapping_table.resize(max_uchar+1, 0);
        uint16_t mapped = 1; // start with 1 to include special char '\0' ('$')
        for_each_char([&](uchar_type c) {
            mapping_table[c] = mapped;
            ++mapped;
        });
    }

    inline void init_sizes() {
        unsigned int num_chars = 0;
        for_each_char([&](uchar_type) {
            ++num_chars;
        });
        m_sigma = num_chars;
        m_bits_per_char = ceillog2(m_sigma+1);
    }

    template <typename count_type>
    alphabet(const std::vector<count_type>& hist) {
        assert(hist.size() == max_uchar+1);
        chars_used = std::vector<bool>(hist.size(), false);
        for (unsigned int i = 0; i < hist.size(); ++i) {
            if (hist[i])
                chars_used[i] = true;
        }
        init_mapping_table();
        init_sizes();
    }

public:
    /// default constructor and assignment operators
    alphabet() = default;
    alphabet(const alphabet&) = default;
    alphabet(alphabet&&) = default;
    alphabet& operator=(const alphabet&) = default;
    alphabet& operator=(alphabet&&) = default;

    template <typename index_t>
    static alphabet from_hist(const std::vector<index_t>& hist) {
        alphabet a(hist);
        return a;
    }

    template <typename Iterator>
    static alphabet from_sequence(Iterator begin, Iterator end, const mxx::comm& comm = mxx::comm()) {
        static_assert(std::is_same<char_type, typename std::iterator_traits<Iterator>::value_type>::value, "Character type of alphabet must match the value type of input sequence");
        // get histogram
        std::vector<size_t> alphabet_hist = alphabet_histogram<size_t>(begin, end, comm);
        return alphabet::from_hist(alphabet_hist);
    }

    inline unsigned int sigma() const {
        return m_sigma;
    }

    inline unsigned int bits_per_char() const {
        return m_bits_per_char;
    }

    template <typename word_type>
    inline unsigned int chars_per_word() const {
        unsigned int bits_per_word = sizeof(word_type)*8;
        // TODO: this is currently a "work-around": if the type is signed, we
        //       can't use the msb, thus we need to subtract one
        if (std::is_signed<word_type>::value)
            --bits_per_word;
        return bits_per_word/bits_per_char();
    }

    // TODO: datatypes?
    inline uint16_t encode(char_type c) const {
        uint32_t index = static_cast<uint32_t>(c);
        assert(0 <= index && index < mapping_table.size());
        return mapping_table[index];
    }

    inline std::vector<char_type> unique_chars() {
        std::vector<char_type> result;
        for_each_char([&](uchar_type c) {
            result.push_back(c);
        });
        return result;
    }
};


#endif // ALPHABET_HPP
