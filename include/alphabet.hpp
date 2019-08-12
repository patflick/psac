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
#include <iostream>
#include <fstream>
#include <mxx/comm.hpp>
#include <mxx/reduction.hpp>

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

/// scans the input sequence and updates the given frequency histogram of characters
template<typename T, typename Iterator>
void update_histogram(Iterator begin, Iterator end, std::vector<T>& hist) {
    using char_type = typename std::iterator_traits<Iterator>::value_type;
    using uchar_type = typename std::make_unsigned<char_type>::type;
    while (begin != end) {
        char_type c = *begin;
        uchar_type uc = c;
        size_t s = uc;
        ++hist[s];
        ++begin;
    }
}

/// Returns a frequency histogram of characters from a given input sequence
template<typename T, typename Iterator>
std::vector<T> get_histogram(Iterator begin, Iterator end, std::size_t size = 0) {
    if (size == 0)
        size = static_cast<std::size_t>(*std::max_element(begin, end)) + 1;
    std::vector<T> hist(size);
    update_histogram(begin, end, hist);
    return hist;
}

/// Returns a frequency histogram of characters from a given distributed stringset (collective call)
template <typename index_t, typename StringSet>
std::vector<index_t> alphabet_histogram(const StringSet& ss, const mxx::comm& comm) {
    std::vector<index_t> hist(256, 0);
    for (size_t i = 0; i < ss.sizes.size(); ++i) {
        // add all local characters to the histogram
        update_histogram(ss.str_begins[i], ss.str_begins[i] + ss.sizes[i], hist);
    }
    std::vector<index_t> out_hist = mxx::allreduce(hist, comm);
    return out_hist;
}

/// Returns a frequency histogram of characters from a given input sequence.
template <typename index_t, typename InputIterator>
std::vector<index_t> alphabet_histogram(InputIterator begin, InputIterator end) {
    static_assert(std::is_same<typename std::iterator_traits<InputIterator>::value_type, char>::value, "Iterator must be of value type `char`.");
    // get local histogram of alphabet characters
    std::vector<index_t> hist = get_histogram<index_t>(begin, end, 256);
    return hist;
}

/// Returns a frequency histogram of characters from a given distributed input sequence (collective call)
template <typename index_t, typename InputIterator>
std::vector<index_t> alphabet_histogram(InputIterator begin, InputIterator end, const mxx::comm& comm) {
    static_assert(std::is_same<typename std::iterator_traits<InputIterator>::value_type, char>::value, "Iterator must be of value type `char`.");
    // get local histogram of alphabet characters
    std::vector<index_t> hist = get_histogram<index_t>(begin, end, 256);
    std::vector<index_t> out_hist = mxx::allreduce(hist, comm);
    return out_hist;
}


/**
 * @brief Class representing an alphabet with character size <= 2 bytes.
 */
template<typename CharType>
class alphabet {
    static_assert(sizeof(CharType) <= 2, "Dynamic alphabet supports at most 2-byte (16bits) alphabets");
public:
    /// the character type of the alphabet
    using char_type = CharType;
    /// unsigned version of `char_type`
    using uchar_type = typename std::make_unsigned<char_type>::type;
    /// limit of `uchar_type`, defines size of mapping table
    static constexpr uchar_type max_uchar = std::numeric_limits<uchar_type>::max();

    /// executes the given function on every char in the alphabet
    /// (eg, for iterating through characters used within the alphabet)
    template <typename Func>
    inline void for_each_char(Func f) const {
        for (uchar_type c = 0; ; ++c) {
            if (chars_used[c]) {
                f(static_cast<char_type>(c));
            }
            if (c == max_uchar)
                break;
        }
    }

private:

    /// bitmap of characters in the alphabet
    std::vector<bool> chars_used;

    /// maps each unsigned char to a new integer using at most log(sigma+1) bits
    std::vector<uchar_type> mapping_table;
    /// maps the encoded characters back to their original
    std::vector<char_type> inverse_mapping;

    /// alphabet size (number of unique characters excl. '\0')
    unsigned int m_sigma;
    /// number of bits each character can be represented/encoded as
    unsigned int m_bits_per_char;


    /// initializes the alphabet size and bit sizes
    void init_sizes() {
        unsigned int num_chars = 0;
        for_each_char([&](uchar_type) {
            ++num_chars;
        });
        m_sigma = num_chars;
        m_bits_per_char = ceillog2(m_sigma+1);
    }

    /// initializes the alphabet mapping table
    void init_mapping_table() {
        mapping_table.resize(max_uchar+1, 0);
        uint16_t mapped = 1; // start with 1 to include special char '\0' ('$')
        for_each_char([&](uchar_type c) {
            mapping_table[c] = mapped;
            ++mapped;
        });
    }

    /// initializes the inverse mapping table
    void init_inverse_mapping() {
        inverse_mapping.push_back('\0');
        for_each_char([&](uchar_type c) {
            inverse_mapping.push_back(c);
        });
    }

    /// constructs the alphabet from a given character frequency histogram
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
        init_inverse_mapping();
    }

public:
    /// default constructor and assignment operators
    alphabet() = default;
    alphabet(const alphabet&) = default;
    alphabet(alphabet&&) = default;
    alphabet& operator=(const alphabet&) = default;
    alphabet& operator=(alphabet&&) = default;

    /// Constructs the alphabet from a given character frequency histogram
    template <typename index_t>
    static alphabet from_hist(const std::vector<index_t>& hist) {
        alphabet a(hist);
        return a;
    }

    /// Constructs the alphabet from a given sequence
    template <typename Iterator>
    static alphabet from_sequence(Iterator begin, Iterator end) {
        static_assert(std::is_same<char_type, typename std::iterator_traits<Iterator>::value_type>::value, "Character type of alphabet must match the value type of input sequence");
        std::vector<size_t> alphabet_hist = alphabet_histogram<size_t>(begin, end);
        return alphabet::from_hist(alphabet_hist);
    }

    /// Constructs the alphabet from a given distributed sequence (collective call)
    template <typename Iterator>
    static alphabet from_sequence(Iterator begin, Iterator end, const mxx::comm& comm) {
        static_assert(std::is_same<char_type, typename std::iterator_traits<Iterator>::value_type>::value, "Character type of alphabet must match the value type of input sequence");
        // get histogram
        std::vector<size_t> alphabet_hist = alphabet_histogram<size_t>(begin, end, comm);
        return alphabet::from_hist(alphabet_hist);
    }

    /// Constructs the alphabet from a given string
    static alphabet from_string(const std::basic_string<char_type>& str) {
        return alphabet::from_sequence(str.begin(), str.end());
    }

    /// Constructs the alphabet from a given distributed string (collective call)
    static alphabet from_string(const std::basic_string<char_type>& str, const mxx::comm& comm) {
        return alphabet::from_sequence(str.begin(), str.end(), comm);
    }

    /// Constructs the alphabet from a distributed stringset (collective call)
    template <typename StringSet>
    static alphabet from_stringset(const StringSet& ss, const mxx::comm& comm) {
        std::vector<size_t> alphabet_hist = alphabet_histogram<size_t>(ss, comm);
        return alphabet::from_hist(alphabet_hist);
    }

    /// Returns the number of unique characters in the alphabet (i.e. the
    //  alphabet size)
    inline unsigned int sigma() const {
        return m_sigma;
    }

    /// Returns the alphabet size (same as .sigma())
    inline unsigned int size() const {
        return m_sigma;
    }

    /// Returns the number of bits a single character can be represented by
    inline unsigned int bits_per_char() const {
        return m_bits_per_char;
    }

    /// Returns the number of characters that fit into the given word_type
    template <typename word_type>
    inline unsigned int chars_per_word() const {
        static_assert(sizeof(word_type) >= sizeof(char_type), "expecting the word_type to be larger than the char type");
        unsigned int bits_per_word = sizeof(word_type)*8;
        // if the type is signed, we can't use the msb, thus we need to subtract one
        if (std::is_signed<word_type>::value)
            --bits_per_word;
        return bits_per_word/bits_per_char();
    }

    /// Returns the unique characters in this alphabet
    inline std::vector<char_type> unique_chars() const {
        std::vector<char_type> result;
        for_each_char([&](uchar_type c) {
            result.push_back(c);
        });
        return result;
    }

    /// encodes the given character to use at most `bits_per_char` bits
    inline uchar_type encode(char_type c) const {
        uchar_type uc = c;
        assert(0 <= uc && uc < mapping_table.size());
        return mapping_table[uc];
    }

    /// Returns the decoded character from the encoded (compressed) character
    inline char_type decode(uchar_type c) const {
        return inverse_mapping[c];
    }

    /// equality check
    bool operator==(const alphabet& o) const {
        return o.chars_used == this->chars_used && o.m_sigma == this->m_sigma;
    }

    /// in-equality check
    bool operator!=(const alphabet& o) const {
        return !(*this == o);
    }

    /// write alphabet to file
    void write_to(std::ostream& os) const {
        for_each_char([&os](char_type c) {
            os.write(&c,sizeof(c));
        });
    }

    /// read alphabet from file
    static alphabet read_from(std::istream& is) {
        // read in everything
        std::basic_string<char_type> s;
        char_type c;
        while (is.read(&c,sizeof(char_type))) {
            s.push_back(c);
        }
        return from_string(s);
    }

    /// write alphabet to file
    void write(const std::string& filename) const {
        std::ofstream f(filename);
        write_to(f);
    }

    /// write alphabet to file (collective call)
    void write(const std::string& filename, const mxx::comm& comm) const {
        // only one processor writes the file
        if (comm.rank() == 0) {
            write(filename);
        }
    }

    /// read alphabet from file
    void read(const std::string& filename) {
        std::ifstream f(filename);
        *this = alphabet::read_from(f);
    }

    /// read alphabet from file (collective call)
    void read(const std::string& filename, const mxx::comm& comm) {
        std::basic_string<char_type> s;
        // only one processor reads the file and then broadcasts the alphabet
        if (comm.rank() == 0) {
            std::ifstream f(filename);
            char_type c;
            while (f.read(&c,sizeof(char_type))) {
                s.push_back(c);
            }
        }
        mxx::bcast(s, 0, comm);

        *this = alphabet::from_string(s);
    }
};

template <typename char_type>
std::ostream& operator<<(std::ostream& os, const alphabet<char_type>& a) {
     return os << "{sigma=" << a.sigma() << ", l=" << a.bits_per_char() << ", A=" << a.unique_chars() << "}";
}

template<typename IntType>
class int_alphabet {

public:
    // types
    using char_type = IntType;
    using uchar_type = typename std::make_unsigned<char_type>::type;

    // limits of the given int datatype
    static constexpr uchar_type UCHAR_MAXLIM = std::numeric_limits<uchar_type>::max();
    static constexpr char_type  MINLIM       = std::numeric_limits<char_type>::min();
    static constexpr char_type  MAXLIM       = std::numeric_limits<char_type>::max();

    template <typename CharType>
    friend std::ostream& operator<<(std::ostream&, const int_alphabet<CharType>&);


private:
    char_type min_char;
    char_type max_char;
    char_type offset;
    uchar_type m_sigma;
    unsigned int m_bits_per_char;


public:
    /**
     * @brief Construct the Integer alphabet given the range of valid values
     *        [min_char, max_char]. The range has to be at least by 1 smaller
     *        than the data type `char_type` valid range, since the SA construction
     *        reserves the 0 value for special use.
     *
     * @param min_char  Smallest possible value in alphabet
     * @param max_char  Largest possible value in alphabet
     */
    int_alphabet(char_type min_char, char_type max_char)
        : min_char(min_char),
          max_char(max_char),
          offset(-min_char + 1),
          m_sigma(max_char - min_char + 1) {
        if (min_char > max_char || (min_char == MINLIM && max_char == MAXLIM)) {
            // error, range is either empty or too large, throw argument exception
            throw std::runtime_error("the [min_char, max_char] value range is too large to be represented by the given type");
        }
        m_bits_per_char = ceillog2(m_sigma+1);
    }

    int_alphabet() : int_alphabet(MINLIM, MAXLIM - 1) {};


    /// default constructor and assignment operators
    int_alphabet(const int_alphabet&) = default;
    int_alphabet(int_alphabet&&) = default;
    int_alphabet& operator=(const int_alphabet&) = default;
    int_alphabet& operator=(int_alphabet&&) = default;

    template <typename Iterator>
    static std::pair<char_type,char_type> getminmax(Iterator begin, Iterator end) {
        static_assert(std::is_same<char_type, typename std::iterator_traits<Iterator>::value_type>::value,
                      "Character type of alphabet must match the value type of input sequence");
        char_type minval = std::numeric_limits<char_type>::max();
        char_type maxval = std::numeric_limits<char_type>::min();
        for (Iterator it = begin; it != end; ++it) {
            if (*it < minval) {
                minval = *it;
            }
            if (*it > maxval) {
                maxval = *it;
            }
        }
        return std::pair<char_type,char_type>(minval, maxval);
    }

    /**
     * @brief Creates alphabet from underlying sequence.
     *
     * This method scans the underlying sequence for its minimum and
     * maximum value and creates an integer alphabet given that range.
     *
     */
    template <typename Iterator>
    static int_alphabet from_sequence(Iterator begin, Iterator end) {
        static_assert(std::is_same<char_type, typename std::iterator_traits<Iterator>::value_type>::value,
                      "Character type of alphabet must match the value type of input sequence");
        char_type minval, maxval;
        std::tie(minval, maxval) = getminmax(begin, end);
        return int_alphabet(minval, maxval);
    }


    /**
     * @brief Creates alphabet from underlying distributed sequence.
     *
     * This method scans the underlying sequence for its minimum and
     * maximum value, globally determines the overall min and max,
     * and then creates an integer alphabet given that range.
     *
     */
    template <typename Iterator>
    static int_alphabet from_sequence(Iterator begin, Iterator end, const mxx::comm& comm) {
        static_assert(std::is_same<char_type, typename std::iterator_traits<Iterator>::value_type>::value,
                      "Character type of alphabet must match the value type of input sequence");
        char_type minval, maxval;
        std::tie(minval, maxval) = getminmax(begin, end);
        minval = mxx::allreduce(minval, mxx::min<char_type>(), comm);
        maxval = mxx::allreduce(maxval, mxx::max<char_type>(), comm);
        return int_alphabet(minval, maxval);
    }

    /**
     * @brief Returns the number of characters of the alphabet (excluding the reserved \0 character).
     */
    inline uchar_type sigma() const {
        return m_sigma;
    }

    /**
     * @brief Returns the number of bits required to represent a character of
     * the alphabet: ceil(log_2(sigma+1))
     */
    inline unsigned int bits_per_char() const {
        return m_bits_per_char;
    }

    template <typename word_type>
    inline unsigned int chars_per_word() const {
        unsigned int bits_per_word = sizeof(word_type)*8;
        // if the type is signed, we can't use the msb, thus we need to subtract one
        if (std::is_signed<word_type>::value)
            --bits_per_word;
        return bits_per_word/bits_per_char();
    }

    /// @brief Returns the encoded character, using `bits_per_char` bits.
    inline uchar_type encode(char_type c) const {
        // c - min_char + 1
        return c + offset;
    }

    /// @brief Returns the original character given the encoded character
    inline char_type decode(uchar_type c) const {
        // c + min_char - 1
        return c - offset;
    }

    // TODO:
    // write/read file IO
};

template <typename char_type>
std::ostream& operator<<(std::ostream& os, const int_alphabet<char_type>& a) {
     return os << "int_alphabet{sigma=" << a.sigma() << ", range=[" << a.min_char << "," << a.max_char << "], l=" << a.bits_per_char() << "}";
}

template <typename CharType>
struct alphabet_helper {
    using char_type = CharType;
    using alphabet_type = typename std::conditional<sizeof(char_type) <= 2, alphabet<char_type>, int_alphabet<char_type>>::type;
};


#endif // ALPHABET_HPP
