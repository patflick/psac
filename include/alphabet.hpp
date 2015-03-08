#ifndef ALPHABET_HPP
#define ALPHABET_HPP

#include <mpi.h>  // TODO: remove MPI dependency from this file once the MPI functions are relocated elsewhere
#include <vector>
#include <string>

#include "mpi_utils.hpp"
#include "bitops.hpp"


// TODO: put the histrogram functions somewhere else and leave the alphabet
//       to be non mpi
template<typename T, typename Iterator>
std::vector<T> get_histogram(Iterator begin, Iterator end, std::size_t size = 0)
{
    if (size == 0)
        size = static_cast<std::size_t>(*std::max_element(begin, end)) + 1;
    std::vector<T> hist(size);

    while (begin != end)
    {
        ++hist[static_cast<std::size_t>(*(begin++))];
    }

    return hist;
}

template <typename InputIterator, typename index_t>
std::vector<index_t> alphabet_histogram(InputIterator begin, InputIterator end, MPI_Comm comm)
{
    static_assert(std::is_same<typename std::iterator_traits<InputIterator>::value_type, char>::value, "Iterator must be of value type `char`.");
    // get local histogram of alphabet characters
    std::vector<index_t> hist = get_histogram<index_t>(begin, end, 256);

    std::vector<index_t> out_hist(256);
    // get MPI type
    mxx::datatype<index_t> dt;
    MPI_Datatype mpi_dt = dt.type();


    MPI_Allreduce(&hist[0], &out_hist[0], 256, mpi_dt, MPI_SUM, comm);

    return out_hist;
}

template <typename index_t>
std::vector<char> alphabet_mapping_tbl(const std::vector<index_t>& global_hist)
{
    std::vector<char> mapping(256, 0);

    char next = static_cast<char>(1);
    for (std::size_t c = 0; c < 256; ++c)
    {
        if (global_hist[c] != 0)
        {
            mapping[c] = next;
            ++next;
        }
    }
    return mapping;
}

template <typename index_t>
unsigned int alphabet_unique_chars(const std::vector<index_t>& global_hist)
{
    unsigned int unique_count = 0;
    for (std::size_t c = 0; c < 256; ++c)
    {
        if (global_hist[c] != 0)
        {
            ++unique_count;
        }
    }
    return unique_count;
}


unsigned int alphabet_bits_per_char(unsigned int sigma)
{
    // since we have to account for the `0` character, we use ceil(log(unique_chars + 1))
    return ceillog2(sigma+1);
}

template<typename word_t>
unsigned int alphabet_chars_per_word(unsigned int bits_per_char)
{
    unsigned int bits_per_word = sizeof(word_t)*8;
    // TODO: this is currently a "work-around": if the type is signed, we
    //       can't use the msb, thus we need to subtract one
    if (std::is_signed<word_t>::value)
        --bits_per_word;
    return bits_per_word/bits_per_char;
}

#endif // ALPHABET_HPP
