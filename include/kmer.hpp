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

#ifndef KMER_HPP
#define KMER_HPP

#include <vector>
#include <iterator>
#include "alphabet.hpp"

// get max-mer size for a given alphabet and local input size
template<typename word_type, typename Alphabet>
unsigned int get_optimal_k(const Alphabet& a, size_t local_size, const mxx::comm& comm, unsigned int k = 0) {
    // number of characters per word => the `k` in `k-mer`
    unsigned int max_k = a.template chars_per_word<word_type>();
    if (k == 0 || k > max_k) {
        k = max_k;
    }
    // if the input is too small for `k`, choose a smaller `k`
    size_t min_local_size = mxx::allreduce(local_size, mxx::min<size_t>(), comm);
    if (k >= min_local_size) {
        k = min_local_size;
        if (comm.size() == 1 && k > 1)
            k--;
    }
    return k;
}

template <typename word_type, typename char_type>
std::string decode_kmer(const word_type& kmer, unsigned int k, const alphabet<char_type>& alpha, char_type nullchar = '0') {
    std::string result;
    result.resize(k);
    unsigned int l = alpha.bits_per_char();
    for (unsigned int i = 0; i < k; ++i) {
        result[k-i-1] = alpha.decode((kmer >> (i*l)) & ((1 << l) -1));
        if (result[k-i-1] == '\0')
            result[k-i-1] = nullchar;
    }
    return result;
}

template <typename word_type, typename char_type>
std::vector<std::string> decode_kmers(const std::vector<word_type>& kmers, unsigned int k, const alphabet<char_type>& alpha, char_type nullchar = '0') {
    std::vector<std::string> results(kmers.size());
    for (size_t i = 0; i < kmers.size(); ++i) {
        results[i] = decode_kmer(kmers[i], k, alpha, nullchar);
    }
    return results;
}

// TODO: function to get a specfic character from a kmer
template <typename word_type, typename Alphabet>
inline typename Alphabet::char_type get_kmer_char(const word_type kmer, unsigned int k, const Alphabet& alpha, unsigned int i) {
    const unsigned int l = alpha.bits_per_char();
    return alpha.decode((kmer >> ((k-1-i)*l)) & ((1 << l) - 1));
}

/* sequential kmer generation on purely local sequence (no communication) */

template <typename word_type, typename InputIterator, typename Alphabet, typename Func>
void for_each_kmer(InputIterator begin, InputIterator end, unsigned int k, const Alphabet& alpha, Func func) {
    static_assert(std::is_same<typename std::iterator_traits<InputIterator>::value_type, typename Alphabet::char_type>::value, "alphbet character type has to match the input sequence");
    assert(k > 0);
    size_t size = std::distance(begin, end);
    unsigned int l = alpha.bits_per_char();
    assert(l*k < sizeof(word_type)*8);

    // get k-mer mask
    word_type kmer_mask = ((static_cast<word_type>(1) << (l*k)) - static_cast<word_type>(1));
    if (kmer_mask == 0)
        kmer_mask = ~static_cast<word_type>(0);

    InputIterator str_it = begin;

    // fill first k-1 mer
    word_type kmer = 0;
    for (size_t i = 0; i < std::min<size_t>(k-1, size); ++i) {
        kmer <<= l;
        kmer |= alpha.encode(*str_it);
        ++str_it;
    }
    if (size < k-1) {
        kmer <<= l*(k-1 - size);
    }

    // continue to create all k-mers
    while (str_it != end) {
        // get next kmer
        kmer <<= l;
        kmer |= alpha.encode(*str_it);
        kmer &= kmer_mask;
        // add to bucket number array
        func(kmer);
        // iterate
        ++str_it;
    }

    // last k-1 kmers are filled with 0
    for (size_t i = 0; i < std::min<size_t>(k-1, size); ++i) {
        kmer <<= l;
        kmer &= kmer_mask;
        func(kmer);
    }
}

template <typename word_type, typename InputIterator, typename Alphabet, typename Func>
void par_for_each_kmer(InputIterator begin, InputIterator end, unsigned int k, const Alphabet& alpha, const mxx::comm& comm, Func func) {
    static_assert(std::is_same<typename std::iterator_traits<InputIterator>::value_type, typename Alphabet::char_type>::value, "alphbet character type has to match the input sequence");
    unsigned int l = alpha.bits_per_char();
    // get k-mer mask
    word_type kmer_mask = ((static_cast<word_type>(1) << (l*k)) - static_cast<word_type>(1));
    if (kmer_mask == 0)
        kmer_mask = ~static_cast<word_type>(0);

    MXX_ASSERT(std::distance(begin,end) >= k-1);

    // fill first k-mer (until k-1 positions) and send to left processor
    // filling k-mer with first character = MSBs for lexicographical ordering
    InputIterator str_it = begin;
    word_type kmer = 0;
    for (unsigned int i = 0; i < k-1; ++i) {
        kmer <<= l;
        kmer |= alpha.encode(*str_it);
        ++str_it;
    }

    // send first kmer to left processor
    // TODO: use async left shift!
    word_type last_kmer = mxx::left_shift(kmer, comm);

    // continue to create all k-mers
    while (str_it != end) {
        // get next kmer
        kmer <<= l;
        kmer |= alpha.encode(*str_it);
        kmer &= kmer_mask;
        func(kmer);
        // iterate
        ++str_it;
    }

    // finish the receive to get the last k-1 k-kmers with string data from the
    // processor to the right
    // if not last processor
    if (comm.rank() < comm.size()-1) {
        // TODO: use mxx::future to handle this async left shift
        // wait for the async receive to finish
        //MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
        //req.wait();
    } else {
        // in this case the last k-mers contains shifting `$` signs
        // we assume this to be the `\0` value
        last_kmer = 0;
    }


    // construct last (k-1) k-mers
    for (unsigned int i = 0; i < k-1; ++i) {
        kmer <<= l;
        kmer |= (last_kmer >> (l*(k-i-2)));
        kmer &= kmer_mask;
        func(kmer);
    }
}


template <typename word_type, typename InputIterator, typename Alphabet>
std::vector<word_type> kmer_generation(InputIterator begin, InputIterator end, unsigned int k, const Alphabet& alpha) {
    static_assert(std::is_same<typename std::iterator_traits<InputIterator>::value_type, typename Alphabet::char_type>::value, "alphbet character type has to match the input sequence");
    assert(k > 0);
    size_t size = std::distance(begin, end);

    // init output
    std::vector<word_type> kmers(size);

    if (k > 1) {
        auto buk_it = kmers.begin();
        // create vector of all kmers
        for_each_kmer<word_type>(begin, end, k, alpha, [&buk_it](word_type kmer) {
            *buk_it = kmer;
            ++buk_it;
        });
    } else {
        assert(k == 1);
        std::copy(begin, end, kmers.begin());
    }

    return kmers;
}

template <typename word_type, typename InputIterator, typename Alphabet>
std::vector<word_type> kmer_generation(InputIterator begin, InputIterator end, unsigned int k, const Alphabet& alpha, const mxx::comm& comm) {
    static_assert(std::is_same<typename std::iterator_traits<InputIterator>::value_type, typename Alphabet::char_type>::value, "alphbet character type has to match the input sequence");
    size_t local_size = std::distance(begin, end);
    // init output
    std::vector<word_type> kmers(local_size);
    if (k > 1) {
        auto buk_it = kmers.begin();

        // create vector of local kmers
        par_for_each_kmer<word_type>(begin, end, k, alpha, comm, [&buk_it](word_type kmer){
            *buk_it = kmer;
            ++buk_it;
        });
    } else {
        assert(k == 1);
        std::copy(begin, end, kmers.begin());
    }

    return kmers;
}


template <typename word_type, typename InputIterator>
std::vector<word_type> kmer_hist(InputIterator begin, InputIterator end, unsigned int k, const alphabet<typename std::iterator_traits<InputIterator>::value_type>& alpha) {
    // compute sizes
    assert(k > 0);
    unsigned int l = alpha.bits_per_char();
    assert(l*k < sizeof(word_type)*8);
    size_t tbl_size = 1 << (l*k);

    // init output
    std::vector<word_type> hist(tbl_size, 0);

    // create histogram
    for_each_kmer<word_type>(begin, end, k, alpha, [&hist](word_type kmer){
        ++hist[kmer];
    });

    return hist;
}

template <typename word_type, typename InputIterator>
std::vector<word_type> kmer_hist(InputIterator begin, InputIterator end, unsigned int k, const alphabet<typename std::iterator_traits<InputIterator>::value_type>& alpha, const mxx::comm& comm) {
    // compute sizes
    assert(k > 0);
    unsigned int l = alpha.bits_per_char();
    assert(l*k < sizeof(word_type)*8);
    size_t tbl_size = 1 << (l*k);

    // init output
    std::vector<word_type> hist(tbl_size, 0);

    // create histogram of local kmers
    par_for_each_kmer<word_type>(begin, end, k, alpha, comm, [&hist](word_type kmer){
        ++hist[kmer];
    });

    // allreduce the whole kmer table
    hist = mxx::allreduce(hist, comm);

    return hist;
}

/// kmer generation from a stringset (for GSA, GST, etc)
template <typename word_type, typename StringSet, typename char_type>
std::vector<word_type> kmer_gen_stringset(const StringSet& ss, unsigned int k, const alphabet<char_type>& alpha, const mxx::comm& comm = mxx::comm()) {
    // Two cases: strings are split accross boundaries, or not

    // get k-mer mask
    unsigned int l = alpha.bits_per_char();
    word_type kmer_mask = ((static_cast<word_type>(1) << (l*k)) - static_cast<word_type>(1));
    if (kmer_mask == 0)
        kmer_mask = ~static_cast<word_type>(0);

    MXX_ASSERT(ss.sum_sizes > 0);
    MXX_ASSERT(k <= ss.sum_sizes);

    // allocate output vector of kmers
    std::vector<word_type> kmers(ss.sum_sizes);
    auto buk_it = kmers.begin();
    word_type right_kmer;

    // iterate over all subsequences (strings)
    for (size_t s = 0; s < ss.sizes.size(); ++s) {
        size_t slen = ss.sizes[s];

        // fill first kmer
        auto str_it = ss.str_begins[s];
        auto send = str_it + slen;
        word_type kmer = 0;
        for (unsigned int i = 0; i < std::min<size_t>(slen, k-1); ++i) {
            kmer <<= l;
            word_type s = (unsigned char)(*str_it);
            kmer |= alpha.encode(s);
            ++str_it;
        }
        // if the string ends before k-1, then fill with 0
        // unless its the last one and its split
        if (slen < k-1) {
            kmer <<= l*(k-1 - slen);
        }

        if (s == 0) {
            right_kmer = mxx::left_shift(kmer, comm);
        }

        if (slen >= k-1) { // XXX: maybe not necessary
            // continue to create all k-mers
            while (str_it != send) {
                // get next kmer
                kmer <<= l;
                word_type s = (unsigned char)(*str_it);
                kmer |= alpha.encode(s);
                kmer &= kmer_mask;
                // add to bucket number array
                *buk_it = kmer;
                // iterate
                ++str_it;
                ++buk_it;
            }
        }

        // construct last (k-1) k-mers
        if (s == ss.sizes.size()-1 && ss.last_split) {
            // if last string last is split:
            // use received kmer to fill last few
            size_t start_shift = 1; // how many chars we take for the first kmer
            if (slen < k-1) {
                start_shift = k-slen;
            }
            for (unsigned int i = start_shift; i <= k-1; ++i) {
                kmer <<= l;
                // shift so that we have `i` chars left in the right_kmer
                // assuming the right k-mer is a k-1 mer
                kmer |= (right_kmer >> (l*(k-1 - i)));
                kmer &= kmer_mask;
                *buk_it = kmer;
                ++buk_it;
            }
        } else {
            // otherwise: fill with zero:
            for (unsigned int i = 0; i < std::min<size_t>(slen, k-1); ++i) {
                kmer <<= l;
                kmer &= kmer_mask;
                *buk_it = kmer;
                ++buk_it;
            }
        }
    }
    return kmers;
}


#endif // KMER_HPP

