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
template<typename word_type, typename CharType>
unsigned int get_optimal_k(const alphabet<CharType>& a, size_t local_size, const mxx::comm& comm, unsigned int k = 0) {
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


template <typename word_type, typename InputIterator>
std::vector<word_type> kmer_generation(InputIterator begin, InputIterator end, unsigned int k, const alphabet<typename std::iterator_traits<InputIterator>::value_type>& alpha, const mxx::comm& comm = mxx::comm()) {
    size_t local_size = std::distance(begin, end);
    unsigned int l = alpha.bits_per_char();
    // get k-mer mask
    word_type kmer_mask = ((static_cast<word_type>(1) << (l*k)) - static_cast<word_type>(1));
    if (kmer_mask == 0)
        kmer_mask = ~static_cast<word_type>(0);

    // fill first k-mer (until k-1 positions) and send to left processor
    // filling k-mer with first character = MSBs for lexicographical ordering
    InputIterator str_it = begin;
    word_type kmer = 0;
    for (unsigned int i = 0; i < k-1; ++i) {
        kmer <<= l;
        word_type s = (unsigned char)(*str_it);
        kmer |= alpha.encode(s);
        ++str_it;
    }

    // send first kmer to left processor
    // TODO: use async left shift!
    word_type last_kmer = mxx::left_shift(kmer, comm);

    // init output
    std::vector<word_type> kmers(local_size);
    auto buk_it = kmers.begin();
    // continue to create all k-mers
    while (str_it != end) {
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

        // add to bucket number array
        *buk_it = kmer;
        ++buk_it;
    }

    return kmers;
}

/// kmer generation from a stringset (for GSA, GST, etc)
template <typename word_type, typename StringSet, typename char_type>
std::vector<word_type> kmer_gen_stringset(const StringSet& ss, unsigned int k, const alphabet<char_type>& alpha, const mxx::comm& comm = mxx::comm()) {
    // TODO: how to iterate through strings (fill with 0, etc)?
    // TODO: design appropriate API for stringset
    // Two cases: strings are split accross boundaries, or not

    // get k-mer mask
    unsigned int l = alpha.bits_per_char();
    word_type kmer_mask = ((static_cast<word_type>(1) << (l*k)) - static_cast<word_type>(1));
    if (kmer_mask == 0)
        kmer_mask = ~static_cast<word_type>(0);

    size_t total_length = ss.total_lengths();

    std::vector<word_type> kmers(total_length);
    auto buk_it = kmers.begin();
    if (!ss.is_split()) {
        // easy, no communication necessary, simply iterate over strings and generate kmers and 0-fill at end
        for (auto s : ss) {
            size_t slen = s.size();

            // fill first kmer
            auto str_it = s.begin();
            word_type kmer = 0;
            for (unsigned int i = 0; i < std::min<size_t>(slen, k-1); ++i) {
                kmer <<= l;
                word_type s = (unsigned char)(*str_it);
                kmer |= alpha.encode(s);
                ++str_it;
            }
            if (slen < k-1) {
                kmer <<= l*(k-1 - slen);
            } else {
                // continue to create all k-mers
                while (str_it != s.end()) {
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

            for (unsigned int i = 0; i < std::min<size_t>(slen, k-1); ++i) {
                // output last kmers
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

