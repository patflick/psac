/*
 * Copyright 2018 Georgia Institute of Technology
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

#ifndef LOOKUP_TABLE_HPP
#define LOOKUP_TABLE_HPP

#include <vector>
#include <algorithm>

#include "alphabet.hpp"
#include "kmer.hpp"

// Top-Level Lookup-Table (small kmer index)

/**
 * @brief Top-Level Lookup-Table
 *
 * The table is represented as a prefix-sum over the k-mer histogram.
 *
 * @tparam index_t  Index type (unsigned integer). Should be big enough to hold
 *                  the number of total elements indexed.
 */
template <typename index_t>
struct lookup_index {
    /// prefixsum of k-mer histogram
    std::vector<index_t> table;
    /// k
    unsigned int k;
    /// alphabet
    alphabet<char> alpha;
    /// output type
    using range_t = std::pair<index_t,index_t>;

    /**
     * @brief Constructs the lookup table from a k-mer histogram.
     *
     * @param hist  k-mer histogram
     * @param k
     * @param a     alphabet for underlying string.
     */
    template <typename word_type>
    void construct_from_hist(const std::vector<word_type>& hist, unsigned int k, const alphabet<char>& a) {
        this->alpha = a;
        this->k = k;

        if (table.size() != hist.size()) {
            table.resize(hist.size());
        }
        std::partial_sum(hist.begin(), hist.end(), table.begin());
    }

    /**
     * @brief Constructs the lookup table from a given string and alphabet.
     *
     * @param begin     Iterator the string start.
     * @param end       Iterator to string end.
     * @param bits      Used to compute `k` as `k = bits / log(alphabet size)`.
     *                  The table will have size 2^bits.
     * @param a         The alphabet of the string.
     */
    template <typename Iterator>
    void construct(Iterator begin, Iterator end, unsigned int bits, const alphabet<char>& a) {
        this->alpha = a;

        unsigned int l = this->alpha.bits_per_char();
        this->k = bits / l;
        assert(k >= 1);

        // scan through string to create kmer histogram
        table = kmer_hist<index_t>(begin, end, k, alpha);

        // prefix sum over hist
        std::partial_sum(table.begin(), table.end(), table.begin());
    }

    /**
     * @brief Constructs the lookup table from a given string.
     *
     * @param begin     Iterator the string start.
     * @param end       Iterator to string end.
     * @param bits      Used to compute `k` as `k = bits / log(alphabet size)`.
     *                  The table will have size 2^bits.
     */
    template <typename Iterator>
    void construct(Iterator begin, Iterator end, unsigned int bits) {
        // get alphabet?
        alphabet<char> a = alphabet<char>::from_sequence(begin, end);
        this->construct(begin, end, bits, a);
    }

    /**
     * @brief Queries the lookup table for a given string patter `P`.
     *
     * Returns the range [l,r) of suffixes for the first `k` characters of `P`.
     *
     * @param P     Input string pattern.
     *
     * @return  Range [l,r) of sorted suffixes where `P[0..k-1]` occurs.
     */
    template <typename String>
    range_t lookup(const String& P) const {
        unsigned int l = alpha.bits_per_char();
        assert(l*k < sizeof(index_t)*8);

        if (P.size() >= k) {
            // create kmer from P
            index_t kmer = 0;
            for (size_t i = 0; i < k; ++i) {
                kmer <<= l;
                kmer |= alpha.encode(P[i]);
            }

            if (kmer == 0) {
                return range_t(0, table[0]);
            } else {
                return range_t(table[kmer-1],table[kmer]);
            }
        } else {
            // create partial kmer
            index_t kmer = 0;
            for (size_t i = 0; i < P.size(); ++i) {
                kmer <<= l;
                kmer |= alpha.encode(P[i]);
            }
            size_t extra = k - P.size();
            for (size_t i = P.size(); i < k; ++i) {
                kmer <<= l;
            }
            if (kmer == 0) {
                return range_t(0, table[kmer + (1 << (extra*l))-1]);
            } else {
                return range_t(table[kmer-1], table[kmer + (1 << (extra*l))-1]);
            }
        }
    }
};

#endif // LOOKUP_TABLE_HPP
