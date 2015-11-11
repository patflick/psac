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
 * @file    lcp.hpp
 * @ingroup group
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements LCP construction (from suffix array) sequentially.
 */

#ifndef LCP_HPP
#define LCP_HPP

#include <string>
#include <vector>
#include <iostream>

/**
 * @brief   Constructs the LCP given the string `S`, the suffix array `SA` and
 *          the inverse suffix array (rank array) `ISA`.
 *
 * @tparam index_t  The index type of the SA, ISA and LCP (e.g. uint32_t, or
 *                  std::size_t).
 * @param S         The input string.
 * @param SA        The suffix array.
 * @param ISA       The inverse suffix array (rank array).
 * @param LCP[out]  The Longest-Common-Prefix array for the given suffix array.
 */
template <typename index_t>
void lcp_from_sa(const std::string& S, const std::vector<index_t>& SA, const std::vector<index_t>& ISA, std::vector<index_t>& LCP) {
    // TODO: cite the source for this linear O(n) algorithm!

    // input sizes must be equal
    assert(S.size() == SA.size());
    assert(SA.size() == ISA.size());

    // init LCP array if not yet of correct size
    if (LCP.size() != S.size()) {
        LCP.resize(S.size());
    }

    // first LCP is undefined -> set to 0:
    LCP[0] = 0;

    std::size_t h = 0;

    // in string order!
    for (std::size_t i = 0; i < S.size(); ++i) {
        // length of currently equal characters (in string order, next LCP value
        // is always >= current lcp - 1)
        std::size_t k = 0;
        if (h > 0)
            k = h-1;
        // comparing suffix starting from i=SA[ISA[i]] with the previous
        // suffix in SA order: SA[ISA[i]-1]
        while (i+k < S.size() && ISA[i] > 0 && SA[ISA[i]-1]+k < S.size() && S[i+k] == S[SA[ISA[i]-1]+k])
            k++;
        LCP[ISA[i]] = k;
        h = k;
    }
}


#endif // LCP_HPP
