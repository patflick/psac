/*
 * Copyright 2016 Georgia Institute of Technology
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
 * @brief   Unit tests for suffix tree construction.
 */

#include <gtest/gtest.h>
#include <cxx-prettyprint/prettyprint.hpp>
#include <ansv.hpp>
#include <suffix_array.hpp>
#include <rmq.hpp>
#include <mxx/distribution.hpp>
#include <vector>
#include <algorithm>


TEST(PsacST, SimpleSuffixTree) {
    mxx::comm c;

    // distribute string
    std::string str;
    if (c.rank() == 0) {
        str = "mississippi";
    }
    std::string local_str = mxx::stable_distribute(str, c);

    // build SA and LCP
    suffix_array<std::string::iterator, size_t, true> sa(local_str.begin(), local_str.end(), c);
    sa.construct();

    // build ST
    std::vector<size_t> local_nodes = construct_suffix_tree(sa, c);

    // gather and print on master
    std::vector<size_t> nodes = mxx::gatherv(local_nodes, 0, c);
    std::vector<size_t> sar = mxx::gatherv(sa.local_SA, 0, c);
    std::vector<size_t> lcp = mxx::gatherv(sa.local_LCP, 0, c);

    if (c.rank() == 0) {
        std::cout << "SA: " << sar << std::endl;
        std::cout << "LCP: " << lcp << std::endl;
        std::cout << "ST nodes: " << nodes << std::endl;
    }
}
