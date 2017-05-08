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
#include <suffix_tree.hpp>
#include <check_suffix_array.hpp>
#include <check_suffix_tree.hpp>
#include <cxx-prettyprint/prettyprint.hpp>
#include <suffix_array.hpp>
#include <alphabet.hpp>
#include <rmq.hpp>
#include <mxx/distribution.hpp>
#include <vector>
#include <algorithm>


TEST(PsacST, SimpleSuffixTree) {
    mxx::comm comm;

    // distribute string
    std::string str;
    if (comm.rank() == 0) {
        str = "mississippi";
    }
    mxx::comm c = comm.split(comm.rank() < 11);
    if (comm.rank() < 11) {
        std::string local_str = mxx::stable_distribute(str, c);

        // build SA and LCP
        suffix_array<char, size_t, true> sa(c);
        sa.construct(local_str.begin(), local_str.end());

        // build ST
        std::vector<size_t> local_nodes = construct_suffix_tree(sa, local_str.begin(), local_str.end(), c);

        // gather and print on master
        std::vector<size_t> nodes = mxx::gatherv(local_nodes, 0, c);
        std::vector<size_t> sar = mxx::gatherv(sa.local_SA, 0, c);
        std::vector<size_t> lcp = mxx::gatherv(sa.local_LCP, 0, c);

        if (c.rank() == 0) {
            std::cout << "SA: " << sar << std::endl;
            std::cout << "LCP: " << lcp << std::endl;
            std::cout << "ST nodes: " << nodes << std::endl;

            bool success;
            check_suffix_tree(str, sar, lcp, nodes, success);

            size_t none = 0;
            // the actual solution to the suffix tree:
            std::vector<size_t> solution =
                {0 , 1, 15, 6, 9,
                 11, none, none, 12, 3,
                 none, none, none, none, none,
                 none, none, none, 13, 14,
                 none, none, none, none, none,
                 none, none, none, none, none,
                 none, 16, none, 17, none,
                 none, none, none, none, none,
                 none, none, none, 18, 19,
                 none, 8, none, none, 10,
                 none, none, none, 20, 21};
            ASSERT_EQ(solution.size(), nodes.size());
            for (size_t i = 0; i < solution.size(); ++i) {
                ASSERT_EQ(solution[i], nodes[i]);
            }
        }
    }
}

// TEST suffix tree structure for random DNA sequences
TEST(PsacST, RandDNATest) {
    
    for (size_t n : {116, 1000, 23713, 177861}) {
   //size_t n = 1000;
        mxx::comm comm;
        comm.barrier();
        mxx::comm c = comm.split((size_t)comm.rank() < n);
        if ((size_t)comm.rank() >= n)
           continue;
        std::string str;
        if (c.rank() == 0) {
           str = rand_dna(n, 13);
        }
        std::string local_str = mxx::stable_distribute(str, c);

        // build SA and LCP
        suffix_array<char, size_t, true> sa(c);
        sa.construct(local_str.begin(), local_str.end());

        // build ST
        std::vector<size_t> local_nodes = construct_suffix_tree(sa, local_str.begin(), local_str.end(), c);


        // gather on master
        std::vector<size_t> nodes = mxx::gatherv(local_nodes, 0, c);
        std::vector<size_t> sar = mxx::gatherv(sa.local_SA, 0, c);
        std::vector<size_t> lcp = mxx::gatherv(sa.local_LCP, 0, c);

        if (c.rank() == 0) {
           bool success;
           check_suffix_tree(str, sar, lcp, nodes, success);
        }
    }
}


// TEST suffix tree structure for repeats of the form: (abc)^n
TEST(PsacST, Repeats3Test) {
    for (size_t n : {3, 25, 97, 151, 14681}) {
//    size_t n = 97;
        mxx::comm comm;
        comm.barrier();
        mxx::comm c = comm.split((size_t)comm.rank() < n);
        if ((size_t)comm.rank() >= n)
           continue;
        std::string str;
        if (c.rank() == 0) {
            str.resize(n);
            char chars[] = {'a', 'b', 'c'};
            for (size_t i = 0; i < n; ++i) {
                str[i] = chars[i % 3];
            }
        }
        std::string local_str = mxx::stable_distribute(str, c);

        // build SA and LCP
        suffix_array<char, size_t, true> sa(c);
        sa.construct(local_str.begin(), local_str.end());
        gl_check_correct(sa, local_str.begin(), local_str.end(), c);

        // build ST
        std::vector<size_t> local_nodes = construct_suffix_tree(sa, local_str.begin(), local_str.end(), c);

        // gather on master
        std::vector<size_t> nodes = mxx::gatherv(local_nodes, 0, c);
        std::vector<size_t> sar = mxx::gatherv(sa.local_SA, 0, c);
        std::vector<size_t> lcp = mxx::gatherv(sa.local_LCP, 0, c);

        if (c.rank() == 0) {
            bool success;
            check_suffix_tree(str, sar, lcp, nodes, success);
        }
    }
}
