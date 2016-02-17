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
#include <alphabet.hpp>
#include <rmq.hpp>
#include <mxx/distribution.hpp>
#include <vector>
#include <algorithm>



void check_suffix_tree(const std::string& s, const std::vector<size_t>& sa, const std::vector<size_t>& lcp, const std::vector<size_t>& nodes) {
    // recreate alphabet mapping
    std::vector<size_t> hist = get_histogram<size_t>(s.begin(), s.end(), 256);
    std::vector<uint16_t> alphabet_map = alphabet_mapping_tbl(hist);
    unsigned int sigma = alphabet_unique_chars(hist);


    ASSERT_EQ((sigma+1)*s.size(), nodes.size());

    rmq<typename std::vector<size_t>::const_iterator> minquery(lcp.cbegin(), lcp.cend());

    std::vector<bool> leafs_visited(s.size(), false);
    std::vector<bool> edges_visited(nodes.size(), false);

    size_t none = 0; // TODO: modify suffix tree construction to have special `none` value
    std::deque<std::tuple<size_t, size_t, size_t, size_t>> q;
    q.emplace_back(1, s.size(), 0, 0);

    while (!q.empty()) {
        size_t range_left;
        size_t range_right;
        size_t prev_min;
        size_t prev_pos;
        std::tie(range_left, range_right, prev_min, prev_pos) = q.back();
        q.pop_back();

        if (range_left == range_right) {
            // leaf node with parent `prev_pos` with lcp `prev_min`
            // TODO: get character at SA[i]+LCP[i]
            size_t i = range_left-1;
            size_t c;
            if (sa[i] + prev_min == s.size()) {
                c = 0;
            } else {
                ASSERT_GT(s.size(), sa[i] + prev_min);
                c = alphabet_map[s[sa[i]+prev_min]];
            }
            size_t node_offset = (sigma+1)*prev_pos;
            leafs_visited[i] = true;
            edges_visited[node_offset+c] = true;
            ASSERT_EQ(i+s.size(), nodes[node_offset+c]) << "range=" << range_left << " for node_offset=" << node_offset << ", c=" << c;
        } else {
            ASSERT_LT(range_left, range_right);

            auto min_pos = minquery.query(lcp.cbegin()+range_left, lcp.cbegin()+range_right);
            size_t m = *min_pos;

            if (m == prev_min) {
                // further subdivision of current node into subranges
                size_t split = min_pos - lcp.cbegin();
                // recursion (push right before left segment: left-first-dfs)
                q.emplace_back(split + 1, range_right, prev_min, prev_pos);
                q.emplace_back(range_left, split, prev_min, prev_pos);
            } else {
                ASSERT_LT(prev_min, m);
                // we've found a new internal node
                // RMQ always returns the left most minimum element
                // -> this is the index of the internal node in `nodes`
                size_t split = min_pos - lcp.cbegin();
                q.emplace_back(split + 1, range_right, m, split);
                q.emplace_back(range_left, split, m, split);

                size_t c;
                size_t i = split;
                //std::cout << "sa[i]= " << sa[i] << ", m=" << m << std::endl;
                if (sa[i] + prev_min == s.size()) {
                    c = 0;
                } else {
                    ASSERT_GT(s.size(), sa[i] + prev_min);
                    c = alphabet_map[s[sa[i] + prev_min]];
                }
                size_t node_offset = (sigma+1)*prev_pos;
                edges_visited[node_offset+c] = true;
                ASSERT_EQ(split, nodes[node_offset+c]) << "range: " << range_left << "," << split << "," << range_right;
            }
        }
    }

    // verify that all leafs have been visited
    for (bool b : leafs_visited) {
        ASSERT_TRUE(b);
    }
    // verify that all edges have been visited
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (nodes[i] != none) {
            ASSERT_TRUE(edges_visited[i]);
        }
    }
}

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

            check_suffix_tree(str, sar, lcp, nodes);

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
    mxx::comm c;
    
   for (size_t n : {116, 1000, 23713, 177861}) {
   //size_t n = 1000;
        std::string str;
        if (c.rank() == 0) {
            str = rand_dna(n, 13);
        }
        std::string local_str = mxx::stable_distribute(str, c);

        // build SA and LCP
        suffix_array<std::string::iterator, size_t, true> sa(local_str.begin(), local_str.end(), c);
        sa.construct();

        // build ST
        std::vector<size_t> local_nodes = construct_suffix_tree(sa, c);


        // gather on master
        std::vector<size_t> nodes = mxx::gatherv(local_nodes, 0, c);
        std::vector<size_t> sar = mxx::gatherv(sa.local_SA, 0, c);
        std::vector<size_t> lcp = mxx::gatherv(sa.local_LCP, 0, c);

        if (c.rank() == 0) {
            check_suffix_tree(str, sar, lcp, nodes);
        }
    }
}


// TEST suffix tree structure for random DNA sequences
TEST(PsacST, Repeats3Test) {
    mxx::comm c;
    for (size_t n : {25, 97, 151, 14681}) {
//    size_t n = 97;
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
        suffix_array<std::string::iterator, size_t, true> sa(local_str.begin(), local_str.end(), c);
        sa.construct();

        // build ST
        std::vector<size_t> local_nodes = construct_suffix_tree(sa, c);


        // gather on master
        std::vector<size_t> nodes = mxx::gatherv(local_nodes, 0, c);
        std::vector<size_t> sar = mxx::gatherv(sa.local_SA, 0, c);
        std::vector<size_t> lcp = mxx::gatherv(sa.local_LCP, 0, c);

        if (c.rank() == 0) {
            check_suffix_tree(str, sar, lcp, nodes);
        }
    }
}
