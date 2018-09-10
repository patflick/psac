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
 * @file    check_suffix_tree.hpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Correctness tests for Suffix trees.
 */
#ifndef CHECK_SUFFIX_TREE_HPP
#define CHECK_SUFFIX_TREE_HPP

#include <vector>
#include <string>
#include <iostream>

#include "suffix_array.hpp"
#include "alphabet.hpp"
#include "rmq.hpp"
#include "check_suffix_array.hpp"

void check_suffix_tree(const std::string& s, const std::vector<size_t>& sa, const std::vector<size_t>& lcp, const std::vector<size_t>& nodes, bool& success) {
    // recreate alphabet mapping
    std::vector<size_t> hist = get_histogram<size_t>(s.begin(), s.end(), 256);
    alphabet<char> alpha = alphabet<char>::from_hist(hist);
    unsigned int sigma = alpha.sigma();

    success = false;

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
            size_t i = range_left-1;
            size_t c;
            if (sa[i] + prev_min == s.size()) {
                c = 0;
            } else {
                ASSERT_GT(s.size(), sa[i] + prev_min);
                c = alpha.encode(s[sa[i]+prev_min]);
            }
            size_t node_offset = (sigma+1)*prev_pos;
            leafs_visited[i] = true;
            edges_visited[node_offset+c] = true;
            ASSERT_EQ(i+s.size(), nodes[node_offset+c]);
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
                    c = alpha.encode(s[sa[i] + prev_min]);
                }
                size_t node_offset = (sigma+1)*prev_pos;
                edges_visited[node_offset+c] = true;
                ASSERT_EQ(split, nodes[node_offset+c]);
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
    success = true;
}

/**
 * @brief   Checks the correctness of the distributed suffix and LCP array.
 *
 * This method gathers all arrays to processor 0 and then uses sequential
 * correctness checkers. Thus this method only works for small inputs, where
 * everything fits onto the memory of a single processor.
 *
 * The template parameters will be deduced from the given distributed suffix
 * array instance.
 *
 * @tparam InputIterator    The type of the char/string input iterator.
 * @tparam index_t          The type of the index (e.g. uint32_t, uint64_t).
 * @tparam test_lcp         Whether the LCP was constructed and should be tested.
 *
 * @param sa            The distributed suffix array instance.
 * @param str_begin     Iterator to the string for which the suffix array was
 *                      constructed.
 * @param str_end       End Iterator to the string for which the suffix array
 *                      was constructed.
 * @param comm          The communictor.
 */
template <typename InputIterator>
void gl_check_suffix_tree(const std::string& local_str, const suffix_array<InputIterator, size_t, true>& sa,
                         const std::vector<size_t> local_nodes, const mxx::comm& comm)
{
    // gather all the data to rank 0
    std::vector<size_t> global_SA = mxx::gatherv(sa.local_SA, 0, comm);
    std::vector<size_t> global_ISA = mxx::gatherv(sa.local_B, 0, comm);
    std::vector<size_t> global_LCP = mxx::gatherv(sa.local_LCP, 0, comm);
    std::vector<size_t> global_nodes = mxx::gatherv(local_nodes, 0, comm);

    // gather string
    // TODO: use iterator or std::string version for mxx?
    std::vector<char> global_str_vec = mxx::gatherv(&(*local_str.begin()), local_str.size(), 0, comm);
    std::string global_str(global_str_vec.begin(), global_str_vec.end());

    if (comm.rank() == 0) {
        if (!check_SA(global_SA, global_ISA, global_str)) {
            std::cerr << "[ERROR] Test unsuccessful" << std::endl;
        } else {
            std::cerr << "[SUCCESS] Suffix Array is correct" << std::endl;
        }

        if (!check_lcp(global_str, global_SA, global_ISA, global_LCP)) {
            std::cerr << "[ERROR] Test unsuccessful" << std::endl;
            exit(1);
        } else {
            std::cerr << "[SUCCESS] LCP Array is correct" << std::endl;
        }

        bool success_ST;
        check_suffix_tree(global_str, global_SA, global_LCP, global_nodes, success_ST);
        if (!success_ST) {
            std::cerr << "[ERROR] Test unsuccessful" << std::endl;
            exit(1);
        } else {
            std::cerr << "[SUCCESS] Suffix Tree is correct" << std::endl;
        }
    }
}


#endif // CHECK_SUFFIX_ARRAY_HPP
