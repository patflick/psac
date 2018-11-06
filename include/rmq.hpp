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
 * @file    rmq.hpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements a succinct Range-Minimum-Query in a STL compatible format.
 */
#ifndef RMQ_HPP
#define RMQ_HPP

#include <vector>
#include <iterator>
#include <limits>
#include <algorithm>
#include <iostream>
//#include <cstdint>
#include <stdint.h>
#include <assert.h>

#include "bitops.hpp"


template<typename Iterator, typename index_t = std::size_t>
class rmq {
public:
    // superblock size is log^(2+epsilon)(n)
    // we choose it as bitsize^2:
    //  - 64 bits -> 4096
    //  - 32 bits -> 1024
    //  - both bound by 2^16 -> uint16_t
    //  block size: one cache line
    constexpr static int BLOCK_SIZE = 64/sizeof(index_t);
    // cache line of blocks per superblock
    constexpr static int SUPERBLOCK_SIZE = 64/sizeof(uint16_t) * 64/sizeof(index_t);

    constexpr static int NB_PER_SB = SUPERBLOCK_SIZE / BLOCK_SIZE;

    static_assert(sizeof(index_t) == 8 || sizeof(index_t) == 4, "TODO: RMQ implemented only for sizeof(index_t) = 8 or 4");
    constexpr static int LOG_IDXT = (sizeof(index_t) == 8) ? 3 : 4;
    constexpr static int LOG_B_SIZE = 6 - LOG_IDXT;
    constexpr static int LOG_SB_SIZE = 12 - LOG_IDXT - 1;
    constexpr static int LOG_NB_PER_SB = LOG_SB_SIZE - LOG_B_SIZE;

protected:
    index_t n;

    // the original sequence
    Iterator _begin;
    Iterator _end;

    // number of blocks per level
    index_t n_blocks;
    index_t n_superblocks;

    // saves minimum as block index for combination of superblocks relative to
    // global start index
    std::vector<std::vector<index_t> > superblock_mins;
    // saves minimum for combination of blocks relative to superblock start
    // index
    std::vector<std::vector<uint16_t> > block_mins;



public:

    rmq() : n(0) {}

    rmq(const rmq& o) = default;
    rmq(rmq&& o) = default;
    rmq& operator=(const rmq& o) = default;
    rmq& operator=(rmq&& o) = default;

    rmq(Iterator begin, Iterator end)
        : _begin(begin), _end(end)
    {
        // get size
        assert((index_t)std::distance(begin, end) < std::numeric_limits<index_t>::max());
        n = std::distance(begin, end);

        // get number of blocks
        n_superblocks = ((n-1) >> LOG_SB_SIZE) + 1;
        n_blocks = ((n-1) >> LOG_B_SIZE) + 1;


        // start by finding index of mins per block
        // for each superblock
        superblock_mins.push_back(std::vector<index_t>(n_superblocks));
        block_mins.push_back(std::vector<uint16_t>(n_blocks));
        Iterator it = begin;
        while (it != end) {
            // find index of minimum block in superblock
            Iterator min_pos = it;
            Iterator sb_end_it = it;
            std::advance(sb_end_it, std::min<std::size_t>(std::distance(it, end), SUPERBLOCK_SIZE));
            for (Iterator block_it = it; block_it != sb_end_it; ) {
                Iterator block_end_it = block_it;
                std::advance(block_end_it, std::min<std::size_t>(std::distance(block_it, end), BLOCK_SIZE));
                Iterator block_min_pos = std::min_element(block_it, block_end_it);
                // save minimum for superblock min
                if (*block_min_pos < *min_pos) {
                    min_pos = block_min_pos;
                }
                // save minimum for block min, relative to superblock start
                index_t block_min_idx = static_cast<index_t>(std::distance(it, block_min_pos));
                assert(block_min_idx < SUPERBLOCK_SIZE);
                block_mins[0][std::distance(begin, block_it) >> LOG_B_SIZE] = static_cast<uint16_t>(block_min_idx);
                block_it = block_end_it;
            }
            superblock_mins[0][std::distance(begin, it) >> LOG_SB_SIZE] = static_cast<index_t>(std::distance(begin, min_pos));
            it = sb_end_it;
        }

        // fill superblock lookup with dynamic programming
        index_t level = 1;
        for (index_t dist = 2; dist/2 < n_superblocks; dist <<= 1) {
            superblock_mins.push_back(std::vector<index_t>(n_superblocks - dist/2));
            for (index_t i = 0; i+dist/2 < n_superblocks; ++i) {
                index_t right_idx = std::min(i+dist/2, n_superblocks-dist/4-1);
                if (*(begin + superblock_mins[level-1][right_idx]) < *(begin + superblock_mins[level-1][i])) {
                    assert(i < superblock_mins.back().size());
                    assert(superblock_mins.size() == level+1);
                    assert(superblock_mins[level-1].size() > right_idx);
                    superblock_mins.back()[i] = superblock_mins[level-1][right_idx];
                } else {
                    assert(i < superblock_mins.back().size());
                    assert(superblock_mins.size() == level+1);
                    superblock_mins.back()[i] = superblock_mins[level-1][i];
                }
            }
            level++;
        }

        // now the same thing for blocks (but index relative to their
        // superblock)
        level = 1;
        index_t last_sb_nblocks = n_blocks - ((n_superblocks-1) << LOG_NB_PER_SB);
        for (index_t dist = 2; dist/2 < std::min<size_t>(n_blocks,NB_PER_SB); dist <<= 1) {
            if (n_blocks - n_superblocks*dist/2 == 0)
                break;
            index_t last_sb_cur_nblocks = 0;
            if (last_sb_nblocks > dist/2)
                last_sb_cur_nblocks = last_sb_nblocks - dist/2;
            block_mins.push_back(std::vector<uint16_t>((n_superblocks-1)*(NB_PER_SB - dist/2) + last_sb_cur_nblocks));
            for (index_t sb = 0; sb < n_superblocks; ++sb) {
                index_t pre_sb_offset = sb*(NB_PER_SB - dist/4);
                index_t sb_offset = sb*(NB_PER_SB - dist/2);
                index_t blocks_in_sb = std::min<size_t>(n_blocks - sb*NB_PER_SB, NB_PER_SB);
                for (index_t i = 0; i+dist/2 < blocks_in_sb; ++i) {
                    // TODO: right_idx might become negative for last superblock??
                    index_t right_idx = std::min(i + dist/2, blocks_in_sb-dist/4-1);
                    if (*(begin + block_mins[level-1][right_idx+pre_sb_offset] + (sb << LOG_SB_SIZE))
                      < *(begin + block_mins[level-1][i        +pre_sb_offset] + (sb << LOG_SB_SIZE)))
                    {
                        block_mins.back()[i+sb_offset] = block_mins[level-1][right_idx+pre_sb_offset];
                    } else {
                        block_mins.back()[i+sb_offset] = block_mins[level-1][i+pre_sb_offset];
                    }
                }
            }
            level++;
        }

        //assert(check_superblock_correctness());
        //assert(check_block_correctness());
    }

    /**
     * @brief Query the exclusive iterator range [begin, end)
     *
     * @return Iterator to minimum item in the query range.
     */
    Iterator query(Iterator begin, Iterator end) const {
        // find superblocks fully contained within range
        index_t begin_idx = std::distance(_begin, begin);
        index_t end_idx = std::distance(_begin, end);
        return _begin + (*this)(begin_idx, end_idx-1);
    }

    /**
     * @brief Query the inclusive query range [l,r]
     *
     * @return The index of the minimum item in the range [l,r]
     */
    size_t operator()(const size_t l, const size_t r) const {
        const size_t begin_idx = l;
        const size_t end_idx = r+1;
        assert(begin_idx < end_idx);
        assert(end_idx <= n);

        // round up to next superblock
        index_t left_sb  = ((begin_idx - 1) >> LOG_SB_SIZE) + 1;
        if (begin_idx == 0)
            left_sb = 0;
        // round down to prev superblock
        index_t right_sb = end_idx >> LOG_SB_SIZE;

        // init result
        Iterator min_pos = _begin + begin_idx;

        // if there is at least one superblock
        if (left_sb < right_sb) {
            // get largest power of two that doesn't exceed the number of
            // superblocks from (left,right)
            index_t n_sb = right_sb - left_sb;
            unsigned int dist = floorlog2(n_sb);

            assert(dist < superblock_mins.size() && left_sb < superblock_mins[dist].size());
            min_pos = _begin + superblock_mins[dist][left_sb];
            assert(dist < superblock_mins.size() && right_sb - (1<<dist) < superblock_mins[dist].size());
            Iterator right_sb_min = _begin + superblock_mins[dist][right_sb - (1 << dist)];
            if (*right_sb_min < *min_pos) {
                min_pos = right_sb_min;
            }
        }

        // go to left -> blocks -> sub-block
        if (left_sb <= right_sb && left_sb != 0 && begin_idx != (left_sb << LOG_SB_SIZE)) {
            index_t left_b = ((begin_idx - 1) >> LOG_B_SIZE) + 1;
            index_t left_b_gidx = left_b << LOG_B_SIZE;
            left_b -= (left_sb - 1) << LOG_NB_PER_SB;
            index_t n_b = NB_PER_SB - left_b;
            if (n_b > 0) {
                unsigned int level = ceillog2(n_b);
                index_t sb_offset = (left_sb-1)*(NB_PER_SB - (1<<level)/2);
                Iterator block_min_it = _begin + block_mins[level][left_b + sb_offset] + ((left_sb-1)<<LOG_SB_SIZE);
                // return this new min if its the same or smaller
                if (!(*min_pos < *block_min_it))
                    min_pos = block_min_it;
            }

            // go left into remaining block, if elements left
            if (left_b_gidx > begin_idx) {
                // linearly search (at most block_size elements)
                Iterator inblock_min_it = std::min_element(_begin + begin_idx, _begin + left_b_gidx);
                if (!(*min_pos < *inblock_min_it)) {
                    min_pos = inblock_min_it;
                }
            }
        }

        // go to right -> blocks -> sub-block
        if (left_sb <= right_sb && right_sb != n_superblocks && end_idx != (right_sb << LOG_SB_SIZE)) {
            index_t left_b = right_sb << LOG_NB_PER_SB;
            index_t right_b = end_idx >> LOG_B_SIZE;
            index_t n_b = right_b - left_b;
            if (n_b > 0) {
                unsigned int dist = floorlog2(n_b);
                index_t sb_offset = right_sb*((1<<dist)/2);
                Iterator block_min_it = _begin + block_mins[dist][left_b - sb_offset] + ((right_sb) << LOG_SB_SIZE);
                if (*block_min_it < *min_pos)
                    min_pos = block_min_it;
                block_min_it = _begin + block_mins[dist][right_b - sb_offset - (1<<dist)] + ((right_sb) << LOG_SB_SIZE);
                if (*block_min_it < *min_pos)
                    min_pos = block_min_it;
            }

            // go right into remaining block, if elements left
            index_t left_gl_idx = right_b << LOG_B_SIZE;
            if (left_gl_idx < end_idx) {
                // linearly search (at most block_size elements)
                Iterator inblock_min_it = std::min_element(_begin + left_gl_idx, _begin + end_idx);
                if (*inblock_min_it < *min_pos) {
                    min_pos = inblock_min_it;
                }
            }
        }

        // if there are no superblocks covered (both indeces in same superblock)
        if (left_sb > right_sb) {
            index_t left_b = ((begin_idx - 1) >> LOG_B_SIZE) + 1;
            if (begin_idx == 0)
                left_b = 0;
            index_t right_b = end_idx >> LOG_B_SIZE;


            if (left_b < right_b) {
                // if blocks are in between: get mins of blocks in range
                // NOTE: there was a while if-else block here to handle the
                //       case if blocks would span accross the boundary of two
                //       superblocks, this should however never happen
                //       git blame this line to find where this code was removed
                // assert blocks lie in the same superblock
                assert(left_b >> LOG_NB_PER_SB == right_b >> LOG_NB_PER_SB);

                unsigned int dist = floorlog2(right_b - left_b);
                index_t sb_offset = 0;
                index_t sb_size_offset = 0;
                if (left_sb > 1) {
                    sb_offset = (left_sb-1)*((1<<dist)/2);
                    sb_size_offset = (left_sb-1) << LOG_SB_SIZE;
                }
                Iterator block_min_it = _begin + block_mins[dist][left_b - sb_offset] + sb_size_offset;
                if (*block_min_it < *min_pos)
                    min_pos = block_min_it;
                block_min_it = _begin + block_mins[dist][right_b - sb_offset - (1<<dist)] + sb_size_offset;
                if (*block_min_it < *min_pos)
                    min_pos = block_min_it;

                // remaining inblock
                if (begin_idx < (left_b << LOG_B_SIZE)) {
                    Iterator inblock_min_it = std::min_element(_begin + begin_idx, _begin + (left_b << LOG_B_SIZE));
                    if (!(*min_pos < *inblock_min_it)) {
                        min_pos = inblock_min_it;
                    }
                }
                if (end_idx > (right_b << LOG_B_SIZE)) {
                    Iterator inblock_min_it = std::min_element(_begin + (right_b << LOG_B_SIZE), _begin + end_idx);
                    if (*inblock_min_it < *min_pos) {
                        min_pos = inblock_min_it;
                    }
                }
            } else {
                // no blocks at all
                Iterator inblock_min_it = std::min_element(_begin + begin_idx, _begin + end_idx);
                if (*inblock_min_it < *min_pos) {
                    min_pos = inblock_min_it;
                }
            }

        }

        // return the minimum found
        return min_pos - _begin;
    }
}; // class my_rmq

#endif // my_rmq_HPP
