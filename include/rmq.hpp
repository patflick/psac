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
 * @brief   Implements the Range-Minimum-Query in a STL compatible format.
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
    // we choose it as bitsize*2:
    //  - 64 bits -> 4096
    //  - 32 bits -> 1024
    //  - both bound by 2^16 -> uint16_t
    //  block size: one cache line
    const static int DEFAULT_BLOCK_SIZE = 64/sizeof(index_t);
    // cache line of blocks per superblock
    const static int DEFAULT_SUPERBLOCK_SIZE = 64/sizeof(uint16_t) * 64/sizeof(index_t);
protected:
    index_t n;

    // the original sequence
    Iterator _begin;
    Iterator _end;
    // level sizes
    index_t superblock_size;
    index_t block_size;

    // number of blocks per level
    index_t n_blocks;
    index_t n_superblocks;
    index_t n_blocks_per_superblock;

    // saves minimum as block index for combination of superblocks relative to
    // global start index
    std::vector<std::vector<index_t> > superblock_mins;
    // saves minimum for combination of blocks relative to superblock start
    // index
    std::vector<std::vector<uint16_t> > block_mins;



public:
    rmq(Iterator begin, Iterator end,
        index_t superblock_size = DEFAULT_SUPERBLOCK_SIZE,
        index_t block_size = DEFAULT_BLOCK_SIZE)
        : _begin(begin), _end(end), superblock_size(superblock_size), block_size(block_size)
    {
        // get size
        assert((index_t)std::distance(begin, end) < std::numeric_limits<index_t>::max());
        n = std::distance(begin, end);


        // superblock size should be divisable by block size
        assert(superblock_size % block_size == 0);

        // get number of blocks
        n_superblocks = (n-1) / superblock_size + 1;
        n_blocks = (n-1) / block_size + 1;
        n_blocks_per_superblock = superblock_size / block_size;

        assert(superblock_size-1 <= std::numeric_limits<uint16_t>::max());


        // start by finding index of mins per block
        // for each superblock
        superblock_mins.push_back(std::vector<index_t>(n_superblocks));
        block_mins.push_back(std::vector<uint16_t>(n_blocks));
        Iterator it = begin;
        while (it != end)
        {
            // find index of minimum block in superblock
            Iterator min_pos = it;
            Iterator sb_end_it = it;
            std::advance(sb_end_it, std::min<std::size_t>(std::distance(it, end), superblock_size));
            for (Iterator block_it = it; block_it != sb_end_it; )
            {
                Iterator block_end_it = block_it;
                std::advance(block_end_it, std::min<std::size_t>(std::distance(block_it, end), block_size));
                Iterator block_min_pos = std::min_element(block_it, block_end_it);
                // save minimum for superblock min
                if (*block_min_pos < *min_pos)
                {
                    min_pos = block_min_pos;
                }
                // save minimum for block min, relative to superblock start
                index_t block_min_idx = static_cast<index_t>(std::distance(it, block_min_pos));
                assert(block_min_idx < superblock_size);
                block_mins[0][std::distance(begin, block_it) / block_size] = static_cast<uint16_t>(block_min_idx);
                block_it = block_end_it;
            }
            superblock_mins[0][std::distance(begin, it) / superblock_size] = static_cast<index_t>(std::distance(begin, min_pos));
            it = sb_end_it;
        }

        // fill superblock lookup with dynamic programming
        index_t level = 1;
        for (index_t dist = 2; dist/2 < n_superblocks; dist <<= 1)
        {
            superblock_mins.push_back(std::vector<index_t>(n_superblocks - dist/2));
            for (index_t i = 0; i+dist/2 < n_superblocks; ++i)
            {
                index_t right_idx = std::min(i+dist/2, n_superblocks-dist/4-1);
                if (*(begin + superblock_mins[level-1][i]) < *(begin + superblock_mins[level-1][right_idx]))
                {
                    assert(i < superblock_mins.back().size());
                    assert(superblock_mins.size() == level+1);
                    superblock_mins.back()[i] = superblock_mins[level-1][i];
                }
                else
                {
                    assert(i < superblock_mins.back().size());
                    assert(superblock_mins.size() == level+1);
                    assert(superblock_mins[level-1].size() > right_idx);
                    superblock_mins.back()[i] = superblock_mins[level-1][right_idx];
                }
            }
            level++;
        }

        // now the same thing for blocks (but index relative to their
        // superblock)
        level = 1;
        index_t last_sb_nblocks = n_blocks - (n_superblocks-1)*n_blocks_per_superblock;
        for (index_t dist = 2; dist/2 < std::min(n_blocks,n_blocks_per_superblock); dist <<= 1)
        {
            if (n_blocks - n_superblocks*dist/2 == 0)
                break;
            index_t last_sb_cur_nblocks = 0;
            if (last_sb_nblocks > dist/2)
                last_sb_cur_nblocks = last_sb_nblocks - dist/2;
            block_mins.push_back(std::vector<uint16_t>((n_superblocks-1)*(n_blocks_per_superblock - dist/2) + last_sb_cur_nblocks));
            for (index_t sb = 0; sb < n_superblocks; ++sb)
            {
                index_t pre_sb_offset = sb*(n_blocks_per_superblock - dist/4);
                index_t sb_offset = sb*(n_blocks_per_superblock - dist/2);
                index_t blocks_in_sb = std::min(n_blocks - sb*n_blocks_per_superblock, n_blocks_per_superblock);
                for (index_t i = 0; i+dist/2 < blocks_in_sb; ++i)
                {
                    // TODO: right_idx might become negative for last superblock??
                    index_t right_idx = std::min(i + dist/2, blocks_in_sb-dist/4-1);
                    if (*(begin + block_mins[level-1][i        +pre_sb_offset] + sb*superblock_size)
                      < *(begin + block_mins[level-1][right_idx+pre_sb_offset] + sb*superblock_size))
                    {
                        block_mins.back()[i+sb_offset] = block_mins[level-1][i+pre_sb_offset];
                    } else {
                        block_mins.back()[i+sb_offset] = block_mins[level-1][right_idx+pre_sb_offset];
                    }
                }
            }
            level++;
        }

        //assert(check_superblock_correctness());
        //assert(check_block_correctness());
    }

    Iterator query(Iterator range_begin, Iterator range_end)
    {
        // find superblocks fully contained within range
        index_t begin_idx = std::distance(_begin, range_begin);
        index_t end_idx = std::distance(_begin, range_end);
        assert(end_idx <= n);

        // round up to next superblock
        index_t left_sb  = (begin_idx - 1) / superblock_size + 1;
        if (begin_idx == 0)
            left_sb = 0;
        // round down to prev superblock
        index_t right_sb = end_idx / superblock_size;

        // init result
        Iterator min_pos = range_begin;

        // if there is at least one superblock
        if (left_sb < right_sb)
        {
            // get largest power of two that doesn't exceed the number of
            // superblocks from (left,right)
            index_t n_sb = right_sb - left_sb;
            unsigned int dist = floorlog2(n_sb);

            assert(dist < superblock_mins.size() && left_sb < superblock_mins[dist].size());
            min_pos = _begin + superblock_mins[dist][left_sb];
            assert(dist < superblock_mins.size() && right_sb - (1<<dist) < superblock_mins[dist].size());
            Iterator right_sb_min = _begin + superblock_mins[dist][right_sb - (1 << dist)];
            if (*min_pos > *right_sb_min)
            {
                min_pos = right_sb_min;
            }
        }

        // go to left -> blocks -> sub-block
        if (left_sb <= right_sb && left_sb != 0 && begin_idx != left_sb*superblock_size)
        {
            index_t left_b = (begin_idx - 1) / block_size + 1;
            index_t left_b_gidx = left_b * block_size;
            left_b -= (left_sb - 1)*n_blocks_per_superblock;
            index_t n_b = n_blocks_per_superblock - left_b;
            if (n_b > 0)
            {
                unsigned int level = ceillog2(n_b);
                index_t sb_offset = (left_sb-1)*(n_blocks_per_superblock - (1<<level)/2);
                Iterator block_min_it = _begin + block_mins[level][left_b + sb_offset] + (left_sb-1)*superblock_size;
                if (*block_min_it < *min_pos)
                    min_pos = block_min_it;
            }

            // go left into remaining block, if elements left
            if (left_b_gidx > begin_idx)
            {
                // linearly search (at most block_size elements)
                Iterator inblock_min_it = std::min_element(range_begin, _begin + left_b_gidx);
                if (*inblock_min_it < *min_pos)
                {
                    min_pos = inblock_min_it;
                }
            }
        }

        // go to right -> blocks -> sub-block
        if (left_sb <= right_sb && right_sb != n_superblocks && end_idx != right_sb*superblock_size)
        {
            index_t left_b = right_sb*n_blocks_per_superblock;
            index_t right_b = end_idx / block_size;
            index_t n_b = right_b - left_b;
            if (n_b > 0)
            {
                unsigned int dist = floorlog2(n_b);
                index_t sb_offset = right_sb*((1<<dist)/2);
                Iterator block_min_it = _begin + block_mins[dist][left_b - sb_offset] + (right_sb)*superblock_size;
                if (*block_min_it < *min_pos)
                    min_pos = block_min_it;
                block_min_it = _begin + block_mins[dist][right_b - sb_offset - (1<<dist)] + (right_sb)*superblock_size;
                if (*block_min_it < *min_pos)
                    min_pos = block_min_it;
            }

            // go right into remaining block, if elements left
            index_t left_gl_idx = right_b*block_size;
            if (left_gl_idx < end_idx)
            {
                // linearly search (at most block_size elements)
                Iterator inblock_min_it = std::min_element(_begin + left_gl_idx, range_end);
                if (*inblock_min_it < *min_pos)
                {
                    min_pos = inblock_min_it;
                }
            }
        }

        // if there are no superblocks covered:
        if (left_sb > right_sb)
        {
            index_t left_b = (begin_idx - 1) / block_size + 1;
            index_t right_b = end_idx / block_size;

            if (left_b < right_b)
            {
                // there's blocks in between
                if (left_b / n_blocks_per_superblock < right_b / n_blocks_per_superblock)
                {
                    // block span accross boundary of a superblock
                    index_t mid_b = (right_b / n_blocks_per_superblock) * n_blocks_per_superblock;

                    // left part
                    if (left_b < mid_b)
                    {
                        unsigned int left_dist = ceillog2(mid_b - left_b);
                        index_t sb_offset = 0;
                        index_t sb_size_offset = 0;
                        if (left_sb > 1)
                        {
                            sb_offset = (left_sb-1)*((1<<left_dist)/2);
                            sb_size_offset = (left_sb-1)*superblock_size;
                        }
                        Iterator block_min_it = _begin + block_mins[left_dist][left_b - sb_offset] + sb_size_offset;
                        if (*block_min_it < *min_pos)
                            min_pos = block_min_it;
                    }

                    if (mid_b < right_b)
                    {
                        // right part
                        unsigned int right_dist = floorlog2(right_b - mid_b);
                        index_t sb_offset = (1<<right_dist)/2;
                        index_t sb_size_offset = superblock_size;
                        if (left_sb > 1)
                        {
                            sb_offset += (left_sb-1)*((1<<right_dist)/2);
                            sb_size_offset += (left_sb-1)*superblock_size;
                        }
                        Iterator block_min_it = _begin + block_mins[right_dist][mid_b - sb_offset] + sb_size_offset;
                        if (*block_min_it < *min_pos)
                            min_pos = block_min_it;
                        block_min_it = _begin + block_mins[right_dist][right_b - sb_offset - (1<<right_dist)] + sb_size_offset;
                        if (*block_min_it < *min_pos)
                            min_pos = block_min_it;
                    }
                }
                else
                {
                    unsigned int dist = floorlog2(right_b - left_b);
                    index_t sb_offset = 0;
                    index_t sb_size_offset = 0;
                    if (left_sb > 1)
                    {
                        sb_offset = (left_sb-1)*((1<<dist)/2);
                        sb_size_offset = (left_sb-1)*superblock_size;
                    }
                    Iterator block_min_it = _begin + block_mins[dist][left_b - sb_offset] + sb_size_offset;
                    if (*block_min_it < *min_pos)
                        min_pos = block_min_it;
                    block_min_it = _begin + block_mins[dist][right_b - sb_offset - (1<<dist)] + sb_size_offset;
                    if (*block_min_it < *min_pos)
                        min_pos = block_min_it;

                    // remaining inblock
                    if (begin_idx < left_b*block_size)
                    {
                        Iterator inblock_min_it = std::min_element(range_begin, _begin + left_b*block_size);
                        if (*inblock_min_it < *min_pos)
                        {
                            min_pos = inblock_min_it;
                        }
                    }
                    if (end_idx > right_b*block_size)
                    {
                        Iterator inblock_min_it = std::min_element(_begin + right_b*block_size, range_end);
                        if (*inblock_min_it < *min_pos)
                        {
                            min_pos = inblock_min_it;
                        }
                    }
                }
            }
            else
            {
                // no blocks at all
                Iterator inblock_min_it = std::min_element(range_begin, range_end);
                if (*inblock_min_it < *min_pos)
                {
                    min_pos = inblock_min_it;
                }
            }

        }

        // return the minimum found
        return min_pos;
    }
}; // class rmq

#endif // RMQ_HPP
