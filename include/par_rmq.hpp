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
 * @file    par_rmq.hpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements parallel, bulk Range-Minimum-Query.
 */

#ifndef PARALLEL_BULK_RMQ_HPP
#define PARALLEL_BULK_RMQ_HPP

#include <vector>
#include <cstdint>

//#include "mpi_utils.hpp"
#include <mxx/datatypes.hpp>
#include <mxx/partition.hpp>
#include <mxx/collective.hpp>

#include "rmq.hpp"


template <typename index_t>
void bulk_rmq(const std::size_t n, const std::vector<index_t>& local_els,
              std::vector<std::tuple<index_t, index_t, index_t>>& ranges,
              const mxx::comm& comm) {
    // 3.) bulk-parallel-distributed RMQ for ranges (B2[i-1],B2[i]+1) to get min_lcp[i]
    //     a.) allgather per processor minimum lcp
    //     b.) create messages to left-most and right-most processor (i, B2[i])
    //     c.) on rank==B2[i]: get either left-flank min (i < B2[i])
    //         or right-flank min (i > B2[i])
    //         -> in bulk by ordering messages into two groups and sorting by B2[i]
    //            then linear scan for min starting from left-most/right-most element
    //     d.) answer with message (i, min(...))
    //     e.) on rank==i: receive min answer from left-most and right-most
    //         processor for each range
    //          -> sort by i
    //     f.) for each range: get RMQ of intermediary processors
    //                         min(min_p_left, RMQ(p_left+1,p_right-1), min_p_right)

    mxx::partition::block_decomposition_buffered<index_t> part(n, comm.size(), comm.rank());

    // get size parameters
    std::size_t local_size = local_els.size();
    std::size_t prefix_size = part.excl_prefix_size();

    // create RMQ for local elements
    rmq<typename std::vector<index_t>::const_iterator> local_rmq(local_els.begin(), local_els.end());

    // get per processor minimum and it's position
    auto local_min_it = local_rmq.query(local_els.begin(), local_els.end());
    index_t min_pos = (local_min_it - local_els.begin()) + prefix_size;
    index_t local_min = *local_min_it;
    //assert(local_min == *std::min_element(local_els.begin(), local_els.end()));
    std::vector<index_t> proc_mins = mxx::allgather(local_min, comm);
    std::vector<index_t> proc_min_pos = mxx::allgather(min_pos, comm);

    // create range-minimum-query datastructure for the processor minimums
    rmq<typename std::vector<index_t>::iterator> proc_mins_rmq(proc_mins.begin(), proc_mins.end());

    // 1.) duplicate vector of triplets
    // 2.) (asserting that t[1] < t[2]) send to target processor for t[1]
    // 3.) on t[1]: return min and min index of right flank
    // 4.) send duplicated to t[2]
    // 5.) on t[2]: return min and min index of left flank
    // 6.) if (target-p(t[1]) == target-p(t[2])):
    //      a) target-p(t[1]) participates in twice (2x too much, could later be
    //         algorithmically optimized away)
    //         and returns min only from range (t[1],t[2])
    // 7.) on i: sort both by i, then pairwise iterate through and get target-p
    //     for both min indeces. if there are p's in between: use proc_min_rmq
    //     to get value and index of min p, then get global min index of that
    //     processor (needs to be gotten as well during the allgather)
    //  TODO: maybe template value type as different type than index_type
    //        (right now has to be identical, which suffices for LCP)

    std::vector<std::tuple<index_t, index_t, index_t> > ranges_right(ranges);

    // first communication
    mxx::all2all_func(
        ranges,
        [&](const std::tuple<index_t, index_t, index_t>& x) {
            return part.target_processor(std::get<1>(x));
        }, comm);
    // find mins from start to right border locally
    for (auto it = ranges.begin(); it != ranges.end(); ++it)
    {
        assert(std::get<1>(*it) < std::get<2>(*it));
        auto range_begin = local_els.begin() + (std::get<1>(*it) - prefix_size);
        auto range_end = local_els.end();
        if (std::get<2>(*it) < prefix_size + local_size)
        {
            range_end = local_els.begin() + (std::get<2>(*it) - prefix_size);
        }
        auto range_min = local_rmq.query(range_begin, range_end);
        std::get<1>(*it) = (range_min - local_els.begin()) + prefix_size;
        std::get<2>(*it) = *range_min;
    }
    // send results back to originator
    mxx::all2all_func(
        ranges,
        [&](const std::tuple<index_t, index_t, index_t>& x) {
            return part.target_processor(std::get<0>(x));
        }, comm);

    // second communication
    mxx::all2all_func(
        ranges_right,
        [&](const std::tuple<index_t, index_t, index_t>& x) {
            return part.target_processor(std::get<2>(x)-1);
        }, comm);
    // find mins from start to right border locally
    for (auto it = ranges_right.begin(); it != ranges_right.end(); ++it)
    {
        assert(std::get<1>(*it) < std::get<2>(*it));
        auto range_end = local_els.begin() + (std::get<2>(*it) - prefix_size);
        auto range_begin = local_els.begin();
        if (std::get<1>(*it) >= prefix_size)
        {
            range_begin = local_els.begin() + (std::get<1>(*it) - prefix_size);
        }
        auto range_min = local_rmq.query(range_begin, range_end);
        std::get<1>(*it) = (range_min - local_els.begin()) + prefix_size;
        std::get<2>(*it) = *range_min;
    }
    // send results back to originator
    mxx::all2all_func(
        ranges_right,
        [&](const std::tuple<index_t, index_t, index_t>& x) {
            return part.target_processor(std::get<0>(x));
        }, comm);

    // get total min and save into ranges
    assert(ranges.size() == ranges_right.size());

    // sort both by target index
    std::sort(ranges.begin(), ranges.end(),
              [](const std::tuple<index_t, index_t, index_t>& x,
                 const std::tuple<index_t, index_t, index_t>& y) {
        return std::get<0>(x) < std::get<0>(y);
    });
    std::sort(ranges_right.begin(), ranges_right.end(),
              [](const std::tuple<index_t, index_t, index_t>& x,
                 const std::tuple<index_t, index_t, index_t>& y) {
        return std::get<0>(x) < std::get<0>(y);
    });

    // iterate through both results (left and right)
    // and get overall minimum
    for (std::size_t i=0; i < ranges.size(); ++i)
    {
        assert(std::get<0>(ranges[i]) == std::get<0>(ranges_right[i]));
        std::size_t left_min_idx = std::get<1>(ranges[i]);
        std::size_t right_min_idx = std::get<1>(ranges_right[i]);
        // get min of both ranges
        if (std::get<2>(ranges[i]) > std::get<2>(ranges_right[i]))
        {
            ranges[i] = ranges_right[i];
        }

        // if the answer is different
        if (left_min_idx != right_min_idx)
        {
            int p_left = part.target_processor(left_min_idx);
            int p_right = part.target_processor(right_min_idx);
            // get minimum of both elements
            if (p_left + 1 < p_right)
            {
                // get min from per process min RMQ
                auto p_min_it =
                    proc_mins_rmq.query(proc_mins.begin() + p_left + 1,
                                        proc_mins.begin() + p_right);
                index_t p_min = *p_min_it;
                // if smaller, save as overall minimum
                if (p_min < std::get<2>(ranges[i]))
                {
                    std::get<1>(ranges[i]) = proc_min_pos[p_min_it - proc_mins.begin()];
                    std::get<2>(ranges[i]) = p_min;
                }
            }
        }
    }
}

#endif // PARALLEL_BULK_RMQ_HPP
