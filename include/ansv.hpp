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

#ifndef ANSV_HPP
#define ANSV_HPP

#include <vector>
#include <deque>
#include <mxx/comm.hpp>

// parallel all nearest smallest value
// TODO: iterator version??
// TODO: comparator type
// TODO: more compact via index_t template instead of size_t
template <typename T>
std::vector<size_t> ansv_index(const std::vector<T>& in, const mxx::comm& comm) {
    // first run ANSV left
    // then right, save all non-matched elements into vector as left_mins and right_mins

    // TODO: use deque or vector? -> check performance difference!!
    std::deque<std::pair<T,size_t> > q;
    size_t prefix = 0; // TODO: global prefix

    // forward direction (right minimal)
    for (size_t i = 0; i < in.size(); ++i) {
        while (!q.empty() && in[i] < q.back().first)
            q.pop_back();
        if (q.back().first < in[i]) // add only if not equal
            q.push_back(std::pair<T, size_t>(in[i], prefix+i));
    }
    // add results to right_mins
    std::vector<std::pair<T,size_t>> right_mins(q.rbegin(), q.rend());
    q.clear();
    // backwards direction
    for (size_t i = in.size(); i > 0; --i) {
        while (!q.empty() && in[i-1] < q.back().first)
            q.pop_back();
        if (q.back().first < in[i-1]) // add only if not equal
            q.push_back(std::pair<T, size_t>(in[i-1], prefix+i-1));
    }
    // add results to left_mins
    // TODO: only one of these should be reversed
    std::vector<std::pair<T,size_t>> left_mins(q.rbegin(), q.rend());
    q.clear();

    // allgather min and max of all left and right mins
    // calculate which processor to exchange unmatched left_mins and right_mins with
    // exchange via all2all_(inplace?) or send/recv?
    // solve locally given the exchanged values
    // TODO
}

#endif // ANSV_HPP
