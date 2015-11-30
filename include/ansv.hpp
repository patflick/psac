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
std::vector<size_t> ansv(const std::vector<T>& in, const mxx::comm& comm) {
    // first run ANSV left
    // then right, save all non-matched elements into vector as left_mins and right_mins

    // TODO: use deque or vector? -> check performance difference!!
    std::deque<std::pair<T,size_t> > q;
    size_t prefix = 0; // TODO: global prefix

    // backwards direction (left mins)
    for (size_t i = in.size(); i > 0; --i) {
        while (!q.empty() && in[i-1] < q.back().first)
            q.pop_back();
        if (q.back().first < in[i-1]) // add only if not equal
            q.push_back(std::pair<T, size_t>(in[i-1], prefix+i-1));
    }
    // add results to left_mins
    std::vector<std::pair<T,size_t>> lr_mins(q.begin(), q.end());
    size_t n_left_mins = q.size();
    assert(n_left_mins >= 1);
    q.clear();

    // forward direction (right mins)
    for (size_t i = 0; i < in.size(); ++i) {
        while (!q.empty() && in[i] < q.back().first)
            q.pop_back();
        if (q.back().first < in[i]) // add only if not equal
            q.push_back(std::pair<T, size_t>(in[i], prefix+i));
    }
    // add results to right_mins
    size_t n_right_mins = q.size();
    lr_mins.insert(lr_mins.end(), q.rbegin(), q.rend());
    assert(n_right_mins >= 1);
    q.clear();
    assert(n_right_mins + n_left_mins == lr_mins.size());


    // allgather min and max of all left and right mins
    T local_min = lr_mins.back();
    assert(local_min == lr_mins.front());
    std::vector<T> allmins = mxx::allgather(local_min, comm);

    std::vector<size_t> send_counts(comm.size(), 0);
    std::vector<size_t> send_displs(comm.size(), 0);

    // calculate which processor to exchange unmatched left_mins and right_mins with
    // 1) calculate communication parameters for left processes
    if (comm.rank() > 0) {
        size_t left_idx = n_left_mins-1;
        T prev_mins_min = allmins[comm.rank()-1];
        for (int i = comm.rank()-1; i >= 0; --i) {
            size_t start_idx = left_idx;
            while (left_idx > 0 && lr_mins[left_idx] >= allmins[i]) {
                --left_idx;
            }
            // don't send anything if min is dominated by previous
            if (allmins[i] > prev_mins_min) {
                continue;
            } else {
                prev_mins_min = allmins[i];
            }

            // send all from [left_idx, start_idx] (inclusive) to proc `i`
            send_counts[i] = start_idx - left_idx + 1;
            send_displs[i] = left_idx;

            // set comm parameters
            if (allmins[i] < local_min) {
                break;
            }
        }
    }

    // 2) calculate communication parameters for right process
    // if not last process
    if (comm.rank() < comm.size()-1) {
        size_t right_idx = n_left_mins;
        T prev_mins_min = allmins[comm.rank()+1];
        for (int i = comm.rank()+1; i < comm.size(); ++i) {
            // find the range which should be send
            size_t start_idx = right_idx;
            while (right_idx < lr_mins.size() && lr_mins[right_idx] >= allmins[i]) {
                ++right_idx;
            }

            // don't send if the minimum of `i` is dominated by a prev min right of me
            if (prev_mins_min < allmins[i]) {
                continue;
            } else {
                prev_mins_min = allmins[i];
            }

            // now right_idx is the first one that is smaller than the min of `i`
            // send all elements from [start_idx, right_idx]
            send_counts[i] = right_idx - start_idx + 1;
            send_displs[i] = start_idx;

            // can stop if an overall smaller min than my own has been found
            if (allmins[i] < local_min) {
                break;
            }
        }
    }

    // exchange lr_mins via all2all including overlaps!! (using custom all2all with custom displacements)
    // TODO: extend mxx for this case
    // TODO: and test if sending with overlap works!
    std::vector<size_t> recv_counts = mxx::all2all(send_counts, comm);
    std::vector<size_t> recv_displs = mxx::impl::get_displacements(recv_counts);
    size_t recv_size = recv_counts.back() + recv_displs.back();
    std::vector<std::pair<T, size_t>> recved(recv_size);
    mxx::all2allv(&lr_mins[0], send_counts, send_displs, &recved[0], recv_counts, recv_displs, comm);

    // exchange via all2all_(inplace?) or send/recv
    // solve locally given the exchanged values (by prefilling min-queue before running locally)
    // TODO
}

#endif // ANSV_HPP
