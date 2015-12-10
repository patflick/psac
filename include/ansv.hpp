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
#include <cxx-prettyprint/prettyprint.hpp>

#include <mxx/comm.hpp>
#include <suffix_array.hpp>

constexpr int nearest_sm = 0;
constexpr int nearest_eq = 1;
constexpr int furthest_eq = 2;


template <typename T, int type = nearest_sm>
inline void update_nsv_queue(std::vector<size_t>& nsv, std::deque<std::pair<T,size_t>>& q, const T& next_value, size_t idx, size_t prefix) {
    while (!q.empty() && next_value < q.back().first) {
        // current element is the min for in[i-1]
        nsv[q.back().second-prefix] = idx;
        if (type == furthest_eq) {
            // if that element is followed in the queue by equal elements,
            // set them to the furthest
            std::pair<T,size_t> furthest = q.back();
            q.pop_back();
            while (!q.empty() && furthest.first == q.back().first) {
                nsv[q.back().second-prefix] = furthest.second;
                q.pop_back();
            }
        } else {
            q.pop_back();
        }
    }
    if (type == nearest_eq) {
        if (!q.empty() && next_value == q.back().first) {
            // replace the equal element
            nsv[q.back().second-prefix] = idx;
            q.pop_back();
        }
    }
}


// parallel all nearest smallest value
// TODO: iterator version??
// TODO: comparator type
// TODO: more compact via index_t template instead of size_t
template <typename T, int left_type = nearest_sm, int right_type = nearest_sm>
void ansv(const std::vector<T>& in, std::vector<size_t>& left_nsv, std::vector<size_t>& right_nsv, std::vector<std::pair<T,size_t> >& lr_mins, const mxx::comm& comm) {
    std::deque<std::pair<T,size_t> > q;
    size_t local_size = in.size();
    size_t prefix = mxx::exscan(local_size, comm);

    // backwards direction (left mins)
    for (size_t i = in.size(); i > 0; --i) {
        while (!q.empty() && in[i-1] < q.back().first)
            q.pop_back();
        if (left_type == furthest_eq) {
            // remove all equal elements but the last
            while (q.size() >= 2 && in[i-1] == q.back().first && in[i-1] == (q.rbegin()+1)->first) {
                q.pop_back();
            }
        } else {
            while (!q.empty() && in[i-1] == q.back().first)
                q.pop_back();
        }
        q.push_back(std::pair<T, size_t>(in[i-1], prefix+i-1));
    }
    // add results to left_mins
    lr_mins = std::vector<std::pair<T,size_t>>(q.rbegin(), q.rend());
    size_t n_left_mins = q.size();
    assert(n_left_mins >= 1);
    q.clear();

    // forward direction (right mins)
    for (size_t i = 0; i < in.size(); ++i) {
        while (!q.empty() && in[i] < q.back().first)
            q.pop_back();
        if (right_type == furthest_eq) {
            while (q.size() >= 2 && in[i] == q.back().first && in[i] == (q.rbegin()+1)->first) {
                // remove all but the last one that is equal, TODO: can be `if` instead
                q.pop_back();
            }
        } else {
            while (!q.empty() && in[i] == q.back().first)
                q.pop_back();
        }
        q.push_back(std::pair<T, size_t>(in[i], prefix+i));
    }
    // add results to right_mins
    size_t n_right_mins = q.size();
    lr_mins.insert(lr_mins.end(), q.begin(), q.end());
    T local_min = q.front().first;
    assert(n_right_mins >= 1);
    q.clear();
    assert(n_right_mins + n_left_mins == lr_mins.size());


    // allgather min and max of all left and right mins
    //assert(local_min == lr_mins.front().first);
    std::vector<T> allmins = mxx::allgather(local_min, comm);
    std::vector<size_t> send_counts(comm.size(), 0);
    std::vector<size_t> send_displs(comm.size(), 0);

    // calculate which processor to exchange unmatched left_mins and right_mins with
    // 1) calculate communication parameters for left processes
    if (comm.rank() > 0) {
        size_t left_idx = 0;
        T prev_mins_min = allmins[comm.rank()-1];
        for (int i = comm.rank()-1; i >= 0; --i) {
            size_t start_idx = left_idx;
            // move start back to other `equal` elements (there can be at most 2)
            if (right_type == furthest_eq && i < comm.rank()-1) {
                if (start_idx > 0 && lr_mins[start_idx-1].first <= allmins[i+1]) {
                    --start_idx;
                }
            }
            while (left_idx+1 < n_left_mins && lr_mins[left_idx].first >= allmins[i]) {
                ++left_idx;
            }
            // don't send anything if min is dominated by previous
            if (allmins[i] > prev_mins_min) {
                continue;
            } else {
                prev_mins_min = allmins[i];
            }

            // send all from [left_idx, start_idx] (inclusive) to proc `i`
            send_counts[i] = left_idx - start_idx + 1;
            send_displs[i] = start_idx;

            // set comm parameters
            if (allmins[i] < local_min) {
                break;
            }
        }
    }

    // 2) calculate communication parameters for right process
    if (comm.rank() < comm.size()-1) {
        size_t right_idx = lr_mins.size()-1; // n_left_mins;
        T prev_mins_min = allmins[comm.rank()+1];
        for (int i = comm.rank()+1; i < comm.size(); ++i) {
            // find the range which should be send
            size_t end_idx = right_idx;
            if (left_type == furthest_eq && i > comm.rank()+1) {
                // move start back to other `equal` elements
                while (end_idx+1 < lr_mins.size() && lr_mins[end_idx+1].first <= allmins[i-1]) {
                    ++end_idx;
                }
            }
            while (right_idx > n_left_mins && lr_mins[right_idx].first >= allmins[i]) {
                --right_idx;
            }

            // don't send if the minimum of `i` is dominated by a prev min right of me
            if (prev_mins_min < allmins[i]) {
                continue;
            } else {
                prev_mins_min = allmins[i];
            }

            // now right_idx is the first one that is smaller than the min of `i`
            // send all elements from [start_idx, right_idx]
            send_counts[i] = end_idx - right_idx + 1;
            send_displs[i] = right_idx;

            // can stop if an overall smaller min than my own has been found
            if (allmins[i] < local_min) {
                break;
            }
        }
    }

    // exchange lr_mins via all2all
    std::vector<size_t> recv_counts = mxx::all2all(send_counts, comm);
    std::vector<size_t> recv_displs = mxx::impl::get_displacements(recv_counts);
    size_t recv_size = recv_counts.back() + recv_displs.back();
    std::vector<std::pair<T, size_t>> recved(recv_size);
    mxx::all2allv(&lr_mins[0], send_counts, send_displs, &recved[0], recv_counts, recv_displs, comm);


    // solve locally given the exchanged values (by prefilling min-queue before running locally)
    size_t n_left_recv = recv_displs[comm.rank()];
    size_t n_right_recv = recv_size - n_left_recv;

    left_nsv.resize(local_size);
    right_nsv.resize(local_size);

    q.clear();
    // iterate backwards to get the nearest smaller element to left for each element
    // TODO: this doesn't really require pairs in the queue (index suffices)
    for (size_t i = in.size(); i > 0; --i) {
        update_nsv_queue<T,left_type>(left_nsv, q, in[i-1], prefix+i-1, prefix);
        q.push_back(std::pair<T, size_t>(in[i-1], prefix+i-1));
    }

    // now go backwards through the right-mins from the previous processors,
    // in order to resolve all local elements
    for (size_t i = 0; i < n_left_recv; ++i) {
        update_nsv_queue<T,left_type>(left_nsv, q, recved[n_left_recv - i - 1].first, recved[n_left_recv - i - 1].second, prefix);
    }
    // TODO: elements still in the queue do not have a smaller value to the left
    //       -> set these to a special value?

    // iterate forwards to get the nearest smaller value to the right for each element
    q.clear();
    for (size_t i = 0; i < in.size(); ++i) {
        update_nsv_queue<T,right_type>(right_nsv, q, in[i], prefix+i, prefix);
        q.push_back(std::pair<T, size_t>(in[i], prefix+i));
    }

    // now go forwards through left-mins of succeeding processors
    for (size_t i = 0; i < n_right_recv; ++i) {
        update_nsv_queue<T,right_type>(right_nsv, q, recved[n_left_recv + i].first, recved[n_left_recv+i].second, prefix);
    }
    // TODO: elements still in the queue do not have a smaller value to the right
    //       -> set these to a special value?

    lr_mins = recved;
}

template <typename T, int left_type = nearest_sm, int right_type = nearest_sm>
void ansv(const std::vector<T>& in, std::vector<size_t>& left_nsv, std::vector<size_t>& right_nsv, const mxx::comm& comm) {
    std::vector<std::pair<T, size_t>> lr_mins;
    ansv<T, left_type, right_type>(in, left_nsv, right_nsv, lr_mins, comm);
}

template <typename InputIterator, typename index_t = std::size_t>
void construct_suffix_tree(const suffix_array<InputIterator, index_t, true>& sa, const mxx::comm& comm) {
    // ansv of lcp!
    // TODO: use index_t instead of size_t
    std::vector<size_t> left_nsv;
    std::vector<size_t> right_nsv;
    std::vector<std::pair<index_t, size_t>> lr_mins;

    // ANSV with furthest eq for left and smallest for right
    // TODO: also return the lr_mins
    // TODO: different indexing for lr_mins !!!
    ansv<index_t, furthest_eq, nearest_sm>(sa.LCP, left_nsv, right_nsv, lr_mins, comm);

    // TODO:
    // - get max of ansvs as the `parent` node (requires lr_mins for non local values)
    // - get character for internal nodes (first character of suffix starting at
    //   the node S[SA[i]+LCP[i]]) [i.e. requesting a full permutation] needed for
    //   all edges, i.e. up to 2n-1 (SA and LCP aligned)
    // - TODO: howto efficiently represent edges while string remains distributed??
}

template <typename T>
void ansv_sequential(const std::vector<T>& in, std::vector<size_t>& left_nsv, std::vector<size_t>& right_nsv) {
    // resize outputs
    left_nsv.resize(in.size());
    right_nsv.resize(in.size());

    std::deque<size_t> q;
    // iterate backwards to get the nearest smaller element to left for each element
    for (size_t i = in.size(); i > 0; --i) {
        while (!q.empty() && in[i-1] < in[q.back()]) {
            // current element is the min for in[i-1]
            left_nsv[q.back()] = i-1;
            q.pop_back();
        }
        // TODO: potentially handle `equal` elements differently
        q.push_back(i-1);
    }
    // iterate forwards to get the nearest smaller value to the right for each element
    q.clear();
    for (size_t i = 0; i < in.size(); ++i) {
        while (!q.empty() && in[i] < in[q.back()]) {
            // current element is the min for in[i-1]
            right_nsv[q.back()] = i;
            q.pop_back();
        }
        // TODO: potentially handle `equal` elements differently
        q.push_back(i);
    }
}

#endif // ANSV_HPP
