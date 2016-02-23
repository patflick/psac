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

constexpr int global_indexing = 0;
constexpr int local_indexing = 1;


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
template <typename T, int left_type = nearest_sm, int right_type = nearest_sm, int indexing_type = global_indexing>
void ansv(const std::vector<T>& in, std::vector<size_t>& left_nsv, std::vector<size_t>& right_nsv, std::vector<std::pair<T,size_t> >& lr_mins, const mxx::comm& comm, size_t nonsv = 0) {
    
    mxx::section_timer t;

    std::deque<std::pair<T,size_t> > q;
    size_t local_size = in.size();
    size_t prefix = mxx::exscan(local_size, comm);

    /*****************************************************************
     *  Step 1: Locally calculate ANSV and save un-matched elements  *
     *****************************************************************/

    // backwards direction (left mins)
    for (size_t i = in.size(); i > 0; --i) {
        while (!q.empty() && in[i-1] < q.back().first)
            q.pop_back();
        if (right_type == furthest_eq) {
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
        if (left_type == furthest_eq) {
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
    MXX_ASSERT(n_right_mins >= 1);
    q.clear();
    if (n_right_mins + n_left_mins != lr_mins.size()) {
        MXX_ASSERT(false);
    }

    t.end_section("ANSV: local ansv");

    /***************************************************************
     *  Step 2: communicate un-matched elements to correct target  *
     ***************************************************************/


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
                if (start_idx > 0 && lr_mins[start_idx-1].first <= prev_mins_min) {
                    --start_idx;
                }
            }
            while (left_idx+1 < n_left_mins && lr_mins[left_idx].first >= allmins[i]) {
                ++left_idx;
            }
            if (right_type == furthest_eq) {
                while (left_idx+1 < n_left_mins && lr_mins[left_idx].first == lr_mins[left_idx+1].first) {
                    ++left_idx;
                }
            }
            // remember most min we have seen so far
            if (allmins[i] < prev_mins_min) {
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
                while (end_idx+1 < lr_mins.size() && lr_mins[end_idx+1].first <= prev_mins_min) {
                    ++end_idx;
                }
            }
            while (right_idx > n_left_mins && lr_mins[right_idx].first >= allmins[i]) {
                --right_idx;
            }
            if (left_type == furthest_eq) {
                while (right_idx > n_left_mins && lr_mins[right_idx].first == lr_mins[right_idx-1].first) {
                    --right_idx;
                }
            }

            // remember most min we have seen so far
            if (prev_mins_min > allmins[i]) {
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

    t.end_section("ANSV: calc comm params");
    mxx::all2allv(&lr_mins[0], send_counts, send_displs, &recved[0], recv_counts, recv_displs, comm);
    t.end_section("ANSV: all2all");

    // solve locally given the exchanged values (by prefilling min-queue before running locally)
    size_t n_left_recv = recv_displs[comm.rank()];
    size_t n_right_recv = recv_size - n_left_recv;


    /***************************************************************
     *  Step 3: Again solve ANSV locally and use lr_mins as tails  *
     ***************************************************************/

    left_nsv.resize(local_size);
    right_nsv.resize(local_size);

    q.clear();
    const size_t iprefix = (indexing_type == global_indexing) ? prefix : 0;

    if (left_type == furthest_eq) {
        // iterate forwards through the received elements to fill the queue
        for (size_t i = 0; i < n_left_recv; ++i) {
            size_t rcv_idx = i;
            // TODO: I should be able to assume that the sequence is non-decreasing
            while (!q.empty() && recved[rcv_idx].first < q.back().first) {
                q.pop_back();
            }
            if (q.empty() || recved[rcv_idx].first > q.back().first) {
                // push only if this element is larger (not equal)
                if (indexing_type == global_indexing) {
                    q.push_back(recved[rcv_idx]);
                } else {
                    q.push_back(std::pair<T, size_t>(recved[rcv_idx].first, local_size + rcv_idx));
                }
            }
        }

        // iterate forward through the local items and set their nsv
        // to the first equal or smaller element in the queue
        for (size_t i = 0; i < in.size(); ++i) {
            while (!q.empty() && in[i] < q.back().first) {
                q.pop_back();
            }
            if (q.empty()) {
                left_nsv[i] = nonsv;
            } else {
                left_nsv[i] = q.back().second;
            }
            if (q.empty() || in[i] > q.back().first) {
                q.push_back(std::pair<T, size_t>(in[i], iprefix+i));
            }
        }
    } else {

        // iterate backwards to get the nearest smaller element to left for each element
        for (size_t i = in.size(); i > 0; --i) {
            if (indexing_type == global_indexing) {
                update_nsv_queue<T,left_type>(left_nsv, q, in[i-1], prefix+i-1, prefix);
                q.push_back(std::pair<T, size_t>(in[i-1], prefix+i-1));
            } else { // indexing_type == local_indexing
                // prefix=0 for local indexing
                update_nsv_queue<T,left_type>(left_nsv, q, in[i-1], i-1, 0);
                q.push_back(std::pair<T, size_t>(in[i-1], i-1));
            }
        }

        // now go backwards through the right-mins from the previous processors,
        // in order to resolve all local elements
        for (size_t i = 0; i < n_left_recv; ++i) {
            if (q.empty()) {
                break;
            }
            size_t rcv_idx = n_left_recv - i - 1;

            // set nsv for all larger elements in the queue
            if (indexing_type == global_indexing) {
                update_nsv_queue<T,left_type>(left_nsv, q, recved[rcv_idx].first, recved[rcv_idx].second, prefix);
            } else { // indexing_type == local_indexing
                // prefix = 0, recv.2nd = local_size+rcv_idx,
                update_nsv_queue<T,left_type>(left_nsv, q, recved[rcv_idx].first, local_size + rcv_idx, 0);
            }
        }

        // elements still in the queue do not have a smaller value to the left
        //  -> set these to a special value and handle case if furthest_eq
        //     elements are still waiting in queue
        for (auto it = q.rbegin(); it != q.rend(); ++it) {
            left_nsv[it->second - iprefix] = nonsv;
        }

    }

    // iterate forwards to get the nearest smaller value to the right for each element
    q.clear();

    if (right_type == furthest_eq) {
        // iterate backwards through the received elements to fill the queue
        for (size_t i = 0; i < n_right_recv; ++i) {
            //size_t rcv_idx = n_left_recv - i - 1;
            size_t rcv_idx = recved.size() - i - 1;
            while (!q.empty() && recved[rcv_idx].first < q.back().first) {
                q.pop_back();
            }
            if (q.empty() || recved[rcv_idx].first > q.back().first) {
                // push only if this element is larger (not equal)
                if (indexing_type == global_indexing) {
                    q.push_back(recved[rcv_idx]);
                } else {
                    q.push_back(std::pair<T, size_t>(recved[rcv_idx].first, local_size + rcv_idx));
                }
            }
        }
        // iterate backwards through the local items and set their nsv
        // to the first equal or smaller element in the queue
        for (size_t idx = 0; idx < in.size(); ++idx) {
            size_t i = in.size() - idx - 1;
            while (!q.empty() && in[i] < q.back().first) {
                q.pop_back();
            }
            if (q.empty()) {
                right_nsv[i] = nonsv;
            } else {
                right_nsv[i] = q.back().second;
            }
            if (q.empty() || in[i] > q.back().first) {
                q.push_back(std::pair<T, size_t>(in[i], iprefix+i));
            }
        }
    } else {
        for (size_t i = 0; i < in.size(); ++i) {
            if (indexing_type == global_indexing) {
                update_nsv_queue<T,right_type>(right_nsv, q, in[i], prefix+i, prefix);
                q.push_back(std::pair<T, size_t>(in[i], prefix+i));
            } else { // indexing_type == local_indexing
                // handle as if prefix = 0
                update_nsv_queue<T,right_type>(right_nsv, q, in[i], i, 0);
                q.push_back(std::pair<T, size_t>(in[i], i));
            }
        }

        // now go forwards through left-mins of succeeding processors
        for (size_t i = 0; i < n_right_recv; ++i) {
            size_t rcv_idx = n_left_recv + i;
            if (q.empty()) {
                break;
            }
            // set nsv for all larger elements in the queue
            if (indexing_type == global_indexing) {
                update_nsv_queue<T,right_type>(right_nsv, q, recved[rcv_idx].first, recved[rcv_idx].second, prefix);
            } else { // indexing_type == local_indexing
                update_nsv_queue<T,right_type>(right_nsv, q, recved[rcv_idx].first, local_size+rcv_idx, 0);
            }
        }

        // elements still in the queue do not have a smaller value to the left
        // -> set these to a special value and handle case if furthest_eq elements
        // are still waiting in queue
        for (auto it = q.rbegin(); it != q.rend(); ++it) {
            right_nsv[it->second - iprefix] = nonsv;
        }
    }

    lr_mins = recved;

    t.end_section("ANSV: finish ansv local");
}

template <typename T, int left_type = nearest_sm, int right_type = nearest_sm>
void ansv(const std::vector<T>& in, std::vector<size_t>& left_nsv, std::vector<size_t>& right_nsv, const mxx::comm& comm) {
    std::vector<std::pair<T, size_t>> lr_mins;
    ansv<T, left_type, right_type, global_indexing>(in, left_nsv, right_nsv, lr_mins, comm);
}

template <typename InputIterator, typename index_t = std::size_t>
std::vector<size_t> construct_suffix_tree(const suffix_array<InputIterator, index_t, true>& sa, const mxx::comm& comm) {
    mxx::section_timer t;
    // get input sizes
    size_t local_size = sa.local_SA.size();
    size_t global_size = mxx::allreduce(local_size, comm);
    size_t prefix = mxx::exscan(local_size, comm);
    // assert n >= p, or rather at least one element per process
    MXX_ASSERT(mxx::all_of(local_size >= 1, comm));
    // ansv of lcp!
    // TODO: use index_t instead of size_t
    std::vector<size_t> left_nsv;
    std::vector<size_t> right_nsv;
    std::vector<std::pair<index_t, size_t>> lr_mins;

    const size_t nonsv = std::numeric_limits<size_t>::max();
    t.end_section("pre ansv");

    // ANSV with furthest eq for left and smallest for right
    ansv<index_t, furthest_eq, nearest_sm, local_indexing>(sa.local_LCP, left_nsv, right_nsv, lr_mins, comm, nonsv);

    t.end_section("ansv");

    //const size_t sigma = 4; // TODO determine sigma or use local hashing
    // at most `n` internal nodes?
    // we represent only internal nodes. if a child pointer points to {total-size + X}, its a leaf
    // leafs are not represented as tree nodes inside the suffix tree structure
    // edges contain starting character? and/or length?
    //std::vector<size_t> tree_nodes(sigma*local_size); // TODO: determine real size

    // each SA[i] lies between two LCP values
    // LCP[i] = lcp(S[SA[i-1]], S[SA[i]])
    // leaf nodes are the suffix array positions. Their parent is the either their left or their right
    // LCP, depending on which one is larger

    std::vector<std::tuple<size_t, size_t, size_t>> parent_reqs;
    parent_reqs.reserve(2*local_size);
    // parent request where the character is the last `$`/`0` character
    // these don't have to be requested, but are locally fulfilled
    std::vector<std::tuple<size_t, size_t, size_t>> dollar_reqs;

    // get the first LCP value of the next processor
    index_t next_first_lcp = mxx::left_shift(sa.local_LCP[0], comm);
    for (size_t i = 0; i < local_size; ++i) {
        // for each suffix array position SA[i], we check the longest-common-prefix
        // with the neighboring suffixes SA[i-1] and SA[i+1]. Whichever one it
        // shares the larger common prefix with, is its sibling in the ST and
        // they share a parent at the depth given by the larger LCP value. The
        // index of the LCP that has that value will be the index of the parent
        // node.
        //
        // This means for every `i`, we need argmax_i {LCP[i], LCP[i+1]}, where
        // `i+1` might be on the next processor.
        //
        // If there are multiple leafs > 2 for an internal node, the parent
        // will be the index of the furthest equal element. We thus need
        // to use the NSV for determining the left parent.
        // If the right LCP is larger, then that one is the direct parent,
        // since there can't be any equal elements to the left (since the
        // right one was larger).

        // parent will be an index into LCP
        size_t parent = std::numeric_limits<size_t>::max();
        index_t lcp_val;

        // the globally first element has parent 1
        if (comm.rank() == 0 && i == 0) {
            // globally first leaf: SA[0]
            // -> parent = 1, since it is the common prefix between SA[0] and SA[1]
            parent = 1;
            if (local_size > 1) {
                lcp_val = sa.local_LCP[1];
            } else {
                MXX_ASSERT(global_size > 1);
                lcp_val = next_first_lcp;
            }
        } else {
            // To determine whether the left or right LCP is the parent,
            // we take the max of LCP[i]=lcp(SA[i-1],SA[i]) and LCP[i+1]=lcp(SA[i], SA[i+1])
            // There are two special cases to handle:
            // 1) locally last element: we need to use the first LCP value of the next processor
            //    in place of LCP[i+1]
            // 2) globally last element: parent is always the left furthest eq nsv
            if ((i == local_size-1
                 && (comm.rank() == comm.size() || sa.local_LCP[local_size-1] >= next_first_lcp))
                || (i < local_size-1 && sa.local_LCP[i] >= sa.local_LCP[i+1])) {
                // the parent is the left furthest eq or nearest sm
                size_t nsv;
                if (left_nsv[i] < local_size) {
                    nsv = prefix + left_nsv[i];
                    lcp_val = sa.local_LCP[left_nsv[i]];
                } else {
                    nsv = lr_mins[left_nsv[i] - local_size].second;
                    lcp_val = lr_mins[left_nsv[i] - local_size].first;
                }
                if (lcp_val == sa.local_LCP[i]) {
                    parent = nsv;
                } else {
                    parent = prefix + i;
                    lcp_val = sa.local_LCP[i];
                }
            } else {
                // SA[i] shares a longer prefix with its right neighbor SA[i+1]
                // they converge at internal node prefix+i+1
                parent = prefix + i + 1;
                if (i == local_size - 1)
                    lcp_val = next_first_lcp;
                else
                    lcp_val = sa.local_LCP[i+1];
            }
        }
        if (sa.local_SA[i] + lcp_val >= global_size) {
            MXX_ASSERT(sa.local_SA[i] + lcp_val == global_size);
            dollar_reqs.emplace_back(parent, global_size + prefix + i, global_size);
        } else {
            parent_reqs.emplace_back(parent, global_size + i + prefix, sa.local_SA[i] + lcp_val);
        }
    }

    // get parents of internal nodes (via LCP)
    for (size_t i = 0; i < local_size; ++i) {
        size_t parent = std::numeric_limits<size_t>::max();
        index_t lcp_val;
        // for each LCP position, get ANSV left-furthest-eq and right-nearest-sm
        // and the max of the two is the parent
        // Special cases: first (LCP[0]) and globally last LCP
        if (comm.rank() == 0 && i == 0) {
            // globally first position in LCP
            parent = 0; // TODO: this is the root, no parent!
            continue;

        //} else if (comm.rank() == comm.size() - 1 && i == local_size - 1) {
            // globally last element (no right ansv)
            // this case is identical to the regular case, since for the right
            // most element, right_nsv[i] will be == nonsv
            // and as such is handled in the corresponding case below
        } else {
            if (sa.local_LCP[i] == 0) {
                // this is a dupliate of the root node which is located at
                // position 0 on processor 0
                continue;
            } else {
                // left NSV can't be non-existant because LCP[0] = 0
                assert(left_nsv[i] != nonsv);
                if (right_nsv[i] == nonsv) {
                    // use left one
                    size_t nsv;
                    if (left_nsv[i] < local_size) {
                        nsv = prefix + left_nsv[i];
                        lcp_val = sa.local_LCP[left_nsv[i]];
                    } else {
                        nsv = lr_mins[left_nsv[i] - local_size].second;
                        lcp_val = lr_mins[left_nsv[i] - local_size].first;
                    }
                    if (lcp_val == sa.local_LCP[i]) {
                        // duplicate node, don't add!
                        continue;
                    }
                    parent = nsv;
                } else {
                    // get left NSV index and value
                    size_t lnsv;
                    index_t left_lcp_val;
                    if (left_nsv[i] < local_size) {
                        lnsv = prefix + left_nsv[i];
                        left_lcp_val = sa.local_LCP[left_nsv[i]];
                    } else {
                        lnsv = lr_mins[left_nsv[i] - local_size].second;
                        left_lcp_val = lr_mins[left_nsv[i] - local_size].first;
                    }
                    // get right NSV index and value
                    size_t rnsv;
                    index_t right_lcp_val;
                    if (right_nsv[i] < local_size) {
                        rnsv = prefix + right_nsv[i];
                        right_lcp_val = sa.local_LCP[right_nsv[i]];
                    } else {
                        rnsv = lr_mins[right_nsv[i] - local_size].second;
                        right_lcp_val = lr_mins[right_nsv[i] - local_size].first;
                    }
                    // parent is the NSV for which LCP is larger.
                    // if same, use left furthest_eq
                    if (left_lcp_val >= right_lcp_val) {
                        if (left_lcp_val == sa.local_LCP[i]) {
                            // this is a duplicate node, and won't be added
                            continue;
                        }
                        parent = lnsv;
                        lcp_val = left_lcp_val;
                    } else {
                        parent = rnsv;
                        lcp_val = right_lcp_val;
                    }
                }
            }
        }
        if (sa.local_SA[i] + lcp_val >= global_size) {
            MXX_ASSERT(sa.local_SA[i] + lcp_val == global_size);
            dollar_reqs.emplace_back(parent, prefix + i, global_size);
        } else {
            parent_reqs.emplace_back(parent, i + prefix, sa.local_SA[i] + lcp_val);
        }
    }

    t.end_section("locally calc parents");

    // 1) send tuples (parent, i, SA[i]+LCP[i]) to 3rd index)
#define USE_RMA 0
#if !USE_RMA
    mxx::partition::block_decomposition_buffered<size_t> part(global_size, comm.size(), comm.rank());
    mxx::all2all_func(parent_reqs, [&part](const std::tuple<size_t,size_t,size_t>& t) {return part.target_processor(std::get<2>(t));}, comm);

    t.end_section("all2all_func: req characters");

    // replace string request with character from original string
    for (size_t i = 0; i < parent_reqs.size(); ++i) {
        size_t offset = std::get<2>(parent_reqs[i]);
        if (offset == global_size) {
            // return `0` character
            // TODO: these cases do not have to be send to the last processor, but can simply be send to the `parent` destination
            // TODO: "often?" the parent might be local, so sending everything twice seems wasteful...
            // TODO: send to destination and there read character via MPI_Win based RMA?
            std::get<2>(parent_reqs[i]) = 0;
        } else {
            // get character from that global string position
            std::get<2>(parent_reqs[i]) = static_cast<size_t>(*(sa.input_begin+(std::get<2>(parent_reqs[i])-prefix)));
        }
    }
    // append the "dollar" requests
    parent_reqs.insert(parent_reqs.end(), dollar_reqs.begin(), dollar_reqs.end());
    dollar_reqs.clear(); dollar_reqs.shrink_to_fit();

    t.end_section("locally answer char queries");

    // 2) send tuples (parent, i, S[SA[i]+LCP[i]) to 1st index) [to parent]
    mxx::all2all_func(parent_reqs, [&part](const std::tuple<size_t,size_t,size_t>& t) {return part.target_processor(std::get<0>(t));}, comm);
    t.end_section("all2all_func: send to parent");
#else
    mxx::partition::block_decomposition_buffered<size_t> part(global_size, comm.size(), comm.rank());
    // TODO: filter by local parents (which should be the majority) and use all2all only for non-local parents
    parent_reqs.insert(parent_reqs.end(), dollar_reqs.begin(), dollar_reqs.end());
    dollar_reqs.clear(); dollar_reqs.shrink_to_fit();
    mxx::all2all_func(parent_reqs, [&part](const std::tuple<size_t,size_t,size_t>& t) {return part.target_processor(std::get<0>(t));}, comm);
    t.end_section("all2all_func: send to parent");

    // TODO: create MPI_Win for input string, create character array for size of parents
    //       and use RMA to request (read) all characters which are not `$`
    MPI_Win win;
    MPI_Win_create(&(*sa.input_begin), local_size, 1, MPI_INFO_NULL, comm, &win);

    MPI_Win_fence(0, win);

    // read characters here!
    std::vector<char> edge_chars(parent_reqs.size());
    for (size_t i = 0; i < parent_reqs.size(); ++i) {
        size_t offset = std::get<2>(parent_reqs[i]);
        // read global index offset
        if (offset == global_size) {
            // TODO: handle specially?
            edge_chars[i] = 0;
        } else {
            int proc = part.target_processor(offset);
            size_t proc_offset = offset - part.excl_prefix_size(proc);
            // request proc_offset from processor `proc` in window win
            MPI_Get(&edge_chars[i], 1, MPI_CHAR, proc, proc_offset, 1, MPI_CHAR, win);
        }
    }
    // fence to complete all requests
    //MPI_Win_fence(MPI_MODE_NOSTORE | MPI_MODE_NOPUT | MPI_MODE_NOSUCCEED, win);
    MPI_Win_fence(0, win);

    // iterate through the received characters and create 

    MPI_Win_free(&win);

    t.end_section("RMA read chars");
    //mxx::sync_cout(comm) << "reqs: " << parent_reqs << std::endl << "edge_chars:" << edge_chars << std::endl;
#endif

    // TODO: (alternatives for full lookup table in each node:)
    // local hashing key=(node-idx, char), value=(child idx)
    //            or multimap key=(node-idx), value=(char, child idx)
    //            2nd enables iteration over children, but not direct lookup
    //            of specific child
    //            2nd no different than fixed std::vector<std::list>

    // one internal node for each LCP entry, each internal node is sigma cells
    std::vector<size_t> internal_nodes((sa.sigma+1)*local_size);
    for (size_t i = 0; i < parent_reqs.size(); ++i) {
        size_t parent = std::get<0>(parent_reqs[i]);
        size_t node_idx = (parent - prefix)*(sa.sigma+1);
#if !USE_RMA
        size_t x = std::get<2>(parent_reqs[i]);
        uint16_t c = (x == global_size) ? 0 : sa.alphabet_mapping[x];
#else
        uint16_t c = sa.alphabet_mapping[edge_chars[i]];
#endif
        MXX_ASSERT(0 <= c && c < sa.sigma+1);
        size_t cell_idx = node_idx + c;
        internal_nodes[cell_idx] = std::get<1>(parent_reqs[i]);
    }

    t.end_section("locally: create internal nodes");

    // TODO:
    // - get character for internal nodes (first character of suffix starting at
    //   the node S[SA[i]+LCP[i]]) [i.e. requesting a full permutation] needed for
    //   all edges, i.e. up to 2n-1 (SA and LCP aligned)
    //
    // TODO:
    // need character S[SA[i]+maxLCP[i]] for each edge (leaf + internal node)
    // maxLCP is equal to LCP[parent]
    // but `parent` might not be local
    // -> can also be retrieved at `parent` when sending edges
    // SA[i] is local, we can send that, along with `i` since edge is (parent,i)
    // so at least needs sending (parent, i, SA[i])
    // OR: (parent, i, c) if `c` is retrieved before sending edges to `parent`
    // clue: we are sending exactly as many edge elements as in ANSV
    // whereas requesting `c` is full all2all_msgs/queries/RMA?
    //


    // TODO: probably best to send a character S[SA[i]+maxLCP[i]] along for edge labeling
    // each internal node: at LCP index, LCP gives string depth already
    //                     children edges: either by fixed size sigma*n lookup table
    //                     or by hashtable (lcp-index, character)->child-index
    //                     The LCP indeces are predictable [n/p*rank,n/p*(rank+1))
    //                     -> be careful with hash function!

    return internal_nodes;
}


/**
 * @brief   Solves the ANSV problem sequentially in one direction.
 *
 * @tparam T    Type of input elements.
 * @param in    Vector of input elemens
 * @param left  Whether to find left or right smaller value. `True` denotes
 *              left, and `False` denotes finding the smaller value to the right.
 *
 * @return      The nearest smaler value for each element in `in` to the direction
 *              given by `left`.
 */
template <typename T>
std::vector<size_t> ansv_sequential(const std::vector<T>& in, bool left) {
    std::vector<size_t> nsv(in.size());

    std::deque<size_t> q;
    for (size_t i = 0; i < in.size(); ++i) {
        size_t idx = left ? in.size() - 1 - i : i;
        while (!q.empty() && in[idx] < in[q.back()]) {
            // current element is the min for in[i-1]
            nsv[q.back()] = idx;
            q.pop_back();
        }
        // TODO: potentially handle `equal` elements differently
        q.push_back(idx);
    }
    return nsv;
}

#endif // ANSV_HPP
