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
    assert(n_right_mins >= 1);
    q.clear();
    assert(n_right_mins + n_left_mins == lr_mins.size());

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
                while (end_idx+1 < lr_mins.size() && lr_mins[end_idx+1].first <= prev_mins_min) {
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


    /***************************************************************
     *  Step 3: Again solve ANSV locally and use lr_mins as tails  *
     ***************************************************************/

    left_nsv.resize(local_size);
    right_nsv.resize(local_size);

    q.clear();
    // iterate backwards to get the nearest smaller element to left for each element
    // TODO: this doesn't really require pairs in the queue (index suffices)
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

        if (left_type == furthest_eq) {
            if (!q.empty() && recved[rcv_idx].first == q.back().first) {
                // skip till furthest of an equal range
                while (rcv_idx != 0 && recved[rcv_idx].first == recved[rcv_idx-1].first) {
                    i++;
                    rcv_idx = n_left_recv - i - 1;
                }
                // setting this as the furthest_eq for all equal elements in the queue
                // and removing all equal elements from the queue
                while (!q.empty() && q.back().first == recved[rcv_idx].first) {
                    if (indexing_type == global_indexing) {
                        left_nsv[q.back().second - prefix] = recved[rcv_idx].second;
                    } else { // indexing_type == local_indexing
                        left_nsv[q.back().second] = local_size + rcv_idx;
                    }
                    q.pop_back();
                }
            }
        }
    }

    // elements still in the queue do not have a smaller value to the left
    //  -> set these to a special value and handle case if furthest_eq
    //     elements are still waiting in queue
    const size_t iprefix = (indexing_type == global_indexing) ? prefix : 0;
    for (auto it = q.rbegin(); it != q.rend(); ++it) {
        if (left_type == furthest_eq) {
                auto it2 = it;
                // set left most as furthest_eq for all equal elements
                while(it2+1 != q.rend() && it->first == (it2+1)->first) {
                    ++it2;
                    left_nsv[it2->second - iprefix] = it->second;
                }
                left_nsv[it->second - iprefix] = nonsv;
                it = it2;
        } else {
            left_nsv[it->second - iprefix] = nonsv;
        }
    }

    // iterate forwards to get the nearest smaller value to the right for each element
    q.clear();
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

        if (right_type == furthest_eq) {
            if (!q.empty() && recved[rcv_idx].first == q.back().first) {
                // skip till furthest of an equal range
                while (i+1 < n_right_recv  && recved[rcv_idx].first == recved[rcv_idx+1].first) {
                    i++;
                    rcv_idx = n_left_recv + i;
                }
                // setting this as the furthest_eq for all equal elements in the queue
                // and removing all equal elements from the queue
                while (!q.empty() && q.back().first == recved[rcv_idx].first) {
                    if (indexing_type == global_indexing) {
                        right_nsv[q.back().second - prefix] = recved[rcv_idx].second;
                    } else { // indexing_type == local_indexing
                        right_nsv[q.back().second] = local_size+rcv_idx;
                    }
                    q.pop_back();
                }
            }
        }
    }

    // elements still in the queue do not have a smaller value to the left
    // -> set these to a special value and handle case if furthest_eq elements
    // are still waiting in queue
    for (auto it = q.rbegin(); it != q.rend(); ++it) {
        if (right_type == furthest_eq) {
                auto it2 = it;
                // set left most as furthest_eq for all equal elements
                while(it2+1 != q.rend() && it->first == (it2+1)->first) {
                    ++it2;
                    right_nsv[it2->second - iprefix] = it->second;
                }
                right_nsv[it->second - iprefix] = nonsv;
                it = it2;
        } else {
            right_nsv[it->second - iprefix] = nonsv;
        }
    }

    lr_mins = recved;
}

template <typename T, int left_type = nearest_sm, int right_type = nearest_sm>
void ansv(const std::vector<T>& in, std::vector<size_t>& left_nsv, std::vector<size_t>& right_nsv, const mxx::comm& comm) {
    std::vector<std::pair<T, size_t>> lr_mins;
    ansv<T, left_type, right_type, global_indexing>(in, left_nsv, right_nsv, lr_mins, comm);
}

template <typename InputIterator, typename index_t = std::size_t>
void construct_suffix_tree(const suffix_array<InputIterator, index_t, true>& sa, const mxx::comm& comm) {
    // ansv of lcp!
    // TODO: use index_t instead of size_t
    std::vector<size_t> left_nsv;
    std::vector<size_t> right_nsv;
    std::vector<std::pair<index_t, size_t>> lr_mins;

    const size_t nonsv = std::numeric_limits<size_t>::max();

    // ANSV with furthest eq for left and smallest for right
    ansv<index_t, furthest_eq, nearest_sm, local_indexing>(sa.LCP, left_nsv, right_nsv, lr_mins, comm, nonsv);

    size_t local_size = sa.SA.size();
    size_t prefix = mxx::exscan(local_size, comm);
    const size_t sigma = 4; // TODO determine sigma or use local hashing
    // at most `n` internal nodes?
    // we represent only internal nodes. if a child pointer points to {total-size + X}, its a leaf
    // leafs are not represented as tree nodes inside the suffix tree structure
    // edges contain starting character? and/or length?
    std::vector<size_t> tree_nodes(sigma*local_size); // TODO: determine real size

    // each SA[i] lies between two LCP values
    // LCP[i] = lcp(S[SA[i-1]], S[SA[i]])
    // leaf nodes are the suffix array positions. Their parent is the either their left or their right
    // LCP, depending on which one is larger

    // get the first LCP value of the next processor
    index_t next_first_lcp = mxx::left_shift(sa.LCP[0], comm);
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
        // Special cases are for globally the first element and the last element

        // parent will be an index into LCP
        size_t parent = std::numeric_limits<size_t>::max();
        // distinguish certain special cases (SA[0], SA[global_last], and last on processor)
        if (comm.rank() == 0 && i == 0) {
            // globally first leaf: SA[0]
            // -> parent = 1, since it is the common prefix between SA[0] and SA[1]
            parent = 1;
        } else if (comm.rank() == comm.size()-1 && i == local_size-1) {
            // globally last leaf: SA[global_size-1]
            // -> parent = global-size-1, since it is the common prefix between
            //                            this and the previous leaf
            // TODO: furthest-equal for last LCP instead
            parent = global_size - 1;
        } else if (i == local_size-1) {
            // use max of `next_first_lcp` and local LCP[local_size-1] to determine parent
            // for SA[i]
            if (sa.LCP[local_size-1] >= next_first_lcp) {
                // TODO: use left furthest eq of LCP[local_size-1] as the definite parent
            } else {
                // left LCP is smaller -> right LCP is parent (which in this case is the next processor)
                parent = prefix + local_size;
            }
        } else {
            if (sa.LCP[i] >= sa.LCP[i+1]) {
                // use left furthest equal of [i] as the parent
                // TODO
            } else {
                // SA[i] shares a longer prefix with its right neighbor SA[i+1]
                // they converge at internal node prefix+i+1
                parent = prefix + i + 1;
            }
        }

        // TODO: save parent or set pointer FROM parent: tree_nodes[parent] = prefix+i
        // if parent not local: send?
        //
    }
    // TODO: probably best to send a character S[SA[i]+maxLCP[i]] along for edge labeling

    // get parents of internal nodes (via LCP)
    for (size_t i = 0; i < local_size; ++i) {
        size_t parent = std::numeric_limits<size_t>::max();
        // for each LCP position, get ANSV left-furthest-eq and right-nearest-sm
        // and the max of the two is the parent
        // Special cases: first (LCP[0]) and globally last LCP
        if (comm.rank() == 0 && i == 0) {
            // globally first position in LCP
            parent = 0; // TODO: this is the root, no parent!

        //} else if (comm.rank() == comm.size() - 1 && i == local_size - 1) {
            // globally last element (no right ansv)
            // this case is identical to the regular case, since for the right
            // most element, right_nsv[i] will be == nonsv
            // and as such is handled in the corresponding case below
        } else {
            if (sa.LCP[i] == 0) {
                parent = 0; // root is parent if LCP is 0
            } else {
                // left NSV can't be non-existant because LCP[0] = 0
                assert(left_nsv[i] != nonsv);
                if (right_nsv[i] == nonsv) {
                    // use left one
                    size_t nsv;
                    index_t lcp_val;
                    if (left_nsv[i] < local_size) {
                        nsv = prefix + left_nsv[i];
                        lcp_val = sa.LCP[left_nsv[i]];
                    } else {
                        nsv = lr_mins[left_nsv[i] - local_size].second;
                        lcp_val = lr_mins[left_nsv[i] - local_size].first;
                    }
                    parent = nsv;
                } else {
                    // get left NSV index and value
                    size_t lnsv;
                    index_t left_lcp_val;
                    if (left_nsv[i] < local_size) {
                        lnsv = prefix + left_nsv[i];
                        left_lcp_val = sa.LCP[left_nsv[i]];
                    } else {
                        lnsv = lr_mins[left_nsv[i] - local_size].second;
                        left_lcp_val = lr_mins[left_nsv[i] - local_size].first;
                    }
                    // get right NSV index and value
                    size_t rnsv;
                    index_t right_lcp_val;
                    if (right_nsv[i] < local_size) {
                        rnsv = prefix + right_nsv[i];
                        right_lcp_val = sa.LCP[right_nsv[i]];
                    } else {
                        rnsv = lr_mins[right_nsv[i] - local_size].second;
                        right_lcp_val = lr_mins[right_nsv[i] - local_size].first;
                    }
                    // parent is the NSV for which LCP is larger.
                    // if same, use left furthest_eq
                    if (left_lcp_val >= right_lcp_val) {
                        parent = lnsv;
                    } else {
                        parent = rnsv;
                    }
                }
            }
        }
    }

    // each leaf node (SA position) determines its parent via max of two ajacent LCP values
    // each internal node (LCP position) determiens its parent via ANSV
    // each node "hangs" itself underneath its parent node
    // what's the likelyhood of the parent node to be on another processor??
    // since both the LCP and SA are equally distributed, and the ANSV likely to be close by
    // its likely they are "close" as well
    // ?? communication is basically similar the ANSV communication?
    // at most n/p on each processor anyway (can't be parent to more than sigma * n/p)
    // even the "heavy" (low LCP) nodes are distributed equally?



    // TODO:
    // - get max of ansvs as the `parent` node (requires lr_mins for non local values)
    // - get character for internal nodes (first character of suffix starting at
    //   the node S[SA[i]+LCP[i]]) [i.e. requesting a full permutation] needed for
    //   all edges, i.e. up to 2n-1 (SA and LCP aligned)
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
