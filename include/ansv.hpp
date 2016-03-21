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
#include <mxx/timer.hpp>
#include <mxx/algos.hpp>

constexpr int nearest_sm = 0;
constexpr int nearest_eq = 1;
constexpr int furthest_eq = 2;

constexpr int global_indexing = 0;
constexpr int local_indexing = 1;

constexpr bool dir_left = true;
constexpr bool dir_right = false;

// for debugging
//#define SDEBUG(x) mxx::sync_cerr(comm) << "[" << comm.rank() << "]: " #x " = " << (x) << std::endl
#define SDEBUG(x)


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
std::vector<size_t> ansv_sequential(const std::vector<T>& in, bool left, size_t nonsv = 0) {
    std::vector<size_t> nsv(in.size());

    std::deque<size_t> q;
    for (size_t i = 0; i < in.size(); ++i) {
        size_t idx = left ? in.size() - 1 - i : i;
        while (!q.empty() && in[idx] < in[q.back()]) {
            // current element is the min for in[i-1]
            nsv[q.back()] = idx;
            q.pop_back();
        }
        q.push_back(idx);
    }
    for (auto& x : q) {
        nsv[x] = nonsv;
    }
    return nsv;
}

template <typename T, int type>
inline void update_nsv_queue(std::vector<size_t>& nsv, std::deque<std::pair<T,size_t>>& q, const T& next_value, size_t idx, size_t prefix) {
    while (!q.empty() && next_value < q.back().first) {
        // current element is the min for the last element in the queue
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

// TODO: this is yet uncorrect for furthest_eq
template <typename Iterator, typename T, int nsv_type, bool dir>
void local_indexing_nsv(Iterator begin, Iterator end, std::vector<std::pair<T, size_t>>& unmatched, std::vector<size_t>& nsv) {
    size_t n = std::distance(begin, end);
    std::deque<std::pair<T,size_t> > q;
    for (Iterator it = begin; it != end; ++it) {
        size_t idx = (dir == dir_left) ? n - std::distance(begin,it) - 1 : std::distance(begin,it);
        size_t prefix = 0;
        update_nsv_queue<T, nsv_type>(nsv, q, *it, idx, prefix);
        q.push_back(std::pair<T, size_t>(*it, prefix+idx));
    }
    size_t prev_size = unmatched.size();
    if (dir == dir_left) {
        unmatched = std::vector<std::pair<T,size_t>>(q.rbegin(), q.rend());
    } else {
        unmatched.insert(unmatched.end(), q.begin(), q.end());
    }
    for (size_t i = prev_size; i < unmatched.size(); ++i) {
        nsv[unmatched[i].second] = n + i;
    }
}

// solve all locals with correct local indexing
// unmatched elements point into the lr_mins
// later only lr_mins are used for getting global solutions
/*
template <typename T, int left_type = nearest_sm, int right_type = nearest_sm>
size_t local_ansv_unmatched_nsv(const std::vector<T>& in, const size_t prefix, std::vector<std::pair<T, size_t>>& unmatched, std::vector<std::pair<T, size_t>>& left_nsv, std::vector<std::pair<T,size_t>>& right_nsv) {
    // backwards direction (left mins)
    std::deque<std::pair<T,size_t> > q;

    for (size_t i = in.size(); i > 0; --i) {
        while (!q.empty() && in[i-1] < q.back().first) {
            q.pop_back();
        }
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
    unmatched = std::vector<std::pair<T,size_t>>(q.rbegin(), q.rend());
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
    unmatched.insert(unmatched.end(), q.begin(), q.end());
    MXX_ASSERT(n_right_mins >= 1);
    if (n_right_mins + n_left_mins != unmatched.size()) {
        MXX_ASSERT(false);
    }
    return n_left_mins;
}
*/

template <typename T, int left_type = nearest_sm, int right_type = nearest_sm>
size_t local_ansv_unmatched(const std::vector<T>& in, const size_t prefix, std::vector<std::pair<T, size_t>>& unmatched) {
    // backwards direction (left mins)
    std::deque<std::pair<T,size_t> > q;

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
    unmatched = std::vector<std::pair<T,size_t>>(q.rbegin(), q.rend());
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
    unmatched.insert(unmatched.end(), q.begin(), q.end());
    MXX_ASSERT(n_right_mins >= 1);
    if (n_right_mins + n_left_mins != unmatched.size()) {
        MXX_ASSERT(false);
    }
    return n_left_mins;
}

template <typename T, int left_type, int right_type>
void ansv_comm_allpairs_params(const std::vector<std::pair<T, size_t>>& lr_mins,
                               size_t n_left_mins,
                               std::vector<size_t>& send_counts,
                               std::vector<size_t>& send_displs,
                               const mxx::comm& comm) {
    // allgather min and max of all left and right mins
    T local_min = lr_mins[n_left_mins].first;
    std::vector<T> allmins = mxx::allgather(local_min, comm);
    send_counts = std::vector<size_t>(comm.size(), 0);
    send_displs = std::vector<size_t>(comm.size(), 0);

    // calculate which processor to exchange unmatched left_mins and right_mins with
    // 1) calculate communication parameters for left processes
    if (comm.rank() > 0) {
        size_t left_idx = 0;
        T prev_mins_min = allmins[comm.rank()-1];
        for (int i = comm.rank()-1; i >= 0; --i) {
            size_t start_idx = left_idx;
            // move start back to other `equal` elements (there can be at most 2)
            if (right_type == furthest_eq && i < comm.rank()-1) {
                while (start_idx > 0 && lr_mins[start_idx-1].first <= prev_mins_min) {
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
}


template <typename T, int left_type, int right_type>
size_t ansv_communicate_allpairs(const std::vector<std::pair<T,size_t>>& lr_mins, size_t n_left_mins, std::vector<std::pair<T, size_t>>& remote_seqs, const mxx::comm& comm) {
    mxx::section_timer t(std::cerr, comm);

    std::vector<size_t> send_counts;
    std::vector<size_t> send_displs;
    ansv_comm_allpairs_params<T, left_type, right_type>(lr_mins, n_left_mins, send_counts, send_displs, comm);

    // exchange lr_mins via all2all
    std::vector<size_t> recv_counts = mxx::all2all(send_counts, comm);
    std::vector<size_t> recv_displs = mxx::impl::get_displacements(recv_counts);
    size_t recv_size = recv_counts.back() + recv_displs.back();
    remote_seqs = std::vector<std::pair<T, size_t>>(recv_size);

    t.end_section("ANSV: calc comm params");
    mxx::all2allv(&lr_mins[0], send_counts, send_displs, &remote_seqs[0], recv_counts, recv_displs, comm);
    t.end_section("ANSV: all2allv");

    // solve locally given the exchanged values (by prefilling min-queue before running locally)
    size_t n_left_recv = recv_displs[comm.rank()];
    return n_left_recv;
}

template <typename T, int left_type, int right_type>
void x_ansv_local(const std::vector<T>& in,
                std::vector<std::pair<T, size_t>>& lr_mins,
                std::vector<size_t>& left_nsv, std::vector<size_t>& right_nsv) {
    lr_mins = std::vector<std::pair<T, size_t>>(in.size());
    size_t q = 0;
    // left scan = find left matches
    for (size_t i = 0; i < in.size(); ++i) {
        size_t idx = in.size() - i - 1;
        if (q > 0) {
            if (in[idx] < lr_mins[q-1].first) {
                // a new minimum (potentially)
                if (left_type == furthest_eq) {
                    // TODO
                }
            }
        }
        while (q > 0 && in[idx] < lr_mins[q-1].first) {
            left_nsv[lr_mins[q-1].second] = idx;
            --q;
        }
    }
    // right scan = find right matches
    for (size_t i = 0; i < in.size(); ++i) {
        while (q > 0 && in[i] < lr_mins[q-1].first) {
            right_nsv[lr_mins[q-1].second] = i;
            --q;
        }
    }
}

template <typename T, bool direction, int indexing_type, typename Iterator>
void ansv_local_finish_furthest_eq(const std::vector<T>& in, Iterator tail_begin, Iterator tail_end, size_t prefix, size_t tail_prefix, size_t nonsv, std::vector<size_t>& nsv) {
    std::deque<std::pair<T,size_t> > q;
    const size_t iprefix = (indexing_type == global_indexing) ? prefix : 0;
    // iterate forwards through the received elements to fill the queue
    // TODO: don't need a queue if this is a non-decreasing sequence
    // since we never pop, we just need to keep track of an iterator position
    size_t n_tail = std::distance(tail_begin, tail_end);
    for (size_t i = 0; i < n_tail; ++i) {
        auto r = (direction == dir_left) ? tail_begin+i : tail_begin+(n_tail-i-1);
        // TODO: I should be able to assume that the sequence is non-decreasing
        while (!q.empty() && r->first < q.back().first) {
            q.pop_back();
        }
        if (q.empty() || r->first > q.back().first) {
            // push only if this element is larger (not equal)
            if (indexing_type == global_indexing) {
                q.push_back(*r);
            } else {
                size_t rcv_idx = tail_prefix + std::distance(tail_begin, r);
                q.push_back(std::pair<T, size_t>(r->first, in.size() + rcv_idx));
            }
        }
    }

    // iterate forward through the local items and set their nsv
    // to the first equal or smaller element in the queue
    for (size_t idx = 0; idx < in.size(); ++idx) {
        size_t i = (direction == dir_left) ? idx : in.size() - idx - 1;
        while (!q.empty() && in[i] < q.back().first) {
            q.pop_back();
        }
        if (q.empty()) {
            nsv[i] = nonsv;
        } else {
            nsv[i] = q.back().second;
        }
        if (q.empty() || in[i] > q.back().first) {
            q.push_back(std::pair<T, size_t>(in[i], iprefix+i));
        }
    }
}

template <typename T, int left_type, int right_type, int indexing_type>
void ansv_local_finish_all(const std::vector<T>& in, const std::vector<std::pair<T,size_t>>& recved, size_t n_left_recv, size_t prefix, size_t nonsv, std::vector<size_t>& left_nsv, std::vector<size_t>& right_nsv) {
    left_nsv.resize(in.size());
    right_nsv.resize(in.size());
    size_t local_size = in.size();
    size_t n_right_recv = recved.size() - n_left_recv;

    std::deque<std::pair<T,size_t> > q;
    const size_t iprefix = (indexing_type == global_indexing) ? prefix : 0;

    if (left_type == furthest_eq) {
        ansv_local_finish_furthest_eq<T, dir_left, indexing_type>(in, recved.begin(), recved.begin()+n_left_recv, prefix, 0, nonsv, left_nsv);
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
        ansv_local_finish_furthest_eq<T, dir_right, indexing_type>(in, recved.begin()+n_left_recv, recved.end(), prefix, n_left_recv, nonsv, right_nsv);
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
}

template <typename Iterator> //, int left_type, int right_type>
std::pair<Iterator,Iterator> ansv_merge(Iterator left_begin, Iterator left_end, Iterator right_begin, Iterator right_end) {
    // starting with the largest value (right most in left sequence and leftmost in right sequence)
    Iterator l = left_end;
    Iterator r = right_begin;
    while (l != left_begin && r != right_end) {
        Iterator l1 = l-1;
        if (l1->first < r->first) {
            *r = *l1;
            ++r;
        } else if (r->first < l1->first) {
            *l1 = *r;
            --l;
        } else {
            // r == l
            // TODO: important case in the merge
            // for now assume nearest_sm for both
            // find smaller items by iterating up
            Iterator lsm = l1;
            while (lsm != left_begin && l1->first == (lsm-1)->first)
                --lsm;

            Iterator rsm = r+1;
            while (rsm != right_end && r->first == rsm->first)
                ++rsm;

            // TODO: in case of furthest eq, use the last offset
            // assign the smaller r (rsm) to all the equal left if there are
            if (rsm != right_end) {
                for (Iterator li = l1; li != (lsm-1); --li) {
                    *li = *rsm;
                }
            }
            // assign the smaller l (lsm) to all equal right
            if (lsm != left_begin) {
                for (Iterator ri = r; ri != rsm; ++ri) {
                    *ri = *(lsm-1);
                }
            }
            r = rsm;
            l = lsm;
        }
    }
    return std::pair<Iterator, Iterator>(l-1, r);
}

// parallel all nearest smallest value
// TODO: iterator version??
// TODO: comparator type
// TODO: more compact via index_t template instead of size_t
template <typename T, int left_type = nearest_sm, int right_type = nearest_sm, int indexing_type = global_indexing>
void ansv(const std::vector<T>& in, std::vector<size_t>& left_nsv, std::vector<size_t>& right_nsv, std::vector<std::pair<T,size_t> >& lr_mins, const mxx::comm& comm, size_t nonsv = 0) {
    mxx::section_timer t(std::cerr, comm);

    size_t local_size = in.size();
    size_t prefix = mxx::exscan(local_size, comm);

    /*****************************************************************
     *  Step 1: Locally calculate ANSV and save un-matched elements  *
     *****************************************************************/
    size_t n_left_mins = local_ansv_unmatched<T, left_type, right_type>(in, prefix, lr_mins);
    t.end_section("ANSV: local ansv");

    /***************************************************************
     *  Step 2: communicate un-matched elements to correct target  *
     ***************************************************************/
    std::vector<std::pair<T, size_t>> recved;
    size_t n_left_recv = ansv_communicate_allpairs<T, left_type, right_type>(lr_mins, n_left_mins, recved, comm);
    t.end_section("ANSV: communicate all");

    /***************************************************************
     *  Step 3: Again solve ANSV locally and use lr_mins as tails  *
     ***************************************************************/
    ansv_local_finish_all<T, left_type, right_type, indexing_type>(in, recved, n_left_recv, prefix, nonsv, left_nsv, right_nsv);
    lr_mins = recved;
    t.end_section("ANSV: finish ansv local");
}

template <typename T, int indexing_type = global_indexing>
void my_ansv(const std::vector<T>& in, std::vector<size_t>& left_nsv, std::vector<size_t>& right_nsv, std::vector<std::pair<T,size_t> >& lr_mins, const mxx::comm& comm, size_t nonsv = 0) {
    mxx::section_timer t(std::cerr, comm);

    size_t local_size = in.size();
    size_t prefix = mxx::exscan(local_size, comm);

    /*****************************************************************
     *  Step 1: Locally calculate ANSV and save un-matched elements  *
     *****************************************************************/
    if (left_nsv.size() != in.size())
        left_nsv.resize(in.size());
    if (right_nsv.size() != in.size())
        right_nsv.resize(in.size());
    //size_t n_left_mins = local_ansv_unmatched<T, left_type, right_type>(in, prefix, lr_mins);
    local_indexing_nsv<decltype(in.rbegin()), T, nearest_sm, dir_left>(in.rbegin(), in.rend(), lr_mins, left_nsv);
    size_t n_left_mins = lr_mins.size();
    local_indexing_nsv<decltype(in.begin()), T, nearest_sm, dir_right>(in.begin(), in.end(), lr_mins, right_nsv);
    // change lrmin indexing to global
    for (size_t i = 0; i < lr_mins.size(); ++i) {
        lr_mins[i].second += prefix;
    }

    t.end_section("ANSV: local ansv");


    //mxx::sync_cout(comm) << "in=" << in << std::endl;
    //mxx::sync_cout(comm) << "lr_mins=" << lr_mins << std::endl;
    //mxx::sync_cout(comm) << "left_nsv=" << left_nsv << std::endl;
    //mxx::sync_cout(comm) << "right_nsv=" << right_nsv << std::endl;

    /***************************************************************
     *  Step 2: communicate un-matched elements to correct target  *
     ***************************************************************/
    std::vector<std::pair<T, size_t>> recved;
    size_t n_left_recv = ansv_communicate_allpairs<T, nearest_sm, nearest_sm>(lr_mins, n_left_mins, recved, comm);
    t.end_section("ANSV: communicate all");
    //mxx::sync_cout(comm) << "recved=" << recved << std::endl;

    /***************************************************************
     *  Step 3: Again solve ANSV locally and use lr_mins as tails  *
     ***************************************************************/
    // TODO: finish via merge?
    ansv_merge(recved.begin(), recved.begin()+n_left_recv, lr_mins.begin(), lr_mins.begin()+n_left_mins);
    ansv_merge(lr_mins.begin()+n_left_mins, lr_mins.end(), recved.begin()+n_left_recv, recved.end());
    //mxx::sync_cout(comm) << "merged lrmins=" << lr_mins << std::endl;
    //mxx::sync_cout(comm) << "merged lrmins=" << lr_mins << std::endl;
    //ansv_local_finish_all<T, left_type, right_type, indexing_type>(in, recved, n_left_recv, prefix, nonsv, left_nsv, right_nsv);
    // local to global indexing transformation
    for (size_t i = 0; i < in.size(); ++i) {
        if(left_nsv[i] >= local_size) {
            if (lr_mins[left_nsv[i]-local_size].second == i+prefix) {
                left_nsv[i] = nonsv;
            } else {
                left_nsv[i] = lr_mins[left_nsv[i]-local_size].second;
            }
        } else {
            left_nsv[i] += prefix;
        }
        if (right_nsv[i] >= local_size) {
            if (lr_mins[right_nsv[i]-local_size].second == i+prefix) {
                right_nsv[i] = nonsv;
            } else {
                right_nsv[i] = lr_mins[right_nsv[i]-local_size].second;
            }
        } else {
            right_nsv[i] += prefix;
        }
    }
    // mxx::sync_cout(comm) << "left_nsv=" << left_nsv << std::endl;
    // mxx::sync_cout(comm) << "right_nsv=" << right_nsv << std::endl;
    t.end_section("ANSV: finish ansv local");
}

template <typename T, int indexing_type = global_indexing>
void my_ansv_minpair(const std::vector<T>& in, std::vector<size_t>& left_nsv, std::vector<size_t>& right_nsv, std::vector<std::pair<T,size_t> >& lr_mins, const mxx::comm& comm, size_t nonsv = 0) {
    mxx::section_timer t(std::cerr, comm);

    size_t local_size = in.size();
    size_t prefix = mxx::exscan(local_size, comm);

    /*****************************************************************
     *  Step 1: Locally calculate ANSV and save un-matched elements  *
     *****************************************************************/
    if (left_nsv.size() != in.size())
        left_nsv.resize(in.size());
    if (right_nsv.size() != in.size())
        right_nsv.resize(in.size());
    //size_t n_left_mins = local_ansv_unmatched<T, left_type, right_type>(in, prefix, lr_mins);
    local_indexing_nsv<decltype(in.rbegin()), T, nearest_sm, dir_left>(in.rbegin(), in.rend(), lr_mins, left_nsv);
    size_t n_left_mins = lr_mins.size();
    local_indexing_nsv<decltype(in.begin()), T, nearest_sm, dir_right>(in.begin(), in.end(), lr_mins, right_nsv);
    // change lrmin indexing to global
    for (size_t i = 0; i < lr_mins.size(); ++i) {
        lr_mins[i].second += prefix;
    }

    t.end_section("ANSV: local ansv");

    //mxx::sync_cout(comm) << "in=" << in << std::endl;
    //mxx::sync_cout(comm) << "lr_mins=" << lr_mins << std::endl;
    //mxx::sync_cout(comm) << "left_nsv=" << left_nsv << std::endl;
    //mxx::sync_cout(comm) << "right_nsv=" << right_nsv << std::endl;

    /***************************************************************
     *  Step 2: communicate un-matched elements to correct target  *
     ***************************************************************/
    std::vector<std::pair<T, size_t>> recved;
    //size_t n_left_recv = ansv_communicate_allpairs<T, nearest_sm, nearest_sm>(lr_mins, n_left_mins, recved, comm);
    t.end_section("ANSV: communicate all");
    //mxx::sync_cout(comm) << "recved=" << recved << std::endl;
    std::vector<size_t> send_counts;
    std::vector<size_t> send_displs;
    ansv_comm_allpairs_params<T, furthest_eq, furthest_eq>(lr_mins, n_left_mins, send_counts, send_displs, comm);
    // determine min/max for each communication pair
    std::vector<size_t> min_send_counts(send_counts.begin(), send_counts.end());
    std::vector<size_t> min_recv_counts = mxx::all2all(min_send_counts, comm);
    // TODO: modify send/recv counts by selecting the minimum partner
    for (int i = 0; i < comm.size(); ++i) {
        if (min_recv_counts[i] != 0 && min_send_counts[i] != 0) {
            // choose the min and set the other one to 0
            if (min_recv_counts[i] < min_send_counts[i]) {
                min_send_counts[i] = 0;
            } else if (min_recv_counts[i] == min_send_counts[i]) {
                // need to handle this case so that its symmetric
                if (i < comm.rank()) {
                    min_recv_counts[i] = 0;
                } else {
                    min_send_counts[i] = 0;
                }
            } else {
                min_recv_counts[i] = 0;
            }
        }
    }

    SDEBUG(lr_mins);
    SDEBUG(send_counts);
    SDEBUG(min_send_counts);
    SDEBUG(min_recv_counts);

    std::vector<size_t> recv_displs = mxx::local_exscan(min_recv_counts);
    size_t recv_size = min_recv_counts.back() + recv_displs.back();
    recved = std::vector<std::pair<T, size_t>>(recv_size);
    // communicate through an all2all TODO: replace by send/recv + barrier?
    t.end_section("ANSV: calc comm params");
    mxx::all2allv(&lr_mins[0], min_send_counts, send_displs, &recved[0], min_recv_counts, recv_displs, comm);
    t.end_section("ANSV: all2allv");


    SDEBUG(recved);

    /***************************************************************
     *  Step 3: Again solve ANSV locally and use lr_mins as tails  *
     ***************************************************************/
    // use original send_counts as the range for the merge, and merge one sequence at a time
    // at the same time determine the send_counts for sending back solutions
    std::vector<size_t> ret_send_counts(min_recv_counts);
    std::vector<size_t> ret_send_displs(recv_displs);
    typedef typename std::vector<std::pair<T, size_t>>::iterator pair_it;
    for (int i = comm.rank()-1; i >= 0; --i) {
        // local_range = send_displs[i] + [0, send_counts[i])
        if (min_recv_counts[i] > 0) {
            // local merge
            pair_it rec_begin = recved.begin()+recv_displs[i];
            pair_it rec_end = recved.begin()+recv_displs[i]+min_recv_counts[i];
            pair_it loc_begin = lr_mins.begin()+send_displs[i];
            pair_it loc_end = lr_mins.begin()+send_displs[i]+send_counts[i];
            pair_it rec_merge_end, loc_merge_end;
            std::tie(rec_merge_end, loc_merge_end) = ansv_merge(rec_begin, rec_end, loc_begin, loc_end);
            ret_send_counts[i] = min_recv_counts[i] - (rec_merge_end  - rec_begin + 1);
            ret_send_displs[i] += (rec_merge_end - rec_begin + 1);
        } else {
            // this section is waiting for the result from the remote
        }
    }
    for (int i = comm.rank()+1; i < comm.size(); ++i) {
        if (min_recv_counts[i] > 0) {
            pair_it loc_begin = lr_mins.begin()+send_displs[i];
            pair_it loc_end = lr_mins.begin()+send_displs[i]+send_counts[i];
            pair_it rec_begin = recved.begin()+recv_displs[i];
            pair_it rec_end = recved.begin()+recv_displs[i]+min_recv_counts[i];
            pair_it loc_merge_end, rec_merge_end;
            std::tie(loc_merge_end, rec_merge_end) = ansv_merge(loc_begin, loc_end, rec_begin, rec_end);
            ret_send_counts[i] = min_recv_counts[i] - (rec_end - rec_merge_end);
            // displacements stay the same, since we take elements away at the end of the sequence
        } else {
            // remote is answering my queries, so nothing to do here
        }
    }

    // get receive counts via all2all
    std::vector<size_t> ret_recv_counts = mxx::all2all(ret_send_counts, comm);
    std::vector<size_t> ret_recv_displs(send_displs);

    // re-calculate the receive displacementsjjj
    for (int i = comm.rank()+1; i < comm.size(); ++i) {
        ret_recv_displs[i] += (send_counts[i] - ret_recv_counts[i]);
    }

    // return the solved elements via a all2all
    mxx::all2allv(&recved[0], ret_send_counts, ret_send_displs, &lr_mins[0], ret_recv_counts, ret_recv_displs, comm);

    // local to global indexing transformation
    for (size_t i = 0; i < in.size(); ++i) {
        if(left_nsv[i] >= local_size) {
            if (lr_mins[left_nsv[i]-local_size].second == i+prefix) {
                left_nsv[i] = nonsv;
            } else {
                left_nsv[i] = lr_mins[left_nsv[i]-local_size].second;
            }
        } else {
            left_nsv[i] += prefix;
        }
        if (right_nsv[i] >= local_size) {
            if (lr_mins[right_nsv[i]-local_size].second == i+prefix) {
                right_nsv[i] = nonsv;
            } else {
                right_nsv[i] = lr_mins[right_nsv[i]-local_size].second;
            }
        } else {
            right_nsv[i] += prefix;
        }
    }
    t.end_section("ANSV: finish ansv local");
}

template<class PairIt, class T>
inline PairIt pair_lower_bound_dec(PairIt first, PairIt last, const T& value) {
    return std::lower_bound(
        first, last, std::pair<T, size_t>(value, 0),
        [](const std::pair<T, size_t>& x, const std::pair<T, size_t>& y) {
            return x.first > y.first;
        });
}


template <class Iterator, class T>
inline size_t range_displacement(Iterator first, Iterator last, const T* base_ptr) {
    static_assert(std::is_same<typename std::iterator_traits<Iterator>::value_type, T>::value, "Iterator must have value_type `T`.");
    MXX_ASSERT(&(*first) >= base_ptr);
    if (first == last || &(*first) <= &(*last-1)) {
        return &(*first) - base_ptr;
    } else {
        MXX_ASSERT(&(*(last-1)) >= base_ptr);
        return &(*(last-1)) - base_ptr;
    }
}

template <typename T, int indexing_type = global_indexing>
void my_ansv_minpair_lbub(const std::vector<T>& in, std::vector<size_t>& left_nsv, std::vector<size_t>& right_nsv, std::vector<std::pair<T,size_t> >& lr_mins, const mxx::comm& comm, size_t nonsv = 0) {
    mxx::section_timer t(std::cerr, comm);

    size_t local_size = in.size();
    size_t prefix = mxx::exscan(local_size, comm);

    /*****************************************************************
     *  Step 1: Locally calculate ANSV and save un-matched elements  *
     *****************************************************************/
    if (left_nsv.size() != in.size())
        left_nsv.resize(in.size());
    if (right_nsv.size() != in.size())
        right_nsv.resize(in.size());
    //size_t n_left_mins = local_ansv_unmatched<T, left_type, right_type>(in, prefix, lr_mins);
    local_indexing_nsv<decltype(in.rbegin()), T, nearest_sm, dir_left>(in.rbegin(), in.rend(), lr_mins, left_nsv);
    size_t n_left_mins = lr_mins.size();
    local_indexing_nsv<decltype(in.begin()), T, nearest_sm, dir_right>(in.begin(), in.end(), lr_mins, right_nsv);
    // change lrmin indexing to global
    for (size_t i = 0; i < lr_mins.size(); ++i) {
        lr_mins[i].second += prefix;
    }

    t.end_section("ANSV: local ansv");

    //mxx::sync_cout(comm) << "in=" << in << std::endl;
    //mxx::sync_cout(comm) << "lr_mins=" << lr_mins << std::endl;
    //mxx::sync_cout(comm) << "left_nsv=" << left_nsv << std::endl;
    //mxx::sync_cout(comm) << "right_nsv=" << right_nsv << std::endl;

    /***************************************************************
     *  Step 2: communicate un-matched elements to correct target  *
     ***************************************************************/
    std::vector<std::pair<T, size_t>> recved;
    //size_t n_left_recv = ansv_communicate_allpairs<T, nearest_sm, nearest_sm>(lr_mins, n_left_mins, recved, comm);
    t.end_section("ANSV: communicate all");
    //mxx::sync_cout(comm) << "recved=" << recved << std::endl;
    std::vector<size_t> send_counts(comm.size(), 0);
    std::vector<size_t> send_displs(comm.size(), 0);
    //ansv_comm_allpairs_params<T, furthest_eq, furthest_eq>(lr_mins, n_left_mins, send_counts, send_displs, comm);


    /*********************************************************************
     *                   get communication parameters                    *
     *********************************************************************/

    // allgather min and max of all left and right mins
    T local_min = lr_mins[n_left_mins].first;
    std::vector<T> allmins = mxx::allgather(local_min, comm);

    std::vector<size_t> lb_counts(comm.size(), 0);
    std::vector<size_t> lb_displs(comm.size(), 0);
    std::vector<size_t> in_counts(comm.size(), 0);
    std::vector<size_t> in_displs(comm.size(), 0);
    std::vector<size_t> ub_counts(comm.size(), 0);
    std::vector<size_t> ub_displs(comm.size(), 0);

    typedef typename std::vector<std::pair<T, size_t>>::iterator pair_it;
    size_t dir_offset = 0;
    pair_it lr_begin = lr_mins.begin();
    if (comm.rank() > 0) {
        pair_it left_begin = lr_mins.begin();
        pair_it left_end = lr_mins.begin()+n_left_mins;
        pair_it left_it = left_begin;

        pair_it lb_begin;
        pair_it lb_end;
        pair_it in_begin;
        pair_it in_end;
        pair_it ub_begin;
        pair_it ub_end;
        T prev_mins_min = allmins[comm.rank()-1];
        // calc comm parameters for first neighboring processor
        //while (left_it < left_end && left_it->first > allmins[comm.rank()-1])
        //    ++left_it;
        // now `left_it` is either the first equal element OR the end
        // in which case our neighbor has a smaller min than ourselves
        // thus we only have only an internal section, no lower or upper sections
        //in_begin = left_begin;
        //in_end = left_it;

        // initialize lower range as empty
        // lb <- [begin, begin)
        lb_begin = left_begin;
        lb_end = left_begin;

        for (int i = comm.rank()-1; i >= 0; --i) {
            if (i == comm.rank()-1 || allmins[i] < prev_mins_min) {
                // calculate bounds of inner range
                in_begin = lb_end;
                in_end = pair_lower_bound_dec(in_begin, left_end, allmins[i]);
                // calculate bounds of upper range
                // ub ends at the equal range of the first element larger than allmins
                // ub = [in_end, ub_end)
                ub_end = in_end;
                while (ub_end != left_end && ub_end->first <= allmins[i])
                    ++ub_end;
                pair_it ub_mid = ub_end;
                while (ub_end != left_end && ub_end->first == ub_mid->first)
                    ++ub_end;
                MXX_ASSERT(ub_end - in_end <= 4);

                // set send counts and offsets
                lb_counts[i] = std::distance(lb_begin, lb_end);
                lb_displs[i] = range_displacement(lb_begin, lb_end, &lr_mins[0]);

                in_counts[i] = std::distance(in_begin, in_end);
                in_displs[i] = range_displacement(in_begin, in_end, &lr_mins[0]);

                ub_counts[i] = std::distance(in_end, ub_end);
                ub_displs[i] = range_displacement(in_end, ub_end, &lr_mins[0]);

                // lb <- ub
                lb_begin = in_end;
                lb_end = ub_end;
            } else {
                // only send previous lower box
                lb_counts[i] = std::distance(lb_begin, lb_end);
                lb_displs[i] = range_displacement(lb_begin, lb_end, &lr_mins[0]);
            }

            // remember most min we have seen so far
            if (allmins[i] < prev_mins_min) {
                prev_mins_min = allmins[i];
            }
            // stop if we reached a processor with a smaller min than ours
            if (allmins[i] < local_min) {
                break;
            }
        }
        // TODO: generalize the above as a function (is symmetric as left and right)

        // TODO: upper is always send as part of inner, but only
        //       after determining inner min
        // TODO: lower is separate all2all communication step

        /*
        if (left_it == left_end) {
            // just internal section, nothing else
            // TODO: save internal section size
            //break; // TODO: doesn't do anything!
        } else {
            // TODO: replace with std::find_if, or std::lower_bound/std::equal_range
            // find lower bound
            lb_begin = left_it;
            // find lb_end: equal range
            while (left_it < left_end && lb_begin->first == left_it->first)
                ++left_it;
            // next equal range
            pair_it lb_mid = left_it;
            while (left_it < left_end && lb_mid->first == left_it->first)
                ++left_it;
            lb_end = left_it;
            MXX_ASSERT(lb_end - lb_begin <= 4);
        }
        */

        for (int i = comm.rank()-2; i >= 0; --i) {

            size_t start_idx = left_idx;
            // move start back to other `equal` elements (there can be at most 2)
            if (i < comm.rank()-1) {
                while (start_idx > 0 && lr_mins[start_idx-1].first <= prev_mins_min) {
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


    // determine min/max for each communication pair
    std::vector<size_t> min_send_counts(send_counts.begin(), send_counts.end());
    std::vector<size_t> min_recv_counts = mxx::all2all(min_send_counts, comm);
    // TODO: modify send/recv counts by selecting the minimum partner
    for (int i = 0; i < comm.size(); ++i) {
        if (min_recv_counts[i] != 0 && min_send_counts[i] != 0) {
            // choose the min and set the other one to 0
            if (min_recv_counts[i] < min_send_counts[i]) {
                min_send_counts[i] = 0;
            } else if (min_recv_counts[i] == min_send_counts[i]) {
                // need to handle this case so that its symmetric
                if (i < comm.rank()) {
                    min_recv_counts[i] = 0;
                } else {
                    min_send_counts[i] = 0;
                }
            } else {
                min_recv_counts[i] = 0;
            }
        }
    }

    SDEBUG(lr_mins);
    SDEBUG(send_counts);
    SDEBUG(min_send_counts);
    SDEBUG(min_recv_counts);

    std::vector<size_t> recv_displs = mxx::local_exscan(min_recv_counts);
    size_t recv_size = min_recv_counts.back() + recv_displs.back();
    recved = std::vector<std::pair<T, size_t>>(recv_size);
    // communicate through an all2all TODO: replace by send/recv + barrier?
    t.end_section("ANSV: calc comm params");
    mxx::all2allv(&lr_mins[0], min_send_counts, send_displs, &recved[0], min_recv_counts, recv_displs, comm);
    t.end_section("ANSV: all2allv");


    SDEBUG(recved);

    /***************************************************************
     *  Step 3: Again solve ANSV locally and use lr_mins as tails  *
     ***************************************************************/
    // use original send_counts as the range for the merge, and merge one sequence at a time
    // at the same time determine the send_counts for sending back solutions
    std::vector<size_t> ret_send_counts(min_recv_counts);
    std::vector<size_t> ret_send_displs(recv_displs);
    for (int i = comm.rank()-1; i >= 0; --i) {
        // local_range = send_displs[i] + [0, send_counts[i])
        if (min_recv_counts[i] > 0) {
            // local merge
            pair_it rec_begin = recved.begin()+recv_displs[i];
            pair_it rec_end = recved.begin()+recv_displs[i]+min_recv_counts[i];
            pair_it loc_begin = lr_mins.begin()+send_displs[i];
            pair_it loc_end = lr_mins.begin()+send_displs[i]+send_counts[i];
            pair_it rec_merge_end, loc_merge_end;
            std::tie(rec_merge_end, loc_merge_end) = ansv_merge(rec_begin, rec_end, loc_begin, loc_end);
            ret_send_counts[i] = min_recv_counts[i] - (rec_merge_end  - rec_begin + 1);
            ret_send_displs[i] += (rec_merge_end - rec_begin + 1);
        } else {
            // this section is waiting for the result from the remote
        }
    }
    for (int i = comm.rank()+1; i < comm.size(); ++i) {
        if (min_recv_counts[i] > 0) {
            pair_it loc_begin = lr_mins.begin()+send_displs[i];
            pair_it loc_end = lr_mins.begin()+send_displs[i]+send_counts[i];
            pair_it rec_begin = recved.begin()+recv_displs[i];
            pair_it rec_end = recved.begin()+recv_displs[i]+min_recv_counts[i];
            pair_it loc_merge_end, rec_merge_end;
            std::tie(loc_merge_end, rec_merge_end) = ansv_merge(loc_begin, loc_end, rec_begin, rec_end);
            ret_send_counts[i] = min_recv_counts[i] - (rec_end - rec_merge_end);
            // displacements stay the same, since we take elements away at the end of the sequence
        } else {
            // remote is answering my queries, so nothing to do here
        }
    }

    // get receive counts via all2all
    std::vector<size_t> ret_recv_counts = mxx::all2all(ret_send_counts, comm);
    std::vector<size_t> ret_recv_displs(send_displs);

    // re-calculate the receive displacementsjjj
    for (int i = comm.rank()+1; i < comm.size(); ++i) {
        ret_recv_displs[i] += (send_counts[i] - ret_recv_counts[i]);
    }

    // return the solved elements via a all2all
    mxx::all2allv(&recved[0], ret_send_counts, ret_send_displs, &lr_mins[0], ret_recv_counts, ret_recv_displs, comm);

    // local to global indexing transformation
    for (size_t i = 0; i < in.size(); ++i) {
        if(left_nsv[i] >= local_size) {
            if (lr_mins[left_nsv[i]-local_size].second == i+prefix) {
                left_nsv[i] = nonsv;
            } else {
                left_nsv[i] = lr_mins[left_nsv[i]-local_size].second;
            }
        } else {
            left_nsv[i] += prefix;
        }
        if (right_nsv[i] >= local_size) {
            if (lr_mins[right_nsv[i]-local_size].second == i+prefix) {
                right_nsv[i] = nonsv;
            } else {
                right_nsv[i] = lr_mins[right_nsv[i]-local_size].second;
            }
        } else {
            right_nsv[i] += prefix;
        }
    }
    t.end_section("ANSV: finish ansv local");
}

template <typename T> //, int left_type, int right_type>
void hh_ansv_comm_params(const std::vector<std::pair<T,size_t>>& lr_mins,
                         const std::vector<size_t>& lpm, const std::vector<size_t>& rpm,
                         const std::vector<T>& allmins,
                         size_t n_left_mins, std::vector<size_t>& send_counts,
                         std::vector<size_t>& send_offsets, const mxx::comm& comm) {

    send_counts = std::vector<size_t>(comm.size(), 0);
    send_offsets = std::vector<size_t>(comm.size(), 0);
    if (comm.rank() > 0) {
        // left processors
        size_t start_idx = 0;
        for (int i = comm.rank()-1; i >= 0; --i) {
            if ((int)rpm[i] == comm.rank()) {
                // determine the end of sequence S2x
                size_t end_idx = start_idx;
                while (end_idx+1<n_left_mins && lr_mins[end_idx].first >= allmins[i])
                    ++end_idx;
                send_counts[i] = end_idx - start_idx + 1;
                send_offsets[i] = start_idx;
                start_idx = end_idx;
            } else if (allmins[i] == allmins[comm.rank()] && rpm[i] == comm.size()) {
                // if the min on the target procis equal, send all remaining
                // to the left
                send_counts[i] = n_left_mins - start_idx;
                send_offsets[i] = start_idx;
                break;
            } else {
                //break;
            }
        }
    }
    if (comm.rank()+1 < comm.size()) {
        // comm params for right processors
        size_t end_idx = lr_mins.size()-1;
        for (int i = comm.rank()+1; i < comm.size(); ++i) {
            if ((int)lpm[i] == comm.rank()) {
                size_t start_idx = end_idx;
                while (start_idx > n_left_mins && lr_mins[start_idx].first >= allmins[i]) {
                    --start_idx;
                }
                send_counts[i] = end_idx - start_idx + 1;
                send_offsets[i] = start_idx;
                end_idx = start_idx;
            } else {
                //break;
            }
        }
    }
}


template <typename T>
size_t hh_ansv_comm(const std::vector<std::pair<T, size_t>>& lr_mins, const std::vector<size_t>& send_counts, const std::vector<size_t>& send_offsets, const mxx::comm& comm, std::vector<std::pair<T, size_t>>& recved) {
    // TODO: this is only for debug purposes:
    std::vector<size_t> recv_counts = mxx::all2all(send_counts, comm);

    SDEBUG(send_counts);
    SDEBUG(send_offsets);
    SDEBUG(recv_counts);
    // check if there is really just one processor sending me from each direction
    int recv_from_left = -1;
    int recv_from_right = -1;
    size_t n_right_recv = 0;
    size_t n_left_recv = 0;
    for (int i = 0; i < comm.size(); ++i) {
        if (recv_counts[i] > 0) {
            if (i < comm.rank()) {
                if (recv_from_left != -1) {
                    std::cerr << "[ERROR] receiving more than one from left" << std::endl;
                } else {
                    recv_from_left = i;
                    n_left_recv = recv_counts[i];
                }
            }
            if (i > comm.rank()) {
                if (recv_from_right != -1) {
                    std::cerr << "[ERROR] receiving more than one from right" << std::endl;
                } else {
                    recv_from_right = i;
                    n_right_recv = recv_counts[i];
                }
            }
        }
    }

    // actually send and receive
    mxx::datatype dt = mxx::get_datatype<std::pair<T,size_t>>();
    std::vector<MPI_Request> reqs;
    for (int i = 0; i < comm.size(); ++i) {
        if (send_counts[i] > 0) {
            // TODO: use proper mxx::isend and combine futures?
            reqs.push_back(MPI_REQUEST_NULL);
            MPI_Isend(&lr_mins[0]+send_offsets[i], (int)send_counts[i], dt.type(), i, 0, comm, &reqs[0]);
        }
    }

    recved = std::vector<std::pair<T, size_t>>(n_left_recv+n_right_recv);
    MPI_Request recv_req[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    if (recv_from_left != -1) {
        MPI_Irecv(&recved[0], n_left_recv, dt.type(), recv_from_left, 0, comm, &recv_req[0]);
    }
    if (recv_from_right != -1) {
        MPI_Irecv(&recved[0]+n_left_recv, n_right_recv, dt.type(), recv_from_right, 0, comm, &recv_req[1]);
    }

    // wait for all communication to finish
    MPI_Waitall(reqs.size(), &reqs[0], MPI_STATUSES_IGNORE);
    MPI_Waitall(2, recv_req, MPI_STATUSES_IGNORE);

    return n_left_recv;
}


template <typename T>
void hh_ansv(const std::vector<T>& in, std::vector<size_t>& left_nsv, std::vector<size_t>& right_nsv, std::vector<std::pair<T,size_t> >& lr_mins, const mxx::comm& comm, size_t nonsv = 0) {
    mxx::section_timer t(std::cerr, comm);
    size_t local_size = in.size();
    size_t prefix = mxx::exscan(local_size, comm);

    // TODO: first unmatched solution: solve nsv with local_indexing
    //       rather than just calculating the unmatched
    //size_t n_left_mins = local_ansv_unmatched<T, nearest_sm, nearest_sm>(in, prefix, lr_mins);
    if (left_nsv.size() != in.size())
        left_nsv.resize(in.size());
    if (right_nsv.size() != in.size())
        right_nsv.resize(in.size());
    //size_t n_left_mins = local_ansv_unmatched<T, left_type, right_type>(in, prefix, lr_mins);
    local_indexing_nsv<decltype(in.rbegin()), T, nearest_sm, dir_left>(in.rbegin(), in.rend(), lr_mins, left_nsv);
    size_t n_left_mins = lr_mins.size();
    local_indexing_nsv<decltype(in.begin()), T, nearest_sm, dir_right>(in.begin(), in.end(), lr_mins, right_nsv);
    // change lrmin indexing to global
    for (size_t i = 0; i < lr_mins.size(); ++i) {
        lr_mins[i].second += prefix;
    }
    t.end_section("ANSV: local ansv");

    // gather all processor minima
    T min = lr_mins[n_left_mins].first;
    std::vector<T> allmins = mxx::allgather(min, comm);

    // solve ANSV for the processor minima
    std::vector<size_t> lpm = ansv_sequential(allmins, true, comm.size());
    std::vector<size_t> rpm = ansv_sequential(allmins, false, comm.size());

    // get communication parameters
    std::vector<size_t> send_counts;
    std::vector<size_t> send_offsets;
    hh_ansv_comm_params(lr_mins, lpm, rpm, allmins, n_left_mins, send_counts, send_offsets, comm);


    std::vector<std::pair<T,size_t>> recved;
    size_t n_left_recv = hh_ansv_comm(lr_mins, send_counts, send_offsets, comm, recved);

    SDEBUG(in);
    SDEBUG(lr_mins);
    SDEBUG(lpm);
    SDEBUG(rpm);
    SDEBUG(recved);

    // calc where Seq1 is: determine `k` and then a_rm(min(k))
    int k1 = -1;
    int m1 = -1;
    auto seq1_end = lr_mins.end();
    // is there a left match?
    if ((int)rpm[comm.rank()] != comm.size()) {
        if ((int)rpm[comm.rank()] == comm.rank()+1) {
            k1 = comm.rank()+1;
        } else if ((int)rpm[comm.rank()] > comm.rank()+1) {
            //seq3_begin = // find k3, min(k3), and first element smaller than k3
            auto minit = std::min_element(allmins.begin() + (comm.rank()+1), allmins.begin()+rpm[comm.rank()]-1);
            k1 = minit - allmins.begin();
            T k1min = *minit;
            seq1_end = std::find_if(lr_mins.begin()+n_left_mins, lr_mins.end(), [&k1min](const std::pair<T,size_t>& x) { return k1min <= x.first; });
        }
    } else {
        // if there is an equal min, Seq1 is that
        auto all_minit = std::min_element(allmins.begin() + (comm.rank()+1), allmins.end());
        if (*all_minit == allmins[comm.rank()]) {
            //seq3_begin = // find k3, min(k3), and first element smaller than k3
            int rpm = all_minit - allmins.begin();
           if (rpm == comm.rank()+1) {
               m1 = comm.rank()+1;
               k1 = comm.rank()+1;
           } else if (rpm > comm.rank()+1) {
               //seq3_begin = // find k3, min(k3), and first element smaller than k3
               auto minit = std::min_element(allmins.begin() + (comm.rank()+1), allmins.begin()+rpm-1);
               k1 = minit - allmins.begin();
               m1 = rpm;
               T k1min = *minit;
               seq1_end = std::find_if(lr_mins.begin()+n_left_mins, lr_mins.end(), [&k1min](const std::pair<T,size_t>& x) { return k1min <= x.first; });
           }
        }
    }

    // calc where Seq3 is: determine (k') and then a_rm(min(k'))
    int k3 = -1;
    size_t seq3_begin = -1;
    // is there a left match?
    if ((int)lpm[comm.rank()] != comm.size()) {
        if ((int)lpm[comm.rank()] == comm.rank()-1) {
            seq3_begin = 0;
            k3 = comm.rank()-1;
        } else if ((int)lpm[comm.rank()] < comm.rank()-1) {
            //seq3_begin = // find k3, min(k3), and first element smaller than k3
            auto minit = std::min_element(allmins.begin()+lpm[comm.rank()]+1, allmins.begin()+(comm.rank()-1));
            k3 = minit - allmins.begin();
            T k3min = *minit;
            // TODO: find the first position in lr_mins which is smaller than 
            auto seq3_begin_it = std::find_if(lr_mins.begin(), lr_mins.begin()+n_left_mins, [&k3min](const std::pair<T,size_t>& x) { return x.first <= k3min; });
            seq3_begin = std::distance(lr_mins.begin(), seq3_begin_it);
        }
    }

    // solve locally both sides (i.e., execute a merge of S1 with S2, and S3 with S4)
    // TODO: actually calculate the exact sequence position for S1 and S3
    if (k1 >= 0) {
       ansv_merge(lr_mins.begin()+n_left_mins, seq1_end, recved.begin()+n_left_recv, recved.end());
    }
    // Seq3 = (a_rm(min(k'(i))), ..., a_min(i))
    if (k3 >= 0) {
        // merge receved Seq4 with my Seq3
        ansv_merge(recved.begin(), recved.begin()+n_left_recv, lr_mins.begin()+seq3_begin, lr_mins.begin()+n_left_mins);
    }

    SDEBUG(lr_mins);
    SDEBUG(recved);

    // send back S2, all but the last one
    std::vector<size_t> return_send_counts(comm.size(), 0);
    std::vector<size_t> return_send_displs(comm.size(), 0);
    if ((int)lpm[comm.rank()] != comm.size()) {
       return_send_counts[lpm[comm.rank()]] = n_left_recv - 1;
       return_send_displs[lpm[comm.rank()]] = 1;
    }
    if ((int)rpm[comm.rank()] != comm.size()) {
       return_send_counts[rpm[comm.rank()]] = recved.size() - n_left_recv - 1;
       return_send_displs[rpm[comm.rank()]] = n_left_recv;
    } else if (m1 != -1) {
       return_send_counts[m1] = recved.size() - n_left_recv - 1;
       return_send_displs[m1] = n_left_recv;
    }


    std::vector<size_t> return_recv_counts = send_counts;
    std::vector<size_t> return_recv_displs = send_offsets;
    for (int i = 0; i < comm.rank(); ++i) {
       if (return_recv_counts[i] > 0) {
          --return_recv_counts[i];
          //++return_recv_displs[i];
       }
    }
    for (int i = comm.rank() + 1; i < comm.size(); ++i) {
       if (return_recv_counts[i] > 0) {
          --return_recv_counts[i];
          ++return_recv_displs[i];
       }
    }

    SDEBUG(return_send_counts);
    SDEBUG(return_send_displs);
    SDEBUG(return_recv_counts);
    SDEBUG(return_recv_displs);
    // FIXME: currently cheating with an all2all ;)
    mxx::all2allv(&recved[0], return_send_counts, return_send_displs, &lr_mins[0], return_recv_counts, return_recv_displs, comm);
    SDEBUG(lr_mins);

    // local to global indexing transformation
    for (size_t i = 0; i < in.size(); ++i) {
        if(left_nsv[i] >= local_size) {
            if (lr_mins[left_nsv[i]-local_size].second == i+prefix) {
                left_nsv[i] = nonsv;
            } else {
                left_nsv[i] = lr_mins[left_nsv[i]-local_size].second;
            }
        } else {
            left_nsv[i] += prefix;
        }
        if (right_nsv[i] >= local_size) {
            if (lr_mins[right_nsv[i]-local_size].second == i+prefix) {
                right_nsv[i] = nonsv;
            } else {
                right_nsv[i] = lr_mins[right_nsv[i]-local_size].second;
            }
        } else {
            right_nsv[i] += prefix;
        }
    }
    SDEBUG(left_nsv);
    SDEBUG(right_nsv);
}

template <typename T, int left_type = nearest_sm, int right_type = nearest_sm>
void ansv(const std::vector<T>& in, std::vector<size_t>& left_nsv, std::vector<size_t>& right_nsv, const mxx::comm& comm) {
    std::vector<std::pair<T, size_t>> lr_mins;
    ansv<T, left_type, right_type, global_indexing>(in, left_nsv, right_nsv, lr_mins, comm);
}


#endif // ANSV_HPP
