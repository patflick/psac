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


template <typename T, int type>
inline void update_nsv_queue(std::vector<size_t>& nsv, std::deque<std::pair<T,size_t>>& q, const T& next_value, size_t idx, size_t prefix) {
    while (!q.empty() && next_value < q.back().first) {
        // current element is the min for the last element in the queue
        nsv[q.back().second-prefix] = idx;
        q.pop_back();
    }
    if (type == nearest_eq) {
        if (!q.empty() && next_value == q.back().first) {
            // replace the equal element
            nsv[q.back().second-prefix] = idx;
            q.pop_back();
        }
    }
}



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
size_t ansv_communicate_allpairs(const std::vector<std::pair<T,size_t>>& lr_mins, size_t n_left_mins, std::vector<std::pair<T, size_t>>& remote_seqs, const mxx::comm& comm) {
    mxx::section_timer t(std::cerr, comm);
    T local_min = lr_mins[n_left_mins].first;

    // allgather min and max of all left and right mins
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
    remote_seqs = std::vector<std::pair<T, size_t>>(recv_size);

    t.end_section("ANSV: calc comm params");
    mxx::all2allv(&lr_mins[0], send_counts, send_displs, &remote_seqs[0], recv_counts, recv_displs, comm);
    t.end_section("ANSV: all2allv");

    // solve locally given the exchanged values (by prefilling min-queue before running locally)
    size_t n_left_recv = recv_displs[comm.rank()];
    return n_left_recv;
}

constexpr bool dir_left = true;
constexpr bool dir_right = false;

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

template <typename T, int left_type, int right_type>
void hh_ansv_comm_params(const std::vector<std::pair<T,size_t>>& lr_mins,
                         size_t n_left_mins, std::vector<size_t> send_counts,
                         std::vector<size_t> send_offsets, const mxx::comm& comm) {
    T min = lr_mins[n_left_mins].first;
    // gather all processor minima
    std::vector<T> allmins = mxx::allgather(min, comm);

    // solve ANSV for the processor minima
    std::vector<size_t> lpm = ansv_sequential(allmins, true);
    std::vector<size_t> rpm = ansv_sequential(allmins, false);

    send_counts = std::vector<size_t>(comm.size(), 0);
    send_offsets = std::vector<size_t>(comm.size(), 0);
    if (comm.rank() > 0) {
        // left processors
        size_t start_idx = 0;
        for (int i = comm.rank()-1; i >= 0; --i) {
            if ((int)rpm[i] == comm.rank()) {
                // determine the end of sequence S2x
                size_t end_idx = start_idx;
                while (end_idx+1<n_left_mins && lr_mins[end_idx+1] >= allmins[i])
                    ++end_idx;
                send_counts[i] = end_idx - start_idx + 1;
                send_offsets[i] = start_idx;
                start_idx = end_idx;
            } else {
                break;
            }
        }
    }
    if (comm.rank()+1 < comm.size()) {
        // comm params for right processors
        size_t end_idx = lr_mins.size()-1;
        for (int i = comm.rank()+1; i < comm.size(); ++i) {
            if ((int)lpm[i] == comm.rank()) {
                size_t start_idx = end_idx;
                while (start_idx > n_left_mins && lr_mins[start_idx-1] >= allmins[i]) {
                    --start_idx;
                }
                send_counts[i] = end_idx - start_idx + 1;
                send_offsets[i] = start_idx;
                end_idx = start_idx;
            } else {
                break;
            }
        }
    }
}

template <typename Iterator>
void ansv_merge(Iterator left_begin, Iterator left_end, Iterator right_begin, Iterator right_end) {
    // starting with the largest value (right most in left sequence and leftmost in right sequence)
    Iterator l = left_end;
    Iterator r = right_begin;
    // TODO: one of the two sides need to write to the `nsv`, the other needs
    // to be prepared for sending out
    // TODO: global vs local indexing?
    while (l != left_begin && r != right_end) {
        if (l->first < r->first) {
            *r = *(l-1);
            ++r;
        } else if (r->first < l->first) {
            *(l-1) = *r;
            --l;
        } else {
            // r == l
            
        }
    }
}

template <typename T>
void hh_ansv(const std::vector<T>& in, std::vector<size_t>& nsv, std::vector<std::pair<T,size_t> >& lr_mins, const mxx::comm& comm, size_t nonsv = 0) {
    size_t local_size = in.size();
    size_t prefix = mxx::exscan(local_size, comm);

    size_t n_left_mins = local_ansv_unmatched<T, nearest_sm, nearest_sm>(in, prefix, lr_mins);

    // get communication parameters
    std::vector<size_t> send_counts;
    std::vector<size_t> send_offsets;
    hh_ansv_comm_params(lr_mins, n_left_mins, send_counts, send_offsets, comm);

    // TODO: this is only for debug purposes:
    std::vector<size_t> recv_counts = mxx::all2all(send_counts, comm);
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
            MPI_Isend(&lr_mins[0]+send_offsets[i], (int)send_counts[i], dt.type(), i, 0, &reqs[0]);
        }
    }

    std::vector<std::pair<T, size_t>> recved(n_left_recv+n_right_recv);
    MPI_Request recv_req[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    if (recv_from_left != -1) {
        MPI_Irecv(&recved[0], n_left_recv, dt.type(), recv_from_left, 0, &recv_req[0]);
    }
    if (recv_from_right != -1) {
        MPI_Irecv(&recved[0]+n_left_recv, n_right_recv, dt.type(), recv_from_right, 0, &recv_req[1]);
    }

    // wait for all communication to finish
    MPI_Waitall(reqs.size(), &reqs[0], MPI_STATUSES_IGNORE);
    MPI_Waitall(2, recv_req, MPI_STATUSES_IGNORE);

    // solve locally both sides (i.e., execute a merge of S1 with S2, and S3 with S4)

    // send answers back with same comm parameters
    
}

template <typename T, int left_type = nearest_sm, int right_type = nearest_sm>
void ansv(const std::vector<T>& in, std::vector<size_t>& left_nsv, std::vector<size_t>& right_nsv, const mxx::comm& comm) {
    std::vector<std::pair<T, size_t>> lr_mins;
    ansv<T, left_type, right_type, global_indexing>(in, left_nsv, right_nsv, lr_mins, comm);
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
        q.push_back(idx);
    }
    return nsv;
}

#endif // ANSV_HPP
