/*
 * Copyright 2017 Georgia Institute of Technology
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

#ifndef SHIFTING_HPP
#define SHIFTING_HPP

#include <assert.h>
#include <vector>

#include <mxx/comm.hpp>
#include <mxx/future.hpp>
#include <mxx/partition.hpp>


/*********************************************************************
 *               Shifting buckets (i -> i + 2^l) => B2               *
 *********************************************************************/

template <typename T>
std::vector<T> shift_vector(const std::vector<T>& vec, const mxx::blk_dist& dist, std::size_t shift_by, const mxx::comm& comm) {
    // get # elements to the left
    assert(dist.local_size() == vec.size());
    size_t local_size = vec.size();
    size_t prev_size = dist.eprefix_size();


    std::vector<T> result(local_size);

    mxx::datatype mpidt = mxx::get_datatype<T>();

    MPI_Request recv_reqs[2];
    int n_irecvs = 0;
    // receive elements from the right
    if (prev_size + shift_by < dist.global_size()) {
        std::size_t right_first_gl_idx = prev_size + shift_by;
        int p1 = dist.rank_of(right_first_gl_idx);

        std::size_t p1_gl_end = dist.iprefix_size(p1);
        std::size_t p1_recv_cnt = p1_gl_end - right_first_gl_idx;

        if (p1 != comm.rank()) {
            // only receive if the source is not myself (i.e., `rank`)
            // [otherwise results are directly written instead of MPI_Sended]
            assert(p1_recv_cnt < std::numeric_limits<int>::max());
            // increase iterators
            int recv_cnt = p1_recv_cnt;
            MPI_Irecv(&result[0],recv_cnt, mpidt.type(), p1,
                      0, comm, &recv_reqs[n_irecvs++]);
        }

        if (p1_recv_cnt < local_size && p1 != comm.size()-1) {
            // also receive from one more processor
            int p2 = p1+1;
            // since p2 has at least local_size - 1 elements and at least
            // one element came from p1, we can assume that the receive count
            // is our local size minus the already received elements
            std::size_t p2_recv_cnt = local_size - p1_recv_cnt;

            assert(p2_recv_cnt < std::numeric_limits<int>::max());
            int recv_cnt = p2_recv_cnt;
            // send to `p1` (which is necessarily different from `rank`)
            MPI_Irecv(&result[0] + p1_recv_cnt, recv_cnt, mpidt.type(), p2,
                      0, comm, &recv_reqs[n_irecvs++]);
        }
    }

    // send elements to the left (split to at most 2 target processors)
    if (prev_size + local_size - 1 >= shift_by) {
        int p1 = -1;
        if (prev_size >= shift_by) {
            std::size_t first_gl_idx = prev_size - shift_by;
            p1 = dist.rank_of(first_gl_idx);
        }
        std::size_t last_gl_idx = prev_size + local_size - 1 - shift_by;
        int p2 = dist.rank_of(last_gl_idx);

        std::size_t local_split;
        if (p1 != p2) {
            // local start index of area for second processor
            if (p1 >= 0) {
                local_split = dist.iprefix_size(p1) + shift_by - prev_size;
                // send to first processor
                assert(p1 != comm.rank());
                MPI_Send(const_cast<T*>(&vec[0]), local_split, mpidt.type(), p1, 0, comm);
            } else {
                // p1 doesn't exist, then there is no prefix to add
                local_split = shift_by - prev_size;
            }
        } else {
            // only one target processor
            local_split = 0;
        }

        if (p2 != comm.rank()) {
            MPI_Send(const_cast<T*>(&vec[0] + local_split), local_size - local_split, mpidt.type(), p2, 0, comm);
        } else {
            // in this case the split should be exactly at `shift_by`
            assert(local_split == shift_by);
            // locally reassign
            for (std::size_t i = local_split; i < local_size; ++i) {
                result[i-local_split] = vec[i];
            }
        }
    }

    // wait for successful receive:
    MPI_Waitall(n_irecvs, recv_reqs, MPI_STATUS_IGNORE);
    return result;
}

// shifting with arrays (custom data types)
// shifts (L-1) times into the (L-1) additional Bucket entries
template <typename T, std::size_t L>
void multi_shift_inplace(std::vector<std::array<T, 1+L> >& tuples, mxx::blk_dist& dist, size_t shift_by, const mxx::comm& comm) {
    // get # elements to the left
    size_t local_size = tuples.size();
    assert(local_size == dist.local_size());
    std::size_t prev_size = dist.eprefix_size();

    mxx::datatype mpidt = mxx::get_datatype<T>();

    // start receiving into second bucket and then continue with greater
    int bi = 2;
    for (std::size_t k = shift_by; k < L*shift_by; k += shift_by) {
        MPI_Request recv_reqs[2];
        int n_irecvs = 0;
        MPI_Datatype dts[4];
        int n_dts = 0;
        // receive elements from the right
        if (prev_size + k < dist.global_size()) {
            std::size_t right_first_gl_idx = prev_size + k;
            int p1 = dist.rank_of(right_first_gl_idx);

            std::size_t p1_gl_end = dist.iprefix_size(p1);
            std::size_t p1_recv_cnt = p1_gl_end - right_first_gl_idx;

            if (p1 != comm.rank()) {
                // only receive if the source is not myself (i.e., `rank`)
                // [otherwise results are directly written instead of MPI_Sended]
                assert(p1_recv_cnt < std::numeric_limits<int>::max());
                int recv_cnt = p1_recv_cnt;
                // create custom datatype with stride (L+1)
                MPI_Type_vector(recv_cnt,1,L+1,mpidt.type(),&dts[n_dts]);
                MPI_Type_commit(&dts[n_dts]);
                MPI_Irecv(&tuples[0][bi],1, dts[n_dts], p1,
                          0, comm, &recv_reqs[n_irecvs++]);
                n_dts++;
            }

            if (p1_recv_cnt < local_size && p1 != comm.size()-1) {
                // also receive from one more processor
                int p2 = p1+1;
                // since p2 has at least local_size - 1 elements and at least
                // one element came from p1, we can assume that the receive count
                // is our local size minus the already received elements
                std::size_t p2_recv_cnt = local_size - p1_recv_cnt;

                assert(p2_recv_cnt < std::numeric_limits<int>::max());
                int recv_cnt = p2_recv_cnt;
                // send to `p1` (which is necessarily different from `rank`)
                MPI_Type_vector(recv_cnt,1,L+1,mpidt.type(),&dts[n_dts]);
                MPI_Type_commit(&dts[n_dts]);
                MPI_Irecv(&tuples[p1_recv_cnt][bi],1, dts[n_dts], p2,
                          0, comm, &recv_reqs[n_irecvs++]);
                n_dts++;
            }
        }

        // send elements to the left (split to at most 2 target processors)
        if (prev_size + local_size - 1 >= k) {
            int p1 = -1;
            if (prev_size >= k) {
                std::size_t first_gl_idx = prev_size - k;
                p1 = dist.rank_of(first_gl_idx);
            }
            std::size_t last_gl_idx = prev_size + local_size - 1 - k;
            int p2 = dist.rank_of(last_gl_idx);

            std::size_t local_split;
            if (p1 != p2) {
                // local start index of area for second processor
                if (p1 >= 0) {
                    local_split = dist.iprefix_size(p1) + k - prev_size;
                    // send to first processor
                    assert(p1 != comm.rank());
                    MPI_Type_vector(local_split,1,L+1,mpidt.type(),&dts[n_dts]);
                    MPI_Type_commit(&dts[n_dts]);
                    MPI_Send(&tuples[0][1], 1,
                             dts[n_dts], p1, 0, comm);
                    n_dts++;
                } else {
                    // p1 doesn't exist, then there is no prefix to add
                    local_split = k - prev_size;
                }
            } else {
                // only one target processor
                local_split = 0;
            }

            if (p2 != comm.rank()) {
                MPI_Type_vector(local_size - local_split,1,L+1,mpidt.type(),&dts[n_dts]);
                MPI_Type_commit(&dts[n_dts]);
                MPI_Send(&tuples[local_split][1], 1,
                         dts[n_dts], p2, 0, comm);
                n_dts++;
            } else {
                // in this case the split should be exactly at `k`
                assert(local_split == k);
                // locally reassign
                for (std::size_t i = local_split; i < local_size; ++i) {
                    tuples[i-local_split][bi] = tuples[i][1];
                }
            }
        }

        // wait for successful receive:
        MPI_Waitall(n_irecvs, recv_reqs, MPI_STATUS_IGNORE);

        // clean up data types from this round
        for (int i = 0; i < n_dts; ++i) {
            MPI_Type_free(&dts[i]);
        }

        // next target bucket
        bi++;
    }
}


template <typename T>
mxx::requests isend_to_global_range(const std::vector<T>& src, mxx::blk_dist& dist, size_t src_begin, size_t src_end, size_t dst_begin, size_t dst_end, const mxx::comm& comm) {
    assert(src_end > src_begin);
    assert(dst_end > dst_begin);
    assert(src_end - src_begin == dst_end - dst_begin);

    size_t prefix = dist.eprefix_size();
    //assert(dr.eprefix() <= src_begin && src_end <= dr.iprefix());

    mxx::requests r;
    size_t send_size = src_end - src_begin;
    // possibly split [dst_begin, dst_end) by distribution
    size_t recv_begin = dst_begin;
    size_t send_begin = src_begin;
    int p = dist.rank_of(dst_begin);
    while (send_size > 0) {
        size_t pend = std::min<size_t>(dst_end, dist.iprefix_size(p));
        size_t send_cnt = pend - recv_begin;
        mxx::datatype dt = mxx::get_datatype<T>();
        MPI_Isend(const_cast<T*>(&src[send_begin-prefix]), send_cnt, dt.type(), p, 0, comm, &r.add());
        recv_begin += send_cnt;
        send_begin += send_cnt;
        send_size -= send_cnt;
        ++p;
    }
    return r;
}


template <typename T>
mxx::requests irecv_from_global_range(std::vector<T>& dst, mxx::blk_dist& dist, size_t src_begin, size_t src_end, size_t dst_begin, size_t dst_end, const mxx::comm& comm) {
    assert(src_end > src_begin);
    assert(dst_end > dst_begin);
    assert(src_end - src_begin == dst_end - dst_begin);

    size_t prefix = dist.eprefix_size();
    size_t local_size = dist.local_size();
    assert(prefix <= dst_begin && dst_end <= prefix + local_size);

    mxx::requests r;
    //size_t send_size = src_end - src_begin;
    size_t recv_size = dst_end - dst_begin;
    // possibly split [dst_begin, dst_end) by distribution
    size_t recv_begin = dst_begin;
    size_t send_begin = src_begin;
    int p = dist.rank_of(send_begin);
    while (recv_size > 0) {
        size_t pend = std::min<size_t>(src_end, dist.iprefix_size(p));
        size_t recv_cnt = pend - send_begin;
        mxx::datatype dt = mxx::get_datatype<T>();
        MPI_Irecv(&dst[recv_begin-prefix], recv_cnt, dt.type(), p, 0, comm, &r.add());
        recv_begin += recv_cnt;
        send_begin += recv_cnt;
        recv_size -= recv_cnt;
        ++p;
    }
    return r;
}



/*

template <template<class, class> class DRange, typename T, typename D>
dvector<T, typename DRange<T, D>::dist_type> left_shift_drange(const DRange<T, D>& src, size_t shift_by) {
    using result_type = dvector<T, typename DRange<T, D>::dist_type>;
    result_type result(src.comm(), src.local_size());

    // receive from right
    size_t src_begin = std::min(src.eprefix() + shift_by, src.global_size());
    size_t src_end = std::min(src.iprefix() + shift_by, src.global_size());
    mxx::requests req;
    if (src_begin < src_end) {
        req.insert(irecv_from_global_range(result, src_begin, src_end, src_begin - shift_by, src_end - shift_by));
    }

    // send to left
    size_t dst_begin = (src.eprefix() <= shift_by) ? 0 : src.eprefix() - shift_by;
    size_t dst_end = (src.iprefix() <= shift_by) ? 0 : src.iprefix() - shift_by;

    if (dst_begin < dst_end) {
        req.insert(isend_to_global_range(src, dst_begin + shift_by, dst_end + shift_by, dst_begin, dst_end));
    }

    req.waitall();

    return result;
}

template <typename T>
std::vector<T> left_shift_dvec(const std::vector<T>& vec, const mxx::comm& comm, size_t shift_by) {
    dvector_const_wrapper<T, blk_dist> src(vec, comm);
    dvector<T, blk_dist> result = left_shift_drange(src, shift_by);
    return result.vec;
}
*/

template <typename T>
mxx::requests icopy_global_range(const std::vector<T>& src, mxx::blk_dist& dist, size_t src_begin, size_t src_end, std::vector<T>& dst, size_t dst_begin, size_t dst_end, const mxx::comm& comm) {
    assert(src_begin < src_end);
    assert(dst_begin < dst_end);
    assert(src_end - src_begin == dst_end - dst_begin);

    mxx::requests req;

    size_t eprefix = dist.eprefix_size();
    size_t iprefix = dist.iprefix_size();

    // truncate for send
    size_t my_src_begin = std::max(src_begin, eprefix);
    size_t my_src_end = std::min(src_end, iprefix);
    if (my_src_begin < my_src_end) {
        // send
        size_t re_dst_begin = (my_src_begin - src_begin) + dst_begin;
        size_t re_dst_end = re_dst_begin + (my_src_end - my_src_begin);
        req.insert(isend_to_global_range(src, dist, my_src_begin, my_src_end, re_dst_begin, re_dst_end, comm));
    }

    // truncate for receive
    size_t my_dst_begin = std::max(dst_begin, eprefix);
    size_t my_dst_end = std::min(dst_end, iprefix);
    if (my_dst_begin < my_dst_end) {
        // receive
        size_t re_src_begin = (my_dst_begin - dst_begin) + src_begin;
        size_t re_src_end = re_src_begin + (my_dst_end - my_dst_begin);
        req.insert(irecv_from_global_range(dst, dist, re_src_begin, re_src_end, my_dst_begin, my_dst_end, comm));
    }
    return req;
}


template <typename DistSeqs, typename T>
std::vector<T> shift_buckets_ds(const DistSeqs& ss, const std::vector<T>& vec, std::size_t shift_by, const mxx::comm& comm, T fill = T()) {

    // for each bucket: shift
    std::vector<T> result(vec.size(), fill);

    size_t local_size = vec.size();
    size_t global_size = mxx::allreduce(local_size, comm);
    mxx::blk_dist dist(global_size, comm.size(), comm.rank());

    // for each bucket which is split across processors, use global range communication
    mxx::requests req;
    for (auto s : ss.split_seqs()) {
        // icopy range based on bucket range and distance
        size_t ssize = s.second - s.first;
        if (shift_by < ssize) {
            req.insert(icopy_global_range(vec, dist, s.first + shift_by, s.second, result, s.first, s.second - shift_by, comm));
        }
    }

    // for all purely internal buckets: shift using simple std::copy
    if (ss.has_inner_seqs() > 0) {
        size_t sb = ss.prefix_sizes[0] - dist.eprefix_size();
        auto iit = vec.begin() + sb;
        auto oit = result.begin() + sb;
        for (size_t i = 0; i < ss.prefix_sizes.size()-1; ++i) {
            size_t ssize = ss.prefix_sizes[i+1] - ss.prefix_sizes[i];
            if (shift_by < ssize) {
                std::copy(iit+shift_by, iit+ssize, oit);
            }
            iit += ssize;
            oit += ssize;
        }
        if (!ss.is_right_split()) {
            size_t ssize = ss.right_sep - ss.prefix_sizes.back();
            if (shift_by < ssize) {
                std::copy(iit+shift_by, iit+ssize, oit);
            }
        }
    }

    req.waitall();

    return result;
}


#endif // SHIFTING_HPP
