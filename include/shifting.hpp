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


// rough interface for distribution of data elements over the processors of a
// communicator
class dist_base {
protected:
    const mxx::comm& m_comm;
    const unsigned int m_comm_size, m_comm_rank;
    const size_t m_local_size;
    dist_base(const mxx::comm& comm, size_t local_size) :
        m_comm(comm), m_comm_size(comm.size()), m_comm_rank(comm.rank()), m_local_size(local_size) {
    }

public:
    inline size_t local_size() const {
        return m_local_size;
    }

    inline const mxx::comm& comm() const {
        return m_comm;
    }

    inline int comm_size() const {
        return m_comm_size;
    }
    inline int comm_rank() const {
        return m_comm_rank;
    }

    /* need to be implemented by all deriving classes
    inline size_t local_size(int rank) const;
    inline size_t global_size() const;
    inline size_t eprefix() const;
    inline size_t iprefix() const;
    inline size_t eprefix(int rank) const;
    inline size_t iprefix(int rank) const;

    inline int    rank_of(size_t gidx) const;
    inline size_t lidx_of(size_t gidx) const;
    inline size_t gidx_of(int rank, size_t lidx) const;
    */
};

class blk_dist_buf : public dist_base {
public:
    using dist_base::local_size;

    ~blk_dist_buf () {}

    /// collective allreduce for global size
    blk_dist_buf(const mxx::comm& comm, size_t local_size)
        : dist_base(comm, local_size),
          n(mxx::allreduce(local_size, comm)),
          div(n / m_comm_size), mod(n % m_comm_size),
          prefix(div*m_comm_rank + std::min<size_t>(mod, m_comm_rank)),
          div1mod((div+1)*mod)
    {
        assert(local_size == div + (m_comm_rank < mod ? 1 : 0));
    }

    blk_dist_buf(const blk_dist_buf& o) = default;
    blk_dist_buf& operator=(const blk_dist_buf& other) = default;


    inline size_t global_size() const {
        return n;
    }

    inline size_t local_size(unsigned int rank) const {
        return div + (rank < mod ? 1 : 0);
    }

    inline size_t iprefix() const {
        return prefix + m_local_size;
    }

    inline size_t iprefix(unsigned int rank) const {
        return div*(rank+1) + std::min<size_t>(mod, rank + 1);
    }

    inline size_t eprefix() const {
        return prefix;
    }

    inline size_t eprefix(unsigned int rank) const {
        return div*rank + std::min<size_t>(mod, rank);
    }

    // which processor the element with the given global index belongs to
    inline unsigned int rank_of(size_t gidx) const {
        if (gidx < div1mod) {
            // a_i is within the first n % p processors
            return gidx/(div+1);
        } else {
            return mod + (gidx - div1mod)/div;
        }
    }

    inline size_t lidx_of(size_t gidx) const {
        return gidx - eprefix(rank_of(gidx));
    }

    inline size_t gidx_of(int rank, size_t lidx) const {
        return eprefix(rank) + lidx;
    }

private:
    /* data */
    const size_t n;
    // derived/buffered values (for faster computation of results)
    const size_t div; // = n/p
    const size_t mod; // = n%p
    // local size (number of local elements)
    // the exclusive prefix (number of elements on previous processors)
    const size_t prefix;
    /// number of elements on processors with one more element
    const size_t div1mod; // = (n/p + 1)*(n % p)
};

using blk_dist = blk_dist_buf;

// block distributed (consecutive numbers, #elements same as cyclic)
/*
class blk_dist : public dist_base {
    // TODO: move implementation from partition.hpp into here!
    mxx::partition::block_decomposition_buffered<size_t> part;

    inline size_t local_size(int rank) const;
    inline size_t global_size() const;
    inline size_t eprefix() const;
    inline size_t iprefix() const;
    inline size_t eprefix(int rank) const;
    inline size_t iprefix(int rank) const;

    inline int    rank_of(size_t gidx) const;
    inline size_t lidx_of(size_t gidx) const;
    inline size_t gidx_of(int rank, size_t lidx) const;
};
*/

using blk_dist = blk_dist_buf;

class cyl_dist {
};

// simplified block distr: equal number of elements on each processor:
// exactly n/p (e.g.: the required input to bitonic sort)
class eq_dist : public dist_base {
public:
    eq_dist(const mxx::comm& comm, size_t local_size) : dist_base(comm, local_size) {}

    inline size_t local_size(int) {
        return dist_base::local_size();
    }

    inline size_t global_size() const {
        return dist_base::local_size() * comm_size();
    }

    inline size_t eprefix() const {
        return m_local_size * m_comm_rank;
    }

    inline size_t iprefix() const {
        return m_local_size * (m_comm_rank + 1);
    }

    inline size_t eprefix(int rank) const {
        return m_local_size * rank;
    }

    inline size_t iprefix(int rank) const {
        return m_local_size * (rank + 1);
    }

    inline int    rank_of(size_t gidx) const {
        return gidx / m_local_size;
    }

    inline size_t lidx_of(size_t gidx) const {
        return gidx % m_local_size;
    }

    inline size_t gidx_of(int rank, size_t lidx) const {
        return m_local_size * rank + lidx;
    }
};

// wraps around any distribution (initialized by only local_size (local number of elements))
// and provides the general distribution functions for converting between gidx <-> (pidx, lidx)
// representations, and the other helper functions
class gen_dist {
    // TODO constructing is collective
    // each processor has the whole inclusive prefix (which allows answering
    // most queries in two lookups) and rank_of using binary search
};

// checks whether the distribution (given by local_size) is block distirbuted
// or not.and initialized the according "backend"

// always collective
class dist_factory {
private:
    const size_t local_size;
    const mxx::comm& comm;
    size_t global_size;
public:
    dist_factory(const mxx::comm& comm, size_t local_size) : local_size(local_size), comm(comm), global_size(mxx::allreduce(local_size, comm)) {
    }

    inline bool is_equal_dist() {
        return mxx::all_same(local_size, comm);
    }

    eq_dist to_equal_dist() {
        assert(is_equal_dist());
        return eq_dist(comm, local_size);
    }

    inline bool is_blk_dist() {
        size_t expected = global_size / comm.size() + ((global_size % comm.size() < static_cast<size_t>(comm.rank())) ? 1 : 0);
        bool is_blk = local_size == expected;
        return mxx::all_of(is_blk, comm);
    }

    blk_dist to_blk_dist() {
        assert(is_blk_dist());
        return blk_dist(comm, local_size);
    }
};

// distributed range wrapper
template <typename T, typename Dist>
class drange : public Dist {
};


template <typename T, typename dist>
class dvector : public dist {
public:
    using dist_type = dist;
    using value_type = T;

    std::vector<T> vec;

    dvector(const mxx::comm& c, size_t local_size) : dist(c, local_size), vec(local_size) {}


    inline T* data() {
        return vec.data();
    }
    inline const T* data() const {
        return vec.data();
    }
    inline const T* data_at(size_t offset) const {
        return data() + offset;
    }
    inline T* data_at(size_t offset) {
        return data() + offset;
    }


    /* iterators */

    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    iterator begin() {
        return vec.begin();
    }

    iterator end() {
        return vec.end();
    }

    const_iterator begin() const {
        return vec.begin();
    }

    const_iterator end() const {
        return vec.end();
    }
};

template <typename T, typename dist>
class dvector_wrapper : public dist {
public:
    using dist_type = dist;
    using value_type = T;

    std::vector<T>& vec;
    dvector_wrapper(std::vector<T>& vec, const mxx::comm& comm)
        : dist(comm, vec.size()), vec(vec) {
    }

    /* data access */

    inline T* data() {
        return vec.data();
    }
    inline const T* data() const {
        return vec.data();
    }
    inline const T* data_at(size_t offset) const {
        return data() + offset;
    }
    inline T* data_at(size_t offset) {
        return data() + offset;
    }

    /* iterators */

    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    iterator begin() {
        return vec.begin();
    }

    iterator end() {
        return vec.end();
    }

    const_iterator begin() const {
        return vec.begin();
    }

    const_iterator end() const {
        return vec.end();
    }
};

template <typename T, typename dist>
class dvector_const_wrapper : public dist {
public:
    using dist_type = dist;
    using value_type = T;

    const std::vector<T>& vec;
    dvector_const_wrapper(const std::vector<T>& vec, const mxx::comm& comm)
        : dist(comm, vec.size()), vec(vec) {
    }

    /* data access */
    inline const T* data() const {
        return vec.data();
    }
    inline const T* data_at(size_t offset) const {
        return data() + offset;
    }

    /* iterators */
    using const_iterator = typename std::vector<T>::const_iterator;

    const_iterator begin() const {
        return vec.begin();
    }

    const_iterator end() const {
        return vec.end();
    }
};

// TODO how to dynamically decide whether its equally distributed and without using virtual table lookup having inlined functions??
// TODO: need to be able to call the original function with different type
// Thus: dynamically check distribution for each target function and then
// call different type instantiations?
// oh C++, how do I do that again? poops
// meta programming ftw, no way around if else?

/*
 * Input: local_size, comm OR begin, end, comm, OR vector, comm
 * 1) if is_eq_dist(local-size, comm) -> eq_dist range
 *    if is_blk_distributed(local_size, comm) -> blk_dist range
 *    else gen_dist
 *
 *
 */
/*
mxx::requests isend_to_global_range(const std::vector<T>& gvec, size_t src_begin, size_t src_end, size_t dst_begin, size_t dst_end, const mxx::comm& comm) {
}
*/

template <template<class, class> class DRange, typename T, typename D>
mxx::requests isend_to_global_range(const DRange<T, D>& dr, size_t src_begin, size_t src_end, size_t dst_begin, size_t dst_end) {
    assert(src_end > src_begin);
    assert(dst_end > dst_begin);
    assert(src_end - src_begin == dst_end - dst_begin);

    size_t prefix = dr.eprefix();
    assert(dr.eprefix() <= src_begin && src_end <= dr.iprefix());

    mxx::requests r;
    size_t send_size = src_end - src_begin;
    // possibly split [dst_begin, dst_end) by distribution
    size_t recv_begin = dst_begin;
    size_t send_begin = src_begin;
    int p = dr.rank_of(dst_begin);
    while (send_size > 0) {
        size_t pend = std::min<size_t>(dst_end, dr.iprefix(p));
        size_t send_cnt = pend - recv_begin;
        mxx::datatype dt = mxx::get_datatype<T>();
        MPI_Isend(dr.data_at(send_begin-prefix), send_cnt, dt.type(), p, 0, dr.comm(), &r.add());
        recv_begin += send_cnt;
        send_begin += send_cnt;
        send_size -= send_cnt;
        ++p;
    }
    return r;
}

/*
template <typename T>
mxx::requests irecv_from_global_range(std::vector<T>& dst, size_t src_begin, size_t src_end, size_t dst_begin, size_t dst_end, const mxx::comm& comm) {
}
*/

template <template<class, class> class DRange, typename T, typename D>
mxx::requests irecv_from_global_range(DRange<T, D>& dr, size_t src_begin, size_t src_end, size_t dst_begin, size_t dst_end) {
    assert(src_end > src_begin);
    assert(dst_end > dst_begin);
    assert(src_end - src_begin == dst_end - dst_begin);

    size_t prefix = dr.eprefix();
    size_t local_size = dr.local_size();
    assert(prefix <= dst_begin && dst_end <= prefix + local_size);

    mxx::requests r;
    //size_t send_size = src_end - src_begin;
    size_t recv_size = dst_end - dst_begin;
    // possibly split [dst_begin, dst_end) by distribution
    size_t recv_begin = dst_begin;
    size_t send_begin = src_begin;
    int p = dr.rank_of(send_begin);
    while (recv_size > 0) {
        size_t pend = std::min<size_t>(src_end, dr.iprefix(p));
        size_t recv_cnt = pend - send_begin;
        mxx::datatype dt = mxx::get_datatype<T>();
        MPI_Irecv(dr.data_at(recv_begin-prefix), recv_cnt, dt.type(), p, 0, dr.comm(), &r.add());
        recv_begin += recv_cnt;
        send_begin += recv_cnt;
        recv_size -= recv_cnt;
        ++p;
    }
    return r;
}



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

template <typename DRangeSrc, typename DRangeDst>
mxx::requests icopy_global_range(const DRangeSrc& src, size_t src_begin, size_t src_end, DRangeDst& dst, size_t dst_begin, size_t dst_end) {
    using Tsrc = typename DRangeSrc::value_type;
    using Tdst = typename DRangeDst::value_type;
    // TODO: relax once ranges have their own datatype (possibly MPI_Vector with skips)
    static_assert(std::is_same<Tsrc, Tdst>::value, "Types for receiving and sending range must be the same");

    assert(src_begin < src_end);
    assert(dst_begin < dst_end);
    assert(src_end - src_begin == dst_end - dst_begin);

    mxx::requests req;

    // truncate for send
    size_t my_src_begin = std::max(src_begin, src.eprefix());
    size_t my_src_end = std::min(src_end, src.iprefix());
    if (my_src_begin < my_src_end) {
        // send
        size_t re_dst_begin = (my_src_begin - src_begin) + dst_begin;
        size_t re_dst_end = re_dst_begin + (my_src_end - my_src_begin);
        req.insert(isend_to_global_range(src, my_src_begin, my_src_end, re_dst_begin, re_dst_end));
    }

    // truncate for receive
    size_t my_dst_begin = std::max(dst_begin, src.eprefix());
    size_t my_dst_end = std::min(dst_end, src.iprefix());
    if (my_dst_begin < my_dst_end) {
        // receive
        size_t re_src_begin = (my_dst_begin - dst_begin) + src_begin;
        size_t re_src_end = re_src_begin + (my_dst_end - my_dst_begin);
        req.insert(irecv_from_global_range(dst, re_src_begin, re_src_end, my_dst_begin, my_dst_end));
    }
    return req;
}

template <typename DRangeSrc, typename DRangeDst>
void copy_global_range(const DRangeSrc& src, size_t src_begin, size_t src_end, DRangeDst& dst, size_t dst_begin, size_t dst_end) {
    icopy_global_range(src, src_begin, src_end, dst, dst_begin, dst_end).waitall();
}

template <typename drange>
class dbuckets {
    const drange& dr;
    std::vector<size_t> local_begins;
    std::vector<std::pair<size_t,size_t>> split_buckets;

    std::pair<size_t, size_t> inner_buckets;
    //dbuckets(drange, func) {}
    dbuckets(const drange& dr, const std::vector<size_t>& local_begins) : dr(dr), local_begins(local_begins) {}

    void init_split_buckets(const std::vector<size_t>& local_begins) {
        // for each bucket that is split across this processors boundary,
        // we save the [begin, end) global indexes
        //enum split_types {NO_SPLITS=0, LEFT_SPLIT=1, RIGHT_SPLIT=2, BOTH_SPLIT=3};
        //split_types split_type;

        // assume array of global indexes as "bucket splitters"
        size_t last_bucket_start, first_bucket_end;
        if (local_begins.size() == 0) {
            first_bucket_end = std::numeric_limits<size_t>::max();
            last_bucket_start = 0;
            if (dr.comm().rank() == dr.comm().size() - 1) {
                first_bucket_end = dr.global_size();
            }
        } else {
            first_bucket_end = local_begins.front();
            last_bucket_start = local_begins.back();
        }

        size_t left_last_start = mxx::exscan(last_bucket_start, mxx::max<size_t>(), dr.comm());
        size_t right_first_end = mxx::exscan(first_bucket_end, mxx::min<size_t>(), dr.comm().reverse());

        // check how sequences are split
        //split_type = NO_SPLITS;

        // First check if the sequence extends beyond left and right process
        if (dr.comm().rank() != 0 && dr.comm().rank() != dr.comm().size() - 1 && local_begins.empty() && right_first_end != dr.iprefix()) {
            // bucket extends to left and right processes
            split_buckets.emplace_back(left_last_start, right_first_end);
            //split_type = BOTH_SPLIT;
            // no inner buckets
            inner_buckets.first = inner_buckets.second = 0;
        } else {
            // check if left-most bucket extends to previous processor
            if (dr.comm().rank() != 0 && first_bucket_end != dr.eprefix()) {
                split_buckets.emplace_back(left_last_start, first_bucket_end);
            }

            // check if right-most bucket extends to next processor
            if (dr.comm().rank() != dr.comm().size() - 1 && right_first_end != dr.iprefix()) {
                split_buckets.emplace_back(last_bucket_start, right_first_end);
            }
        }
    }

    size_t local_num_splits() {
    }

    bool local_has_splits() {
    }

    // TODO: rename?
    inline std::vector<std::pair<size_t, size_t>> local_split_buckets() const {
        return split_buckets;
    }
};

// TODO: test and use-cases for distributed buckets!
// TODO: test and use case for distributed strings (with splits!)


template <typename StringSet>
std::vector<index_t> shift_buckets_ss_wsplit(const StringSet& ss, const std::vector<index_t>& vec, std::size_t dist) {
    size_t prefix = part.excl_prefix_size();

    // for each bucket: shift
    std::vector<index_t> result(vec.size());

    dvector_const_wrapper<index_t, blk_dist> src(vec, comm);
    dvector_wrapper<index_t, blk_dist> dst(result, comm);

    // for each bucket which is split across processors, use global range communication
    mxx::requests req;
    for (auto s : dbuckets.split_buckets()) {
        // icopy range based on bucket range and distance
        size_t ssize = s.second - s.first;
        if (dist < ssize) {
            req.insert(icopy_global_range(src, s.first + dist, s.second, dst, s.first, s.second - dist));
        }
        // otherwise fill with 0?
    }

    // for all purely internal buckets: shift using simple std::copy
    for (auto s : dbuckets.internal_buckets()) {
        size_t ssize = s.size();

        if (dist < ssize) {
            std::copy(iit+dist, iit+ssize, oit);
        }
    }



    r.waitall();
}

#endif // SHIFTING_HPP
