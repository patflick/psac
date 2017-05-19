#ifndef DVECTOR_HPP
#define DVECTOR_HPP

#include <vector>
#include <iterator>

#include <mxx/comm.hpp>

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

#endif // DVECTOR_HPP
