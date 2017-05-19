#ifndef BUCKETING_HPP
#define BUCKETING_HPP

#include <vector>
#include <utility>
#include <functional>

#include <mxx/comm.hpp>
#include <mxx/shift.hpp>



template <typename T1, typename T2>
struct pair_sum {
    std::pair<T1,T2> operator()(const std::pair<T1,T2>& x, const std::pair<T1,T2>& y) {
        return std::pair<T1,T2>(x.first+y.first,x.second+y.second);
    }
};


template <typename T>
void global_fill_where_zero(std::vector<T>& vec, const mxx::comm& comm) {
    /*
     * Global prefix MAX:
     *  - such that for every item we have it's bucket number, where the
     *    bucket number is equal to the first index in the bucket
     *    this way buckets who are finished, will never receive a new
     *    number.
     */
    // 1.) find the max in the local sequence. since the max is the last index
    //     of a bucket, this should be somewhere at the end -> start scanning
    //     from the end
    auto rev_it = vec.rbegin();
    T local_max = 0;
    while (rev_it != vec.rend() && (local_max = *rev_it) == 0)
        ++rev_it;

    // 2.) distributed scan with max() to get starting max for each sequence
    size_t pre_max = mxx::exscan(local_max, mxx::max<size_t>(), comm);
    if (comm.rank() == 0)
        pre_max = 0;

    // 3.) linear scan and assign bucket numbers
    for (size_t i = 0; i < vec.size(); ++i) {
        if (vec[i] == 0)
            vec[i] = pre_max;
        else
            pre_max = vec[i];
        //assert(local_B[i] <= i+prefix+1);
        // first element of bucket has id of it's own global index:
        //assert(i == 0 || (local_B[i-1] ==  local_B[i] || local_B[i] == i+prefix+1));
    }
}

// assumed sorted order (globally) by tuple (B1[i], B2[i])
// this reassigns new, unique bucket numbers in {1,...,n} globally
template <typename T, typename EqualFunc>
std::pair<size_t,size_t> rebucket(std::vector<T>& v1, std::vector<T>& v2, bool count_unfinished, const mxx::comm& comm, EqualFunc equal) {
    /*
     * NOTE: buckets are indexed by the global index of the first element in
     *       the bucket with a ONE-BASED-INDEX (since bucket number `0` is
     *       reserved for out-of-bounds)
     */
    // assert inputs are of equal size
    assert(v1.size() == v2.size() && v1.size() > 0);
    size_t local_size = v1.size();

    // get my global starting index
    //size_t prefix = part.excl_prefix_size();
    size_t prefix = mxx::exscan(local_size, comm);

    /*
     * assign local zero or one, depending on whether the bucket is the same
     * as the previous one
     */
    std::pair<T, T> last_element = std::make_pair(v1.back(), v2.back());
    std::pair<T, T> prevRight = mxx::right_shift(last_element, comm);
    bool firstDiff = false;
    if (comm.rank() == 0) {
        firstDiff = true;
    } else if (!equal(prevRight, std::make_pair(v1[0], v2[0]))) {
        firstDiff = true;
    }

    // set local_B1 to `(prefix+i)` if previous entry is different:
    // i.e., mark start of buckets, otherwise: set to 0
    bool nextDiff = firstDiff;
    for (std::size_t i = 0; i+1 < v1.size(); ++i) {
        bool setOne = nextDiff;
        nextDiff = !equal(std::make_pair(v1[i], v2[i]), std::make_pair(v1[i+1], v2[i+1]));
        v1[i] = setOne ? prefix+i+1 : 0;
    }

    v1.back() = nextDiff ? prefix+(local_size-1)+1 : 0;

    // init result
    std::pair<size_t,size_t> result;
    if (count_unfinished) {
        // count the number of unfinished elements and buckets
        T prev_right = mxx::right_shift(v1.back(), comm);
        T local_unfinished_buckets = 0;
        T local_unfinished_els = 0;
        if (comm.rank() != 0) {
            local_unfinished_buckets = (prev_right > 0 && v1[0] == 0) ? 1 : 0;
            local_unfinished_els = local_unfinished_buckets;
        }
        for (size_t i = 1; i < v1.size(); ++i) {
            if(v1[i-1] > 0 && v1[i] == 0) {
                ++local_unfinished_buckets;
                ++local_unfinished_els;
            }
            if (v1[i] == 0) {
                ++local_unfinished_els;
            }
        }
        std::pair<size_t,size_t> local_result(local_unfinished_buckets, local_unfinished_els);
        result = mxx::allreduce(local_result, pair_sum<size_t,size_t>(), comm);
    }

    global_fill_where_zero(v1, comm);

    return result;
}

template <typename T>
std::pair<size_t,size_t> rebucket(std::vector<T>& v1, std::vector<T>& v2, bool count_unfinished, const mxx::comm& comm) {
    return rebucket(v1, v2, count_unfinished, comm, std::equal_to<std::pair<T,T>>());
}

template <typename T>
std::pair<size_t,size_t> rebucket_gsa(std::vector<T>& v1, std::vector<T>& v2, bool count_unfinished, const mxx::comm& comm) {
    return rebucket(v1, v2, count_unfinished, comm, [](const std::pair<T,T>& left, const std::pair<T,T>& right){
        return left == right && left.second != 0;
    });
}

template <typename T>
std::pair<size_t,size_t> rebucket_gsa_kmers(std::vector<T>& v1, std::vector<T>& v2, bool count_unfinished, const mxx::comm& comm, unsigned int l) {
    T mask = (static_cast<T>(1) << l) - 1; // last character in k-mer
    return rebucket(v1, v2, count_unfinished, comm, [mask](const std::pair<T,T>& left, const std::pair<T,T>& right){
        return left == right && (mask & left.second) != 0;
    });
}

// func = void (const T prev, T& cur, size_t index)
template <typename Iterator, typename Func>
void foreach_pair(Iterator begin, Iterator end, Func func, const mxx::comm& comm) {
    typedef typename std::iterator_traits<Iterator>::value_type T;

    size_t n = std::distance(begin, end);
    T prev = mxx::right_shift(*(begin+(n-1)), comm);

    Iterator it = begin;
    T cur = *it;

    if (comm.rank() > 0) {
        func(prev, *it, 0);
    }
    prev = cur;

    for (size_t i = 0; i+1 < n; ++i) {
        prev = cur;
        ++it;
        cur = *it;
        func(prev, *it, i+1);
    }
}

// assumed sorted order (globally) by tuple (B1[i], B2[i])
// this reassigns new, unique bucket numbers in {1,...,n} globally
template <size_t L, typename T>
std::pair<size_t,size_t> rebucket_arr(std::vector<std::array<T, L+1> >& tuples, std::vector<T>& local_B, bool count_unfinished, const mxx::comm& comm) {
    /*
     * NOTE: buckets are indexed by the global index of the first element in
     *       the bucket with a ONE-BASED-INDEX (since bucket number `0` is
     *       reserved for out-of-bounds)
     */

    // init result
    std::pair<size_t,size_t> result;
    // get my global starting index
    size_t local_size = tuples.size();
    //size_t prefix = part.excl_prefix_size();
    size_t prefix = mxx::exscan(local_size, comm);
    size_t local_max = 0;

    foreach_pair(tuples.begin(), tuples.end(), [&](const std::array<T, L+1>& prev, std::array<T, L+1>& cur, size_t i) {
        if (!std::equal(&prev[1], &prev[1]+L, &cur[1])) {
            local_max = prefix + i + 1;
            local_B[i] = local_max;
        } else {
            local_B[i] = 0;
        }
    }, comm);

    // specially handle first element of first process
    if (comm.rank() == 0) {
        local_B[0] = 1;
        if (local_max == 0)
            local_max = 1;
    }


    if (count_unfinished) {
        // count the number of unfinished elements and buckets
        T prev_right = mxx::right_shift(local_B.back(), comm);
        T local_unfinished_buckets = 0;
        T local_unfinished_els = 0;
        if (comm.rank() != 0) {
            local_unfinished_buckets = (prev_right > 0 && local_B[0] == 0) ? 1 : 0;
            local_unfinished_els = local_unfinished_buckets;
        }
        for (size_t i = 1; i < local_B.size(); ++i) {
            if(local_B[i-1] > 0 && local_B[i] == 0) {
                ++local_unfinished_buckets;
                ++local_unfinished_els;
            }
            if (local_B[i] == 0) {
                ++local_unfinished_els;
            }
        }
        std::pair<size_t,size_t> local_result(local_unfinished_buckets, local_unfinished_els);
        result = mxx::allreduce(local_result, pair_sum<size_t,size_t>(), comm);
    }

    /*
     * Global prefix MAX:
     *  - such that for every item we have it's bucket number, where the
     *    bucket number is equal to the first index in the bucket
     *    this way buckets who are finished, will never receive a new
     *    number.
     */

    // 2.) distributed scan with max() to get starting max for each sequence
    size_t pre_max = mxx::exscan(local_max, mxx::max<size_t>(), comm);
    if (comm.rank() == 0)
        pre_max = 0;

    // 3.) linear scan and assign bucket numbers
    for (size_t i = 0; i < local_B.size(); ++i) {
        if (local_B[i] == 0)
            local_B[i] = pre_max;
        else
            pre_max = local_B[i];
        assert(local_B[i] <= i+prefix+1);
        // first element of bucket has id of it's own global index:
        assert(i == 0 || (local_B[i-1] ==  local_B[i] || local_B[i] == i+prefix+1));
    }

    return result;
}



#endif // BUCKETING_HPP
