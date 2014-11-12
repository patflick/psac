/**
 * @file    mpi_samplesort.hpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements sample sort for distributed MPI clusters
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 *
 * TODO add Licence
 */

#ifndef MPI_SAMPLESORT_HPP
#define MPI_SAMPLESORT_HPP

#include <mpi.h>

#include <assert.h>

#include <iterator>
#include <algorithm>
#include <vector>
#include <limits>

// for multiway-merge
// TODO: impelement own in case it is not GNU C++
#include <parallel/multiway_merge.h>
#include <parallel/merge.h>

#include "mpi_utils.hpp"


#include "timer.hpp"

#define SS_ENABLE_TIMER 1
#if SS_ENABLE_TIMER
#define SS_TIMER_START() TIMER_START()
#define SS_TIMER_END_SECTION(str) TIMER_END_SECTION(str)
#else
#define SS_TIMER_START()
#define SS_TIMER_END_SECTION(str)
#endif

// TODO put this function elsewhere [io utils!?]
template<typename _Iterator>
void print_range(_Iterator begin, _Iterator end)
{
    while (begin != end)
        std::cerr << *(begin++) << " ";
    std::cerr << std::endl;
}

/**
 * @brief Fixes an unequal distribution into a block decomposition
 */
template<typename _InIterator, typename _OutIterator>
void redo_block_decomposition(_InIterator begin, _InIterator end, _OutIterator out, MPI_Comm comm = MPI_COMM_WORLD)
{
    // get types from iterators
    typedef typename std::iterator_traits<_InIterator>::value_type value_type;

    // get communicator properties
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    // get MPI Datatype for the underlying type
    MPI_Datatype mpi_dt = get_mpi_dt<value_type>();

    // get local size
    std::size_t local_size = std::distance(begin, end);

    // get prefix sum of size and total size
    std::size_t prefix;
    std::size_t total_size;
    MPI_Datatype mpi_size_t = get_mpi_dt<std::size_t>();
    MPI_Allreduce(&local_size, &total_size, 1, mpi_size_t, MPI_SUM, comm);
    MPI_Exscan(&local_size, &prefix, 1, mpi_size_t, MPI_SUM, comm);
    if (rank == 0)
        prefix = 0;

    // calculate where to send elements
    std::vector<int> send_counts(p, 0);
    int first_p = block_partition_target_processor(total_size, p, prefix);
    std::size_t left_to_send = local_size;
    for (; left_to_send > 0 && first_p < p; ++first_p)
    {
        std::size_t nsend = std::min<std::size_t>(block_partition_prefix_size(total_size, p, first_p) - prefix, left_to_send);
        assert(nsend < std::numeric_limits<int>::max());
        send_counts[first_p] = nsend;
        left_to_send -= nsend;
        prefix += nsend;
    }

    // prepare all2all
    std::vector<int> recv_counts = all2allv_get_recv_counts(send_counts, comm);
    std::vector<int> send_displs = get_displacements(send_counts);
    std::vector<int> recv_displs = get_displacements(recv_counts);

    // execute all2all into output iterator
    MPI_Alltoallv(&(*begin), &send_counts[0], &send_displs[0], mpi_dt,
                  &(*out),   &recv_counts[0], &recv_displs[0], mpi_dt,
                  comm);
}

/**
 * @brief Redistributes elements from the given decomposition across processors
 *        into the decomposition given by the requested local_size
 */
template<typename _InIterator, typename _OutIterator>
void redo_arbit_decomposition(_InIterator begin, _InIterator end, _OutIterator out, std::size_t new_local_size, MPI_Comm comm = MPI_COMM_WORLD)
{
    // get types from iterators
    typedef typename std::iterator_traits<_InIterator>::value_type value_type;

    // get communicator properties
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    // get MPI Datatype for the underlying type
    MPI_Datatype mpi_dt = get_mpi_dt<value_type>();

    // get local size
    std::size_t local_size = std::distance(begin, end);

    // get prefix sum of size and total size
    std::size_t prefix;
    std::size_t total_size;
    MPI_Datatype mpi_size_t = get_mpi_dt<std::size_t>();
    MPI_Allreduce(&local_size, &total_size, 1, mpi_size_t, MPI_SUM, comm);
    MPI_Exscan(&local_size, &prefix, 1, mpi_size_t, MPI_SUM, comm);
    if (rank == 0)
        prefix = 0;

    // get the new local sizes from all processors
    std::vector<std::size_t> new_local_sizes(p);
    // this all-gather is what makes the arbitrary decomposition worse
    // in terms of complexity than when assuming a block decomposition
    MPI_Allgather(&new_local_size, 1, mpi_size_t, &new_local_sizes[0], 1, mpi_size_t, comm);
#ifndef NDEBUG
    std::size_t new_total_size = std::accumulate(new_local_sizes.begin(), new_local_sizes.end(), 0);
    assert(total_size == new_total_size);
#endif

    // calculate where to send elements
    std::vector<int> send_counts(p, 0);
    int first_p;
    std::size_t new_prefix = 0;
    for (first_p = 0; first_p < p-1; ++first_p)
    {
        // find processor for which the prefix sum exceeds mine
        // i have to send to the previous
        if (new_prefix + new_local_sizes[first_p] > prefix)
            break;
        new_prefix += new_local_sizes[first_p];
    }

    //= block_partition_target_processor(total_size, p, prefix);
    std::size_t left_to_send = local_size;
    for (; left_to_send > 0 && first_p < p; ++first_p)
    {
        // make the `new` prefix inclusive (is an exlcusive prefix prior)
        new_prefix += new_local_sizes[first_p];
        // send as many elements to the current processor as it needs to fill
        // up, but at most as many as I have left
        std::size_t nsend = std::min<std::size_t>(new_prefix - prefix, left_to_send);
        assert(nsend < std::numeric_limits<int>::max());
        send_counts[first_p] = nsend;
        // update the number of elements i have left (`left_to_send`) and
        // at which global index they start `prefix`
        left_to_send -= nsend;
        prefix += nsend;
    }

    // prepare all2all
    std::vector<int> recv_counts = all2allv_get_recv_counts(send_counts, comm);
    std::vector<int> send_displs = get_displacements(send_counts);
    std::vector<int> recv_displs = get_displacements(recv_counts);

    // execute all2all into output iterator
    MPI_Alltoallv(&(*begin), &send_counts[0], &send_displs[0], mpi_dt,
                  &(*out),   &recv_counts[0], &recv_displs[0], mpi_dt,
                  comm);
}


template<typename _Iterator, typename _Compare>
bool is_sorted(_Iterator begin, _Iterator end, _Compare comp, MPI_Comm comm = MPI_COMM_WORLD)
{
    // get value type of underlying data
    typedef typename std::iterator_traits<_Iterator>::value_type value_type;

    // get communicator properties
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    if (p == 1)
        return std::is_sorted(begin, end, comp);

    // get MPI datatype
    MPI_Datatype mpi_dt = get_mpi_dt<value_type>();

    // check that it is locally sorted
    int sorted = std::is_sorted(begin, end, comp);

    // compare if last element on left processor is not bigger than first
    // element on mine
    MPI_Request req;
    value_type left_el;
    if (rank > 0)
    {
        // start async receive
        // TODO: use some tag
        MPI_Irecv(&left_el, 1, mpi_dt, rank-1, 0, comm, &req);
    }
    if (rank < p-1)
    {
        // send last element to right
        // TODO: use custom tag
        MPI_Send(&(*(end-1)), 1, mpi_dt, rank+1, 0, comm);
    }

    // check the received element
    if (rank > 0)
    {
        // wait for successful receive
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        // check if sorted
        sorted = sorted && !comp(*begin, left_el);
    }

    // get global minimum to determine if the whole sequence is sorted
    int all_sorted;
    MPI_Allreduce(&sorted, &all_sorted, 1, MPI_INT, MPI_MIN, comm);

    // return as boolean
    return (all_sorted > 0);
}

template <typename _Iterator, typename _Compare>
std::vector<typename std::iterator_traits<_Iterator>::value_type>
sample_arbit_decomp(_Iterator begin, _Iterator end, _Compare comp, int s, MPI_Comm comm, MPI_Datatype mpi_dt)
{
    typedef typename std::iterator_traits<_Iterator>::value_type value_type;
    std::size_t local_size = std::distance(begin, end);
    assert(local_size > 0);

    // get communicator properties
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    // get total size n
    std::size_t total_size;
    MPI_Datatype mpi_size_t = get_mpi_dt<std::size_t>();
    MPI_Allreduce(&local_size, &total_size, 1, mpi_size_t, MPI_SUM, comm);

    //  pick a total of s*p samples, thus locally pick ceil((local_size/n)*s*p)
    //  and at least one samples from each processor.
    //  this will result in at least s*p samples.
    std::size_t local_s = ((local_size*s*p)+total_size-1)/total_size;
    local_s = std::max<std::size_t>(local_s, 1);

    //. init samples
    std::vector<value_type> local_splitters(local_s);

    // pick local samples
    _Iterator pos = begin;
    for (std::size_t i = 0; i < local_splitters.size(); ++i)
    {
        std::size_t bucket_size = local_size / (local_s+1) + (i < (local_size % (local_s+1)) ? 1 : 0);
        // pick last element of each bucket
        pos += (bucket_size-1);
        local_splitters[i] = *pos;
        ++pos;
    }

    // 2. gather samples to `rank = 0`
    // - TODO: rather call sample sort
    //         recursively and implement a base case for samplesort which does
    //         gather to rank=0, local sort and redistribute
    std::vector<value_type> all_samples = gather_vectors(local_splitters, comm);

    // sort and pick p-1 samples on master
    if (rank == 0)
    {
        // 3. local sort on master
        std::sort(all_samples.begin(), all_samples.end(), comp);

        // 4. pick p-1 splitters and broadcast them
        if (local_splitters.size() != p-1)
        {
            local_splitters.resize(p-1);
        }
        // split into `p` pieces and choose the `p-1` splitting elements
        _Iterator pos = all_samples.begin();
        for (std::size_t i = 0; i < local_splitters.size(); ++i)
        {
            std::size_t bucket_size = (p*s) / p + (i < static_cast<std::size_t>((p*s) % p) ? 1 : 0);
            // pick last element of each bucket
            local_splitters[i] = *(pos + (bucket_size-1));
            pos += bucket_size;
        }
    }

    // size splitters for receiving
    if (local_splitters.size() != p-1)
    {
        local_splitters.resize(p-1);
    }

    // 4. broadcast and receive final splitters
    MPI_Bcast(&local_splitters[0], local_splitters.size(), mpi_dt, 0, comm);

    return local_splitters;
}


template <typename _Iterator, typename _Compare>
std::vector<typename std::iterator_traits<_Iterator>::value_type>
sample_block_decomp(_Iterator begin, _Iterator end, _Compare comp, int s, MPI_Comm comm, MPI_Datatype mpi_dt)
{
    typedef typename std::iterator_traits<_Iterator>::value_type value_type;
    std::size_t local_size = std::distance(begin, end);
    assert(local_size > 0);

    // get communicator properties
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);


    // 1. samples
    //  - pick `s` samples equally spaced such that `s` samples define `s+1`
    //    subsequences in the sorted order
    std::vector<value_type> local_splitters(s);
    _Iterator pos = begin;
    for (std::size_t i = 0; i < local_splitters.size(); ++i)
    {
        std::size_t bucket_size = local_size / (s+1) + (i < (local_size % (s+1)) ? 1 : 0);
        // pick last element of each bucket
        pos += (bucket_size-1);
        local_splitters[i] = *pos;
        ++pos;
    }

    // 2. gather samples to `rank = 0`
    // - TODO: rather call sample sort
    //         recursively and implement a base case for samplesort which does
    //         gather to rank=0, local sort and redistribute
    if (rank == 0)
    {
        std::vector<value_type> all_samples(p*s);
        MPI_Gather(&local_splitters[0], s, mpi_dt,
                   &all_samples[0], s, mpi_dt, 0, comm);

        // 3. local sort on master
        std::sort(all_samples.begin(), all_samples.end(), comp);

        // 4. pick p-1 splitters and broadcast them
        if (local_splitters.size() != p-1)
        {
            local_splitters.resize(p-1);
        }
        // split into `p` pieces and choose the `p-1` splitting elements
        _Iterator pos = all_samples.begin();
        for (std::size_t i = 0; i < local_splitters.size(); ++i)
        {
            std::size_t bucket_size = (p*s) / p + (i < static_cast<std::size_t>((p*s) % p) ? 1 : 0);
            // pick last element of each bucket
            local_splitters[i] = *(pos + (bucket_size-1));
            pos += bucket_size;
        }
    }
    else
    {
        // simply send
        MPI_Gather(&local_splitters[0], s, mpi_dt, NULL, 0, mpi_dt, 0, comm);

        // resize splitters for receiving
        if (local_splitters.size() != p-1)
        {
            local_splitters.resize(p-1);
        }
    }

    // 4. broadcast and receive final splitters
    MPI_Bcast(&local_splitters[0], local_splitters.size(), mpi_dt, 0, comm);

    return local_splitters;
}


template<typename _Iterator, typename _Compare, bool _Stable = false, bool _AssumeBlockDecomp = true>
void samplesort(_Iterator begin, _Iterator end, _Compare comp, MPI_Comm comm = MPI_COMM_WORLD)
{
    // get value type of underlying data
    typedef typename std::iterator_traits<_Iterator>::value_type value_type;


    // get communicator properties
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    SS_TIMER_START();

    // perform local (stable) sorting
    if (_Stable)
        std::stable_sort(begin, end, comp);
    else
        std::sort(begin, end, comp);

    if (p == 1)
        return;

    SS_TIMER_END_SECTION("local_sort");

    /*
    std::cerr << "on rank = " << rank << std::endl;
    std::cerr << "IN="; print_range(begin, end);
    */

    // get MPI datatype
    MPI_Datatype mpi_dt = get_mpi_dt<value_type>();
    MPI_Type_commit(&mpi_dt);

    // number of samples
    int s = p-1;

    std::size_t local_size = std::distance(begin, end);
    assert(local_size > 0);

    // sample sort
    // 1. pick `s` samples on each processor
    // 2. gather to `rank=0`
    // 3. local sort on master
    // 4. broadcast the p-1 final splitters
    // 5. locally find splitter positions in data
    //    (if an identical splitter appears twice, then split evenly)
    //    => send_counts
    // 6. distribute send_counts with all2all to get recv_counts
    // 7. allocate enough space (may be more than previously allocated) for receiving
    // 8. all2all
    // 9. local reordering
    // A. equalizing distribution into original size (e.g.,block decomposition)
    //    by elements to neighbors

    // get splitters, using the method depending on whether the input consists
    // of arbitrary decompositions or not
    std::vector<value_type> local_splitters;
    if(_AssumeBlockDecomp)
        local_splitters = sample_block_decomp(begin, end, comp, s, comm, mpi_dt);
    else
        local_splitters = sample_arbit_decomp(begin, end, comp, s, comm, mpi_dt);
    SS_TIMER_END_SECTION("get_splitters");

    // 5. locally find splitter positions in data
    //    (if an identical splitter appears at least three times (or more),
    //    then split the intermediary buckets evenly) => send_counts
    std::vector<int> send_counts(p);
    _Iterator pos = begin;
    for (std::size_t i = 0; i < local_splitters.size();)
    {
        // the number of splitters which are equal starting from `i`
        unsigned int split_by = 1;
        if (i > 0 && !comp(local_splitters[i-1], local_splitters[i]))
        {
            while (i+split_by < local_splitters.size()
                   && !comp(local_splitters[i], local_splitters[i+split_by]))
            {
                ++split_by;
            }
        }

        // get bucket boundary and size
        _Iterator next = std::upper_bound(pos, end, local_splitters[i], comp);
        std::size_t bucket_size = std::distance(pos, next);

        // potentially split accross processors
        for (unsigned int j = 0; j < split_by; ++j)
        {
            // TODO: this kind of splitting is not `stable` -> fix it
            // -> send all elements to single processor, but change processor
            //    based on own rank
            std::size_t out_bucket_size = bucket_size / split_by;
            if (split_by > 1 && j < (bucket_size % split_by))
                ++out_bucket_size;
            assert(out_bucket_size < std::numeric_limits<int>::max());
            send_counts[i] = static_cast<int>(out_bucket_size);
            ++i;
        }
        pos = next;
    }

    SS_TIMER_END_SECTION("send_counts");

    // send last elements to last processor
    std::size_t out_bucket_size = std::distance(pos, end);
    assert(out_bucket_size < std::numeric_limits<int>::max());
    send_counts[p-1] = static_cast<int>(out_bucket_size);
    assert(std::accumulate(send_counts.begin(), send_counts.end(), 0) == local_size);


    // 6. distribute send_counts with all2all to get recv_counts
    std::vector<int> recv_counts = all2allv_get_recv_counts(send_counts, comm);

    // 7. allocate enough space (may be more than previously allocated) for receiving
    std::size_t recv_n = std::accumulate(recv_counts.begin(), recv_counts.end(), 0);
    assert(recv_n <= 2* local_size);

    //std::cerr << "Allocating for RECV: " << recv_n << std::endl;
    std::vector<value_type> recv_elements(recv_n);

    // 8. all2all
    std::vector<int> send_displs = get_displacements(send_counts);
    std::vector<int> recv_displs = get_displacements(recv_counts);
    SS_TIMER_END_SECTION("all2all_params");
    MPI_Alltoallv(&(*begin), &send_counts[0], &send_displs[0], mpi_dt,
                  &recv_elements[0], &recv_counts[0], &recv_displs[0], mpi_dt,
                  comm);
    SS_TIMER_END_SECTION("all2all");

    // 9. local reordering
    /*
    if (_Stable)
        std::stable_sort(recv_elements.begin(), recv_elements.end(), comp);
    else
        std::sort(recv_elements.begin(), recv_elements.end(), comp);
    */
    /* multiway-merge (using the implementation in __gnu_parallel) */
    // prepare the sequence offsets
    typedef typename std::vector<value_type>::iterator val_it;
    std::vector<std::pair<val_it, val_it> > seqs(p);
    for (int i = 0; i < p; ++i)
    {
        seqs[i].first = recv_elements.begin() + recv_displs[i];
        seqs[i].second = seqs[i].first + recv_counts[i];
    }
    val_it start_merge_it = recv_elements.begin();
    for (; recv_n > 0;)
    {
        std::size_t merge_n = local_size;
        if (recv_n < local_size)
            merge_n = recv_n;
        // i)   merge at most `local_size` many elements sequentially
        __gnu_parallel::sequential_tag seq_tag;
        __gnu_parallel::multiway_merge(seqs.begin(), seqs.end(), begin, merge_n, comp, seq_tag);

        // ii)  compact the remaining elements in `recv_elements`
        for (int i = p-1; i > 0; --i)
        {
            seqs[i-1].first = std::copy_backward(seqs[i-1].first, seqs[i-1].second, seqs[i].first);
            seqs[i-1].second = seqs[i].first;
        }
        // iii) copy the output buffer `local_size` elements back into
        //      `recv_elements`
        start_merge_it = std::copy(begin, begin + merge_n, start_merge_it);
        assert(start_merge_it == seqs[0].first);

        // reduce the number of elements to be merged
        recv_n -= merge_n;
    }

    SS_TIMER_END_SECTION("local_merge");

    /*
    std::cerr << "on rank = " << rank << std::endl;
    std::cerr << "RV="; print_range(recv_elements.begin(), recv_elements.end());
    */

    // A. equalizing distribution into original size (e.g.,block decomposition)
    //    by elements to neighbors
    //    and save elements into the original iterator positions
    if (_AssumeBlockDecomp)
        redo_block_decomposition(recv_elements.begin(), recv_elements.end(), begin, comm);
    else
        redo_arbit_decomposition(recv_elements.begin(), recv_elements.end(), begin, local_size, comm);

    SS_TIMER_END_SECTION("fix_partition");
}


#endif // MPI_SAMPLESORT_HPP


