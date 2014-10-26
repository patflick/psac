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
    /*
    static_assert(std::is_same<value_type,
                      typename std::iterator_traits<_OutIterator>::value_type
                  >::value,
                  "Input and Output Iterators must be of same value_type");
    */

    // get communicator properties
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    // get MPI Datatype for the underlying type
    MPI_Datatype mpi_dt = get_mpi_dt<value_type>();

    // get local size
    std::size_t local_size = std::distance(begin, end);

    // get prefix and total size
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

template<typename _Iterator, typename _Compare, bool _Stable = false>
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

        SS_TIMER_END_SECTION("gather_samples");

        // 3. local sort on master
        std::stable_sort(all_samples.begin(), all_samples.end(), comp);

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

    SS_TIMER_END_SECTION("bcast_splitters");

    //std::cerr << "SP="; print_range(local_splitters.begin(), local_splitters.end());

    // 5. locally find splitter positions in data
    //    (if an identical splitter appears twice (or more), then split evenly)
    //    => send_counts
    std::vector<int> send_counts(p);
    pos = begin;
    for (std::size_t i = 0; i < local_splitters.size();)
    {
        // the number of splitters which are equal starting from `i`
        unsigned int split_by = 1;
        while (i+split_by < local_splitters.size() && !comp(local_splitters[i], local_splitters[i+split_by]))
        {
            ++split_by;
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

    //std::cerr << "SC="; print_range(send_counts.begin(), send_counts.end());


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
    //    TODO: multisequence merge instead of resorting
    if (_Stable)
        std::stable_sort(recv_elements.begin(), recv_elements.end(), comp);
    else
        std::sort(recv_elements.begin(), recv_elements.end(), comp);

    SS_TIMER_END_SECTION("local_merge");

    /*
    std::cerr << "on rank = " << rank << std::endl;
    std::cerr << "RV="; print_range(recv_elements.begin(), recv_elements.end());
    */

    // A. equalizing distribution into original size (e.g.,block decomposition)
    //    by elements to neighbors
    //    and save elements into the original iterator positions
    redo_block_decomposition(recv_elements.begin(), recv_elements.end(), begin, comm);

    SS_TIMER_END_SECTION("fix_partition");
    MPI_Type_free(&mpi_dt);
}


#endif // MPI_SAMPLESORT_HPP


