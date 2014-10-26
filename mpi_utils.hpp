/**
 * @file    mpi_utils.hpp
 * @author  Nagakishore Jammula <njammula3@mail.gatech.edu>
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements some helpful MPI utility function, mostly for
 *          interacting with MPI using std::vectors
 *
 * Copyright (c) TODO
 *
 * TODO add Licence
 */

#ifndef HPC_MPI_UTILS
#define HPC_MPI_UTILS

// MPI include
#include <mpi.h>

// for sleep()
#include <unistd.h>

// C++ includes
#include <vector>
#include <numeric>
#include <stdexcept>



// own includes
#include "parallel_utils.hpp"



template <typename T>
MPI_Datatype get_mpi_dt()
{
    throw std::runtime_error("Unsupported MPI datatype");
    // default to int
    return MPI_INT;
}

template <>
MPI_Datatype get_mpi_dt<char>()
{
    return MPI_CHAR;
}

template <>
MPI_Datatype get_mpi_dt<unsigned char>()
{
    return MPI_UNSIGNED_CHAR;
}

template <>
MPI_Datatype get_mpi_dt<signed char>()
{
    return MPI_SIGNED_CHAR;
}

template <>
MPI_Datatype get_mpi_dt<unsigned short>()
{
    return MPI_UNSIGNED_SHORT;
}

template <>
MPI_Datatype get_mpi_dt<signed short>()
{
    return MPI_SHORT;
}

template <>
MPI_Datatype get_mpi_dt<int>()
{
    return MPI_INT;
}

template <>
MPI_Datatype get_mpi_dt<unsigned int>()
{
    return MPI_UNSIGNED;
}

template <>
MPI_Datatype get_mpi_dt<unsigned long>()
{
    return MPI_UNSIGNED_LONG;
}

template <>
MPI_Datatype get_mpi_dt<long>()
{
    return MPI_LONG;
}

template <>
MPI_Datatype get_mpi_dt<unsigned long long>()
{
    return MPI_UNSIGNED_LONG_LONG;
}

template <>
MPI_Datatype get_mpi_dt<long long>()
{
    return MPI_LONG_LONG;
}

template <>
MPI_Datatype get_mpi_dt<float>()
{
    return MPI_FLOAT;
}

template <>
MPI_Datatype get_mpi_dt<double>()
{
    return MPI_DOUBLE;
}

template <>
MPI_Datatype get_mpi_dt<long double>()
{
    return MPI_LONG_DOUBLE;
}

template <typename Iterator>
std::vector<typename std::iterator_traits<Iterator>::value_type> gather_range(Iterator begin, Iterator end, MPI_Comm comm)
{
    //static_assert(std::is_same<T, typename std::iterator_traits<Iterator>::value_type>::value, "Return type must of of same type as iterator value type");
    typedef typename std::iterator_traits<Iterator>::value_type T;
    // get MPI parameters
    int rank;
    int p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);

    // get local size
    int local_size = std::distance(begin, end);

    // init result
    std::vector<T> result;

    MPI_Datatype mpi_dt = get_mpi_dt<T>();

    // master process: receive results
    if (rank == 0)
    {
        // gather local array sizes, sizes are restricted to `int` by MPI anyway
        // therefore use int
        std::vector<int> local_sizes(p);
        MPI_Gather(&local_size, 1, MPI_INT,
                   &local_sizes[0], 1, MPI_INT,
                   0, MPI_COMM_WORLD);

        // gather-v to collect all the elements
        int total_size = std::accumulate(local_sizes.begin(), local_sizes.end(), 0);
        result.resize(total_size);
        std::vector<int> recv_displs = get_displacements(local_sizes);

        // gather v the vector data to the root
        MPI_Gatherv(&(*begin), local_size, mpi_dt,
                    &result[0], &local_sizes[0], &recv_displs[0], mpi_dt,
                    0, MPI_COMM_WORLD);
    }
    // else: send results
    else
    {
        // gather local array sizes
        MPI_Gather(&local_size, 1, MPI_INT, NULL, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // sent the actual data
        MPI_Gatherv(&(*begin), local_size, mpi_dt,
                    NULL, NULL, NULL, mpi_dt,
                    0, MPI_COMM_WORLD);
    }

    return result;
}


/**
 * @brief   Gathers local std::vectors to the master processor inside the
 *          given communicator.
 *
 * @param local_vec The local vectors to be gathered.
 * @param comm      The communicator.
 *
 * @return (On the master processor): The vector containing the concatenation
 *                                    of all distributed vectors.
 *         (On the slave processors): An empty vector.
 */
template<typename T>
std::vector<T> gather_vectors(std::vector<T>& local_vec, MPI_Comm comm)
{
    return gather_range(local_vec.begin(), local_vec.end(), comm);
}

template <typename InputIterator, typename OutputIterator>
void copy_n(InputIterator& in, std::size_t n, OutputIterator out)
{
    for (std::size_t i = 0u; i < n; ++i)
        *(out++) = *(in++);
}


template <typename Iterator>
std::vector<typename std::iterator_traits<Iterator>::value_type>
scatter_stream_block_decomp(Iterator input, uint32_t n, MPI_Comm comm)
{
    // get MPI Communicator properties
    int rank, p;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    typedef typename std::iterator_traits<Iterator>::value_type val_t;

    // the local vector size (MPI restricts message sizes to `int`)
    int local_size;

    // get the MPI data type
    MPI_Datatype mpi_dt = get_mpi_dt<val_t>();

    // init result
    std::vector<val_t> local_elements;

    if (rank == 0)
    {
        /* I am the root process */

        //Read from the input filename
        std::vector<int> block_decomp = block_partition(n, p);

        // scatter the sizes to expect
        MPI_Scatter(&block_decomp[0], 1, MPI_INT, &local_size,
                    1, MPI_INT, 0, comm);

        // copy the first block into the masters memory
        local_elements.resize(local_size);
        copy_n(input, local_size, local_elements.begin());

        // distribute the rest
        std::vector<val_t> local_buffer(block_decomp[0]);
        for (int i = 1; i < p; ++i) {
            // copy into local buffer
            copy_n(input, block_decomp[i], local_buffer.begin());
            // send the data to processor i
            MPI_Send (&local_buffer[0], block_decomp[i], mpi_dt,
                      i, i, comm);
        }
    }
    else
    {
        /* I am NOT the root process */
        std::runtime_error("slave called master function");
    }

    // return the local vectors
    return local_elements;
}

template <typename T>
std::vector<T> scatter_stream_block_decomp_slave(MPI_Comm comm)
{
    // get MPI Communicator properties
    int rank, p;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    // the local vector size (MPI restricts message sizes to `int`)
    int local_size;

    // get the MPI data type
    MPI_Datatype mpi_dt = get_mpi_dt<T>();

    // init result
    std::vector<T> local_elements;

    if (rank == 0)
    {
        std::runtime_error("master called slave function");
    }
    else
    {
        /* I am NOT the root process */

        // receive my new local data size
        MPI_Scatter(NULL, 1, MPI_INT,
                    &local_size, 1, MPI_INT,
                    0, comm);
        // resize local data
        local_elements.resize(local_size);
        // actually receive the data
        MPI_Status recv_status;
        MPI_Recv (&local_elements[0], local_size, mpi_dt,
                  0, rank, comm, &recv_status);
    }

    // return the local vectors
    return local_elements;
}


template <typename T>
std::vector<T> scatter_vector_block_decomp(std::vector<T>& global_vec, MPI_Comm comm)
{
    // get MPI Communicator properties
    int rank, p;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);


    // the local vector size (MPI restricts message sizes to `int`)
    int local_size;

    // get the MPI data type
    MPI_Datatype mpi_dt = get_mpi_dt<T>();

    // init result
    std::vector<T> local_elements;

    if (rank == 0)
    {
        /* I am the root process */

        // get size of global array
        uint32_t n = global_vec.size();

        //Read from the input filename
        std::vector<int> block_decomp = block_partition(n, p);

        // scatter the sizes to expect
        MPI_Scatter(&block_decomp[0], 1, MPI_INT, &local_size, 1, MPI_INT, 0, comm);

        // scatter-v the actual data
        // or stream
        //
        local_elements.resize(local_size);
        std::vector<int> displs = get_displacements(block_decomp);
        MPI_Scatterv(&global_vec[0], &block_decomp[0], &displs[0],
                     mpi_dt, &local_elements[0], local_size, mpi_dt, 0, comm);
    }
    else
    {
        /* I am NOT the root process */

        // receive the size of my local array
        MPI_Scatter(NULL, 1, MPI_INT,
                    &local_size, 1, MPI_INT,
                    0, comm);

        // resize result buffer
        local_elements.resize(local_size);
        // actually receive all the data
        MPI_Scatterv(NULL, NULL, NULL,
                     mpi_dt, &local_elements[0], local_size, mpi_dt, 0, comm);
    }

    // return local array
    return local_elements;
}

template<typename T>
void striped_excl_prefix_sum(std::vector<T>& x, MPI_Comm comm)
{
    MPI_Datatype mpi_dt = get_mpi_dt<T>();

    // get sum of all buckets and the prefix sum of that
    std::vector<T> all_sum(x.size());
    MPI_Allreduce(&x[0], &all_sum[0], x.size(), mpi_dt, MPI_SUM, comm);
    excl_prefix_sum(all_sum.begin(), all_sum.end());

    // exclusive prefix scan of vectors gives the number of elements prior
    // this processor in the _same_ bucket
    MPI_Exscan(&x[0], &x[0], x.size(), mpi_dt, MPI_SUM, comm);

    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0)
    {
        // set x to all_sum
        for (std::size_t i = 0; i < x.size(); ++i)
        {
            x[i] = all_sum[i];
        }
    }
    else
    {
        // sum these two vectors for all_sum
        for (std::size_t i = 0; i < x.size(); ++i)
        {
            x[i] += all_sum[i];
        }
    }
}

template<typename Iterator>
void global_prefix_sum(Iterator begin, Iterator end, MPI_Comm comm)
{
  // get types
  typedef typename std::iterator_traits<Iterator>::value_type T;
  MPI_Datatype mpi_dt = get_mpi_dt<T>();

  // local sum
  T sum = std::accumulate(begin, end, static_cast<T>(0));

  // exclusive prefix scan of local sums
  MPI_Exscan(&sum, &sum, 1, mpi_dt, MPI_SUM, comm);
  // first element in MPI_Exscan is undefined, therefore set to zero
  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == 0)
    sum = 0;

  // calculate the inclusive prefix sum of local elements by starting with
  // the global prefix sum value
  while (begin != end)
  {
    sum += *begin;
    *begin = sum;
    ++begin;
  }
}

// assumes only local send_counts for an all2allv operation are available
// this function scatters the send counts to get the receive counts
inline std::vector<int> all2allv_get_recv_counts(std::vector<int>& send_counts, MPI_Comm comm)
{
    std::size_t size = send_counts.size();
    std::vector<int> recv_counts;
    recv_counts.resize(size);
    MPI_Alltoall(&send_counts[0], 1, MPI_INT, &recv_counts[0], 1, MPI_INT, comm);
    return recv_counts;
}


void wait_gdb_attach(int wait_rank, MPI_Comm comm)
{
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);
    if (rank == wait_rank){
      std::cerr << "Rank " << rank << " is waiting in process " << getpid() << std::endl;
      int wait = 1;
      while (wait)
      {
        sleep(1);
      }
    }
    MPI_Barrier(comm);
}

#endif // HPC_MPI_UTILS
