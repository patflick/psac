/**
 * @file    parallel_utils.hpp
 * @ingroup group
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements utility functions needed when implementing parallel
 *          algorithms.
 *
 * Copyright (c) TODO
 *
 * TODO add Licence
 */
#ifndef HPC_PARALLEL_UTILS_H
#define HPC_PARALLEL_UTILS_H


#include <vector>
#include <numeric>

#include "algos.hpp"


/**
 * @brief Returns a block partitioning of an input of size `n` among `p` processors.
 *
 * @param n The number of elements.
 * @param p The number of processors.
 *
 * @return A vector of the number of elements for each processor.
 */
std::vector<int> block_partition(int n, int p)
{
    // init result
    std::vector<int> partition(p);
    // get the number of elements per processor
    int local_size = n / p;
    // and the elements that are not evenly distributable
    int remaining = n % p;
    for (int i = 0; i < p; ++i) {
        if (i < remaining) {
            partition[i] = local_size + 1;
        } else {
            partition[i] = local_size;
        }
    }
    return partition;
}

/**
 * @brief Outputs the number of elements that need to be sent to each processor.
 *
 * Calculates the send count of number of elements to send to each processor,
 * given that this processor has control of the elements given by the range:
 * [offset, offset + number_to_send)
 *
 * @param out       The output sequence for the counts.
 * @param target_partition  The new allocation of processors (num els per
*                           processor).
 * @param offset            The number of elements that lie before this processor.
 * @param number_to_send    The number of elements to send.
 */
template <typename OutputIterator>
void get_send_counts(OutputIterator out, const std::vector<int>& target_partition, int offset, int number_to_send)
{
    for (int i = 0; i < (int) target_partition.size(); ++i)
    {
        int send_count;
        if (target_partition[i] <= offset)
        {
            offset -= target_partition[i];
            send_count = 0;
        }
        else
        {
            // get number of elements to send
            send_count = std::min(number_to_send, target_partition[i] - offset);

            // if there are more receiving processors left, the offset is now zero
            offset = 0;
        }

        // set send count
        (*out++) = send_count;
        // subtract the number of send elements from the number of elements
        // to be send
        number_to_send -= send_count;
    }
}


/**
 * @brief   Gets the `recv_counts` argument for the all2allv communication.
 *
 * This takes the prefix sum of the sizes of the partioned sequences and
 * the processor partioning and retuns the `recv_counts` argument for all2all.
 *
 * @param out               An output iterator to be filled with the recv counts.
 * @param prefix_counts     The prefix sum of length of partitions.
 * @param target_partition  The new processor allocation (num els per processor).
 * @param proc_offset       The offset of this processor in its new communicator.
 */
template <typename OutputIterator>
void get_recv_counts(OutputIterator out, const std::vector<int>& prefix_counts, const std::vector<int>& target_partition, int proc_offset)
{
    // FIXME: maybe do this more efficiently by calculating it directly
    int element_offset = std::accumulate(target_partition.begin(), target_partition.begin() + proc_offset, 0u);

    // get number of elements that need to be received
    int recv_total = target_partition[proc_offset];
    // walk through the prefix sum until the number of elements passed
    // are bigger than the elements that have to lie before this processor
    for (int i = 0; i < (int) prefix_counts.size(); ++i)
    {
        int recv_count;
        if (recv_total == 0 || prefix_counts[i] <= element_offset)
        {
            recv_count = 0;
        }
        else
        {
            // get number of elements to send
            recv_count = std::min(recv_total, prefix_counts[i] - element_offset);
            element_offset += recv_count;
        }

        // set send count
        (*out++) = recv_count;
        // subtract the number of send elements from the number of elements
        // to be send
        recv_total -= recv_count;
    }
}


/**
 * @brief   Returns the displacements vector needed by MPI_Alltoallv.
 *
 * @param counts    The `counts` array needed by MPI_Alltoallv
 *
 * @return The displacements vector needed by MPI_Alltoallv.
 */
std::vector<int> get_displacements(const std::vector<int>& counts)
{
    // copy and do an exclusive prefix sum
    std::vector<int> result = counts;
    excl_prefix_sum(result.begin(), result.end());
    return result;
}

#endif // HPC_PARALLEL_UTILS_H
