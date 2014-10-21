




/*
 * Idea:
 * - bucket by string index ( ~BISA?)
 *      o initially by having bucket-number = k-mer
 *      o 1-mer would be BISA[i] = char2int(S[i])
 *      // o initially by sorting tuples (string-index, k-mer number) [32-64 bits each]
 *
 * - BISA: 2^k -> 2^{k+1}
 *      o Read 2^k ahead (comm with one processor) to get next bucket no (BISA[i+2^k])
 *      o create tuple (BISA[i], BISA[i+2^k], i)
 *      o sort all tuples by first two entries to get next SA
 *      o prefix-scan to assign new bucket numbers (bn, i)
 *      o bucket-"sort" by `i` to get BISA_{k+1} (one all2all)
 *
 * - SA based
 *      o (assume we have suffix array after sorting first k characters [k-mers])
 *          - req: bucket sort (histogram and shuffle)
 *      o create tuples for each SA[i]:  (j=SA[i], i, b)
 *      o send to processor for j -> current ISA
 *      o copy and shift by 2^k
 *      o gives new combined bucket number tuple, prefix scan for new bucket numbers
 *      o inverse (j,i) by sending to processor for i: gives sorting by first
 *        bucket index
 *      o sort by second bucket number (which is potentially only a local sort!)
 *      o ENH: adjust bucket boundaries if close, so that it becomes local sort
 *
 *
 * - LCP while M&M:
 *      o TODO: this doesn't yet make sense and/or work properly :-/
 *      o TODO: see ya tmrw dude :)
 *      o when combining two buckets into tuple, we add the other buckets (+2^k)
 *        current LCP value to the tuple, after sorting into SA order, we can
 *        determine the new LCP value by adding of still in same bucket
 *        with neighbor
 *      o when B[i]
 *
 *
 *      ---------------------------
 * SA:  | | | ... |j|   ...   | | |
 *      ---------------------------
 *                 ^
 *                 i
 *
 *      ---------------------------
 * ISA: | |  ...      |i| ... | | |
 *      ---------------------------
 *                     ^
 *                     j
 *
 */

#include <mpi.h>

#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>

#include <assert.h>

#include "algos.hpp"
#include "parallel_utils.hpp"
#include "mpi_utils.hpp"
#include "mpi_samplesort.hpp"

#define PSAC_TAG_EDGE_KMER 1
#define PSAC_TAG_SHIFT 2

// must be 64 bit int for strings > 4GB
// those need special handleing with MPI anyway, since MPI only works with ints
typedef int index_t; // content of SA and ISA and Bucket-numbers
typedef index_t count_t; // bucket counts for histogram


template<typename T>
void print_vec(const std::vector<T>& vec)
{
    auto it = vec.begin();
    while (it != vec.end())
    {
        std::cerr << *(it++) << " ";
    }
    std::cerr << std::endl;
}

template<typename Iterator, typename T = typename std::iterator_traits<Iterator>::value_type>
T global_max_element(Iterator begin, Iterator end, MPI_Comm comm)
{
    // mpi datatype
    MPI_Datatype mpi_dt = get_mpi_dt<T>();
    // get local max
    T max = *std::max_element(begin, end);

    // get global max
    T gl_max;
    MPI_Allreduce(&max, &gl_max, 1, mpi_dt, MPI_MAX, comm);

    return gl_max;
}

template<typename T, typename Iterator>
std::vector<T> get_histogram(Iterator begin, Iterator end, std::size_t size = 0)
{
    if (size == 0)
        size = static_cast<std::size_t>(*std::max_element(begin, end)) + 1;
    std::vector<T> hist(size);

    while (begin != end)
    {
        ++hist[static_cast<std::size_t>(*(begin++))];
    }

    return hist;
}

// TODO: global character histogram + required bits per character + compression
std::vector<index_t> alphabet_histogram(const std::string& local_str, MPI_Comm comm)
{
    // get local histogram of alphabet characters
    std::vector<index_t> hist = get_histogram<index_t>(local_str.begin(), local_str.end(), 256);

    std::vector<index_t> out_hist(256);

    // global all reduce to get global histogram
    MPI_Datatype mpi_dt = get_mpi_dt<index_t>();
    MPI_Allreduce(&hist[0], &out_hist[0], 256, mpi_dt, MPI_SUM, comm);

    return out_hist;
}

std::vector<char> alphabet_mapping_tbl(const std::vector<index_t>& global_hist)
{
    std::vector<char> mapping(256, 0);

    char next = static_cast<char>(1);
    for (std::size_t c = 0; c < 256; ++c)
    {
        if (global_hist[c] != 0)
        {
            mapping[c] = next;
            ++next;
        }
    }
    return mapping;
}

unsigned int alphabet_unique_chars(const std::vector<index_t>& global_hist)
{
    unsigned int unique_count = 0;
    for (std::size_t c = 0; c < 256; ++c)
    {
        if (global_hist[c] != 0)
        {
            ++unique_count;
        }
    }
    return unique_count;
}

unsigned int ceillog2(unsigned int x)
{
    unsigned int log_floor = 0;
    unsigned int n = x;
    for (;n != 0; n >>= 1)
    {
        ++log_floor;
    }
    --log_floor;
    // add one if not power of 2
    return log_floor + (((x&(x-1)) != 0) ? 1 : 0);
}

unsigned int alphabet_bits_per_char(unsigned int sigma)
{
    // since we have to account for the `0` character, we use ceil(log(unique_chars + 1))
    return ceillog2(sigma+1);
}

template<typename word_t>
unsigned int alphabet_chars_per_word(unsigned int bits_per_char)
{
    // using bit concatenation, NOT multiplication by base
    // TODO: try multiplication by base
    return (sizeof(word_t)*8)/bits_per_char;
}

void initial_bucketing(const std::basic_string<char>& local_str, std::vector<index_t>& local_B, MPI_Comm comm)
{
    // get local size
    std::size_t local_size = local_str.size();

    // get communication parameters
    int rank, p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);

    // get global alphabet histogram
    std::vector<index_t> alphabet_hist = alphabet_histogram(local_str, comm);
    // get mapping table and alphabet sizes
    std::vector<char> alphabet_mapping = alphabet_mapping_tbl(alphabet_hist);
    unsigned int sigma = alphabet_unique_chars(alphabet_hist);
    // bits per character: set l=ceil(log(sigma))
    unsigned int l = alphabet_bits_per_char(sigma);
    // number of characters per word => the `k` in `k-mer`
    unsigned int k = alphabet_chars_per_word<index_t>(l);

    std::cerr << "Detecting alphabet with " << sigma << " characters" << std::endl;
    std::cerr << "  choosing bit size = " << l << " and k=" << k << std::endl;
    std::cerr << "  mapping 'A'->" << (int)alphabet_mapping['A'] << std::endl;
    std::cerr << "  mapping 'C'->" << (int)alphabet_mapping['C'] << std::endl;
    std::cerr << "  mapping 'G'->" << (int)alphabet_mapping['G'] << std::endl;
    std::cerr << "  mapping 'T'->" << (int)alphabet_mapping['T'] << std::endl;

    // get k-mer mask
    index_t kmer_mask = ((static_cast<index_t>(1) << (l*k)) - static_cast<index_t>(1));

    // sliding window k-mer (for prototype only using ASCII alphabet)

    // fill first k-mer (until k-1 positions) and send to left processor
    // filling k-mer with first character = MSBs for lexicographical ordering
    auto str_it = local_str.begin();
    index_t kmer = 0;
    for (unsigned int i = 0; i < k-1; ++i)
    {
        kmer <<= l;
        kmer |= alphabet_mapping[static_cast<index_t>(*str_it)];
        ++str_it;
    }


    // send this to left processor, start async receive from right processor
    // for last in seq
    // start receiving for end
    index_t last_kmer = 0;
    MPI_Request recv_req;
    MPI_Datatype mpi_dt = get_mpi_dt<index_t>();
    if (rank < p-1) // if not last processor
    {
        MPI_Irecv(&last_kmer, 1, mpi_dt, rank+1, PSAC_TAG_EDGE_KMER,
                  comm, &recv_req);
    }
    if (rank > 0) // if not first processor
    {
        // TODO: [ENH] use async send as well and start with the computation
        //             immediately
        MPI_Send(&kmer, 1, mpi_dt, rank-1, PSAC_TAG_EDGE_KMER, comm);
    }


    // init output
    if (local_B.size() != local_size)
        local_B.resize(local_size);
    auto buk_it = local_B.begin();
    // continue to create all k-mers and add into histogram count
    while (str_it != local_str.end())
    {
        // get next kmer
        kmer <<= l;
        kmer |= alphabet_mapping[static_cast<index_t>(*str_it)];
        kmer &= kmer_mask;
        // add to bucket number array
        *buk_it = kmer;
        // increase iterators
        ++str_it;
        ++buk_it;
    }

    // finish the receive to get the last k-1 k-kmers with string data from the
    // processor to the right
    if (rank < p-1) // if not last processor
    {
        // wait for the async receive to finish
        MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
    }
    else
    {
        // in this case the last k-mers contains shifting `$` signs
        // we assume this to be the `\0` value
        // TODO: how can we solve this efficiently when using on 4 character
        // DNA strings? (using a 5 character alphabet might be unnessesary
        // overhead)
    }


    // construct last (k-1) k-mers
    for (unsigned int i = 0; i < k-1; ++i)
    {
        kmer <<= l;
        kmer |= (last_kmer >> (l*(k-i-2)));
        kmer &= kmer_mask;

        // add to bucket number array
        *buk_it = kmer;
        ++buk_it;
    }
}



#if 0

template<bool hasB2>
void large_buckets_b_to_sa(std::size_t n, std::vector<index_t>& local_B, std::vector<index_t>& local_B2, std::vector<index_t>& local_SA, MPI_Comm comm)
{
    // get communication parameters
    int rank, p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);

    // get size (check all arrays are of same size)
    std::size_t local_size = local_B.size();
    if (hasB2)
    {
        assert(local_size == local_B2.size());
    }
    assert(local_size == block_partition_local_size(n, p, rank));

    // create bucket hist
    // 1) get largest bucket no (globally)
    index_t max_bucket = global_max_element(local_B.begin(), local_B.end(), comm);
    std::vector<count_t> hist = get_histogram<count_t>(local_B.begin(), local_B.end(), static_cast<std::size_t>(max_bucket) + 1);


    // local_hist_prefix
    // TODO: [ENH] instead of copying, can revert prefix sum in additional O(n) pass
    //             this comes down to a mem vs. computation tradeoff, since both
    //             are only constant factors
    std::vector<count_t> local_hist_prefix(hist);
    excl_prefix_sum(local_hist_prefix.begin(), local_hist_prefix.end());

    // for local SA
    std::vector<index_t> send_SA(local_size);
    // for local buckets
    std::vector<index_t> send_B(local_size);
    std::vector<index_t> send_B2;
    if (hasB2)
    {
        send_B2.resize(local_size);
    }

    index_t str_off = block_partition_excl_prefix_size(n, p, rank);

    for (std::size_t i = 0; i < local_size; ++i)
    {
        index_t bucket_no = local_B[i];
        index_t out_pos = local_hist_prefix[bucket_no]++;

        // save buckets in sorted order together with their original
        // position in the string (~> incomplete SA)
        // TODO: this is not really cache efficient, could be done better
        send_B[out_pos] = bucket_no;
        send_SA[out_pos] = i + str_off;
        if (hasB2)
        {
            send_B2[out_pos] = local_B2[i];
        }
        // for each bucket, the elements are now sorted by (bucket-no,
        // str-index) where `str-index` is the SA number, since we linearly
        // scanned the input
    }


    // how many elements are there before the elements in a bucket when
    // considering the buckets across all processors:
    //  -> striped prefix sum of historams
    std::vector<count_t> hist_gl_prefix(hist.begin(), hist.end());
    striped_excl_prefix_sum(hist_gl_prefix, comm);

    // scan all buckets, get their global offset from the striped prefix sum
    // and with this determine where to send and how to split the bucket across
    // processors
    std::vector<int> send_counts(p, 0);
    int cur_p = 0;
    for(std::size_t i = 0; i < hist.size(); ++i)
    {
        count_t n_send = hist[i];
        while(n_send > 0)
        {
            // split bucket and switch to next processor
            // TODO: make sure everything works for sizes > MAX_INT
            count_t cur_prefix_partition_size = block_partition_prefix_size(n, p, cur_p);
            count_t send_to_gl_pos = hist_gl_prefix[i] + (hist[i] - n_send);
            // send to the current processor only if the global offset starts
            // within
            if (send_to_gl_pos < cur_prefix_partition_size)
            {
              // send at most the number of open slots on that current processor
              int send_c = std::min(n_send, cur_prefix_partition_size - send_to_gl_pos);
              send_counts[cur_p] += send_c;
              n_send -= send_c;
            }
            else
            {
              // proceed to next processor
              cur_p += 1;
            }
        }
    }



    // send local B and SA to the according processors via all2all
    std::vector<int> recv_counts = all2allv_get_recv_counts(send_counts, comm);
    std::vector<int> send_displs = get_displacements(send_counts);
    std::vector<int> recv_displs = get_displacements(recv_counts);
#ifndef NDEBUG
    std::cerr << "Send count on rank = " << rank << ": "; print_vec(send_counts);
    std::cerr << "Recv count on rank = " << rank << ": "; print_vec(recv_counts);
    std::cerr << "send displs on rank = " << rank << ": "; print_vec(send_displs);
    std::cerr << "Recv displs on rank = " << rank << ": "; print_vec(recv_displs);
#endif
    // allocate receive buffers (TODO [ENH]: receive in place and then do the
    // bucket sort below also in-place)
    //std::vector<index_t> recv_B(local_size);// = buckets; // TODO: assert that all those are actually equal to the local block partition size
    //std::vector<index_t> recv_SA(local_size);
    // allocate the local SA

    if (local_SA.size() != local_size)
        local_SA.resize(local_size);

    // TODO: correct MPI_Datatype
    MPI_Alltoallv(&send_B[0], &send_counts[0], &send_displs[0], MPI_INT,
                  &local_B[0], &recv_counts[0], &recv_displs[0], MPI_INT,
                  comm);
    MPI_Alltoallv(&send_SA[0], &send_counts[0], &send_displs[0], MPI_INT,
                  &local_SA[0], &recv_counts[0], &recv_displs[0], MPI_INT,
                  comm);
    if (hasB2)
    {
        MPI_Alltoallv(&send_B2[0], &send_counts[0], &send_displs[0], MPI_INT,
                      &local_B2[0], &recv_counts[0], &recv_displs[0], MPI_INT,
                      comm);
    }
#ifndef NDEBUG
    std::cerr << "================= After All2All ===============" << std::endl;
    std::cerr << "Send count on rank = " << rank << ": "; print_vec(send_counts);
    std::cerr << "Recv count on rank = " << rank << ": "; print_vec(recv_counts);
    std::cerr << "send displs on rank = " << rank << ": "; print_vec(send_displs);
    std::cerr << "Recv displs on rank = " << rank << ": "; print_vec(recv_displs);
    std::cerr << "send  B :"; print_vec(send_B);
    std::cerr << "send  SA:"; print_vec(send_SA);
    std::cerr << "local B :"; print_vec(local_B);
    std::cerr << "local SA:"; print_vec(local_SA);
#endif


    /* local rearranging */


    // two possible orders of iterating:
    //   1) linear through input and write "randomly
    //   2) linear for output/writes, read "randomly" (implementing this currently)
    // write sorted order (linear write, jump between recv_displacements in input)
    auto outit_B = send_B.begin();
    auto outit_SA = send_SA.begin();
    auto outit_B2 = send_B2.begin();

    for (std::size_t b = 0; b < hist.size(); ++b)
    {
        for (int i = 0; i < p; ++i)
        {
            int recv_pos = recv_displs[i];
            while ((i+1 == p || recv_pos != recv_displs[i+1]) && static_cast<std::size_t>(recv_pos) < local_size && static_cast<std::size_t>(local_B[recv_pos]) == b)
            {
                *(outit_B++) = local_B[recv_pos];
                *(outit_SA++) = local_SA[recv_pos];
                if (hasB2)
                {
                    *(outit_B2++) = local_B2[recv_pos];
                }
                ++recv_pos;
            }
            recv_displs[i] = recv_pos;
        }
    }

    // results are now in the `send` rather than the `local` arrays
    // -> swap content
    local_B.swap(send_B);
    local_SA.swap(send_SA);
    if (hasB2)
        local_B2.swap(send_B2);
}
#endif


// assumed sorted order (globally) by tuple (B1[i], B2[i])
// this reassigns new, unique bucket numbers in {1,...,n} globally
std::size_t rebucket(std::vector<index_t>& local_B1, std::vector<index_t>& local_B2, MPI_Comm comm, bool count_unfinished)
{
    // assert inputs are of equal size
    assert(local_B1.size() == local_B2.size() && local_B1.size() > 0);

    // init result
    std::size_t result = 0;

    // get communication parameters
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    /*
     * send right most element to one processor to the right
     * so that that processor can determine whether the same bucket continues
     * or a new bucket starts with it's first element
     */
    MPI_Request recv_req;
    MPI_Datatype mpi_dt = get_mpi_dt<index_t>();
    index_t prevRight[2];
    if (rank > 0) // if not last processor
    {
        // TODO: fix MPI datatype
        MPI_Irecv(prevRight, 2, mpi_dt, rank-1, PSAC_TAG_EDGE_KMER,
                  comm, &recv_req);
    }
    if (rank < p-1) // if not first processor
    {
        // send my most right element to the right
        index_t myRight[2] = {local_B1.back(), local_B2.back()};
        // TODO: [ENH] use async send as well and start with the computation
        //             immediately
        MPI_Send(myRight, 2, mpi_dt, rank+1, PSAC_TAG_EDGE_KMER, comm);
    }
    if (rank > 0)
    {
      // wait for the async receive to finish
      MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
    }
    else
    {
        prevRight[0] = local_B1[0];
        prevRight[1] = local_B2[0];
    }

    /*
     * assign local zero or one, depending on whether the bucket is the same
     * as the previous one
     */
    bool firstDiff = false;
    if (prevRight[0] != local_B1[0] || prevRight[1] != local_B2[0])
    {
      firstDiff = true;
    }

    // set local_B1 to `1` if previous entry is different:
    // i.e., mark start of buckets
    bool nextDiff = firstDiff;
    for (std::size_t i = 0; i+1 < local_B1.size(); ++i)
    {
        bool setOne = nextDiff;
        nextDiff = (local_B1[i] != local_B1[i+1] || local_B2[i] != local_B2[i+1]);
        local_B1[i] = setOne ? 1 : 0;
    }

    local_B1.back() = nextDiff ? 1 : 0;

    if (count_unfinished)
    {
        // mark 1->0 transitions with 1, if i am the zero and previous is 1
        // (i.e. identical)
        // (i.e. `i` is the second equal element in a bucket)
        // which means counting unfinished buckets, then allreduce
        index_t local_unfinished_buckets = firstDiff ? 0 : 1;
        if (rank == 0)
            local_unfinished_buckets = 0;
        for (std::size_t i = 1; i < local_B1.size(); ++i)
        {
            if(local_B1[i-1] == 1 && local_B1[i] == 0)
                ++local_unfinished_buckets;
        }

        MPI_Allreduce(&local_unfinished_buckets, &result, 1,
                      mpi_dt, MPI_SUM, comm);
    }

    /*
     * run global prefix sum on local_B1
     * this will result in the new bucket numbers in B1
     */
    global_prefix_sum(local_B1.begin(), local_B1.end(), comm);

    return result;
}


void reorder_sa_to_isa(std::size_t n, std::vector<index_t>& local_SA, std::vector<index_t>& local_B, MPI_Comm comm)
{
    assert(local_SA.size() == local_B.size());
    // get processor id
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);
    MPI_Datatype mpi_dt = get_mpi_dt<index_t>();

    // 1.) local bucketing for each processor
    //
    // counting the number of elements for each processor
    std::vector<int> send_counts(p, 0);
    for (index_t sa : local_SA)
    {
        int target_p = block_partition_target_processor(n, p, static_cast<std::size_t>(sa));
        assert(0 <= target_p && target_p < p);
        ++send_counts[target_p];
    }
    std::vector<int> send_displs = get_displacements(send_counts);
    std::vector<index_t> send_SA(local_SA.size());
    std::vector<index_t> send_B(local_B.size());
    // Reorder the SA and B arrays into buckets, one for each target processor.
    // The target processor is given by the value in the SA.
    for (std::size_t i = 0; i < local_SA.size(); ++i)
    {
        int target_p = block_partition_target_processor(n, p, static_cast<std::size_t>(local_SA[i]));
        assert(target_p < p && target_p >= 0);
        std::size_t out_idx = send_displs[target_p]++;
        assert(out_idx < local_SA.size());
        send_SA[out_idx] = local_SA[i];
        send_B[out_idx] = local_B[i];
    }

    // get displacements again (since they were modified above)
    send_displs = get_displacements(send_counts);
    // get receive information
    std::vector<int> recv_counts = all2allv_get_recv_counts(send_counts, comm);
    std::vector<int> recv_displs = get_displacements(recv_counts);

    // perform the all2all communication
    MPI_Alltoallv(&send_B[0], &send_counts[0], &send_displs[0], mpi_dt,
                  &local_B[0], &recv_counts[0], &recv_displs[0], mpi_dt,
                  comm);
    MPI_Alltoallv(&send_SA[0], &send_counts[0], &send_displs[0], mpi_dt,
                  &local_SA[0], &recv_counts[0], &recv_displs[0], mpi_dt,
                  comm);

    // rearrange locally
    // TODO [ENH]: more cache efficient by sorting rather than random assignment
    for (std::size_t i = 0; i < local_SA.size(); ++i)
    {
        index_t out_idx = local_SA[i] - block_partition_excl_prefix_size(n, p, rank);
        assert(0 <= out_idx && out_idx < local_SA.size());
        send_B[out_idx] = local_B[i];
    }
    std::copy(send_B.begin(), send_B.end(), local_B.begin());

    // reassign the SA
    std::size_t global_offset = block_partition_excl_prefix_size(n, p, rank);
    for (std::size_t i = 0; i < local_SA.size(); ++i)
    {
        local_SA[i] = global_offset + i;
    }
}

// in: 2^m, B1
// out: B2
void shift_buckets(std::size_t n, std::size_t dist, std::vector<index_t>& local_B1, std::vector<index_t>& local_B2, MPI_Comm comm)
{
    // get MPI comm parameters
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);
    MPI_Datatype mpi_dt = get_mpi_dt<index_t>();

    // get # elements to the left
    std::size_t prev_size = block_partition_excl_prefix_size(n, p, rank);
    std::size_t local_size = block_partition_local_size(n, p, rank);
    assert(local_size == local_B1.size());

    // init B2
    if (local_B2.size() != local_size){
        local_B2.clear();
        local_B2.resize(local_size, 0);
    }

    MPI_Request recv_reqs[2];
    int n_irecvs = 0;
    // receive elements from the right
    if (prev_size + dist < n)
    {
        std::size_t right_first_gl_idx = prev_size + dist;
        int p1 = block_partition_target_processor(n, p, right_first_gl_idx);

        std::size_t p1_gl_end = block_partition_prefix_size(n, p, p1);
        std::size_t p1_recv_cnt = p1_gl_end - right_first_gl_idx;

        if (p1 != rank)
        {
            // only receive if the source is not myself (i.e., `rank`)
            // [otherwise results are directly written instead of MPI_Sended]
            assert(p1_recv_cnt < std::numeric_limits<int>::max());
            int recv_cnt = p1_recv_cnt;
            // TODO: MPI_Datatype
            MPI_Irecv(&local_B2[0],recv_cnt, mpi_dt, p1,
                      PSAC_TAG_SHIFT, comm, &recv_reqs[n_irecvs++]);
        }

        if (p1_recv_cnt < local_size && p1 != p-1)
        {
            // also receive from one more processor
            int p2 = p1+1;
            // since p2 has at least local_size - 1 elements and at least
            // one element came from p1, we can assume that the receive count
            // is our local size minus the already received elements
            std::size_t p2_recv_cnt = local_size - p1_recv_cnt;

            assert(p2_recv_cnt < std::numeric_limits<int>::max());
            int recv_cnt = p2_recv_cnt;
            // send to `p1` (which is necessarily different from `rank`)
            // TODO: MPI_Datatype
            MPI_Irecv(&local_B2[0] + p1_recv_cnt, recv_cnt, mpi_dt, p2,
                      PSAC_TAG_SHIFT, comm, &recv_reqs[n_irecvs++]);
        }
    }

    // send elements to the left (split to at most 2 target processors)
    if (prev_size + local_size - 1 >= dist)
    {
        int p1 = -1;
        if (prev_size >= dist)
        {
            std::size_t first_gl_idx = prev_size - dist;
            p1 = block_partition_target_processor(n, p, first_gl_idx);
        }
        std::size_t last_gl_idx = prev_size + local_size - 1 - dist;
        int p2 = block_partition_target_processor(n, p, last_gl_idx);

        std::size_t local_split;
        if (p1 != p2)
        {
            // local start index of area for second processor
            if (p1 >= 0)
            {
                local_split = block_partition_prefix_size(n, p, p1) + dist - prev_size;
                // send to first processor
                assert(p1 != rank);
                MPI_Send(&local_B1[0], local_split,
                         mpi_dt, p1, PSAC_TAG_SHIFT, comm);
            }
            else
            {
                // p1 doesn't exist, then there is no prefix to add
                local_split = dist - prev_size;
            }
        }
        else
        {
            // only one target processor
            local_split = 0;
        }

        if (p2 != rank)
        {
            MPI_Send(&local_B1[0] + local_split, local_size - local_split,
                     mpi_dt, p2, PSAC_TAG_SHIFT, comm);
        }
        else
        {
            // in this case the split should be exactly at `dist`
            assert(local_split == dist);
            // locally reassign
            for (std::size_t i = local_split; i < local_size; ++i)
            {
                local_B2[i-local_split] = local_B1[i];
            }
        }
    }

    // wait for successful receive:
    MPI_Waitall(n_irecvs, recv_reqs, MPI_STATUS_IGNORE);
}

struct TwoBSA
{
    index_t B1;
    index_t B2;
    index_t SA;

    inline bool operator<(const TwoBSA& other) const
    {
        // tuple comparison of (B1, B2) with precedence to B1
        return (this->B1 < other.B1)
            || (this->B1 == other.B1 && this->B2 < other.B2);
    }
};

// template specialization for MPI_Datatype for the two buckets and SA structure
// Needed for the samplesort implementation
template<>
MPI_Datatype get_mpi_dt<TwoBSA>()
{
    // keep only one instance around
    // NOTE: this memory will not be destructed.
    MPI_Datatype dt;
    MPI_Datatype element_t = get_mpi_dt<index_t>();
    MPI_Type_contiguous(3, element_t, &dt);
    return dt;
}

// in: B1, B2
// out: reordered B1, B2; and SA
void isa_2b_to_sa(std::size_t n, std::vector<index_t>& B1, std::vector<index_t>& B2, std::vector<index_t>& SA, MPI_Comm comm)
{
    // check input sizes
    std::size_t local_size = B1.size();
    assert(B2.size() == local_size);
    assert(SA.size() == local_size);
    // initialize tuple array
    std::vector<TwoBSA> tuple_vec(local_size);

    // get comm parameters
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    // get global index offset
    std::size_t str_offset = block_partition_excl_prefix_size(n, p, rank);

    // fill tuple vector
    for (std::size_t i = 0; i < local_size; ++i)
    {
        tuple_vec[i].B1 = B1[i];
        tuple_vec[i].B2 = B2[i];
        assert(str_offset + i < std::numeric_limits<index_t>::max());
        tuple_vec[i].SA = str_offset + i;
    }

    // release memory of input (to remain at the minimum 6x words memory usage)
    B1.clear(); B1.shrink_to_fit();
    B2.clear(); B2.shrink_to_fit();
    SA.clear(); SA.shrink_to_fit();

    // parallel, distributed sample-sorting of tuples (B1, B2, SA)
    samplesort(tuple_vec.begin(), tuple_vec.end(), std::less<TwoBSA>());

    // reallocate output
    B1.resize(local_size);
    B2.resize(local_size);
    SA.resize(local_size);

    // read back into input vectors
    for (std::size_t i = 0; i < local_size; ++i)
    {
        B1[i] = tuple_vec[i].B1;
        B2[i] = tuple_vec[i].B2;
        SA[i] = tuple_vec[i].SA;
    }
}

bool gl_check_correct_SA(const std::vector<index_t> SA, const std::vector<index_t>& ISA, const std::string& str)
{
    std::size_t n = SA.size();
    bool success = true;

    for (std::size_t i = 0; i < n; ++i)
    {
        // check valid range
        if (SA[i] >= n || SA[i] < 0)
        {
            std::cerr << "[ERROR] SA[" << i << "] = " << SA[i] << " out of range 0 <= sa < " << n << std::endl;
            success = false;
        }

        // check SA conditions
        if (i >= 1 && SA[i-1] < n-1)
        {
            if (!(str[SA[i]] >= str[SA[i-1]]))
            {
                std::cerr << "[ERROR] wrong SA order: str[SA[i]] >= str[SA[i-1]]" << std::endl;
                success = false;
            }

            // if strings are equal, the ISA of these positions have to be
            // ordered
            if (str[SA[i]] == str[SA[i-1]])
            {
                if (!(ISA[SA[i-1]+1] < ISA[SA[i]+1]))
                {
                    std::cerr << "[ERROR] invalid SA order: ISA[SA[" << i-1 << "]+1] < ISA[SA[" << i << "]+1]" << std::endl;
                    std::cerr << "[ERROR] where SA[i-1]=" << SA[i-1] << ", SA[i]=" << SA[i] << ", ISA[SA[i-1]+1]=" << ISA[SA[i-1]+1] << ", ISA[SA[i]+1]=" << ISA[SA[i]+1] << std::endl;
                    success = false;
                }
            }
        }
    }

    return success;
}

void sa_construction(const std::string& local_str, std::vector<index_t>& local_SA, std::vector<index_t>& local_B, MPI_Comm comm)
{
    // get comm parameters
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    // create initial k-mers and use these as the initial bucket numbers
    // for each character position
    // `k` depends on the alphabet size and the word size of each suffix array
    // element. `k` is choosen to maximize the number of alphabet characters
    // that fit into one machine word
    initial_bucketing(local_str, local_B, comm);

    // TODO: get `n` once by allreduction
    std::size_t n = p*local_str.size();
    // TODO: assert(local_size == block_decomposition(n, p, rank));

    // init local_SA
    if (local_SA.size() != local_B.size())
    {
        local_SA.resize(local_B.size());
    }

    // TODO: start loop at shift size of k (from inital bucketing)
    for (std::size_t i = 2; i < n; i <<= 1)
    {
        // shifting by 2^x
        std::vector<index_t> local_B2;
        shift_buckets(n, i, local_B, local_B2, comm);
#ifndef NDEBUG
        std::cerr << "========  After shift by " << i << "  ========" << std::endl;
        std::cerr << "On processor rank = " << rank << std::endl;
        std::cerr << "B : "; print_vec(local_B);
        std::cerr << "B2: "; print_vec(local_B2);
#endif

        // ISA -> SA (both buckets) [sorting first by bucket 1]
        isa_2b_to_sa(n, local_B, local_B2, local_SA, comm);
#ifndef NDEBUG
        std::cerr << "========  After reorder ISA->SA  ========" << std::endl;
        std::cerr << "On processor rank = " << rank << std::endl;
        std::cerr << "B : "; print_vec(local_B);
        std::cerr << "B2: "; print_vec(local_B2);
        std::cerr << "SA: "; print_vec(local_SA);
#endif

        // rebucketing (assign new B1 bucket numbers)
        std::size_t unfinished_buckets = rebucket(local_B, local_B2, comm ,true);
        if (rank == 0)
            std::cerr << "iteration " << i << ": unfinished buckets = " << unfinished_buckets << std::endl;
#ifndef NDEBUG
        std::cerr << "========  After rebucket  ========" << std::endl;
        std::cerr << "On processor rank = " << rank << std::endl;
        std::cerr << "B : "; print_vec(local_B);
#endif

        // SA->ISA
        if (i << 1 >= n || unfinished_buckets == 0)
        {
            // if last iteration, use copy of local_SA for reorder and keep
            // original SA
            std::vector<index_t> cpy_SA(local_SA);
            reorder_sa_to_isa(n, cpy_SA, local_B, comm);
        }
        else
        {
            reorder_sa_to_isa(n, local_SA, local_B, comm);
        }
#ifndef NDEBUG
        std::cerr << "========  After reorder SA->ISA  ========" << std::endl;
        std::cerr << "On processor rank = " << rank << std::endl;
        std::cerr << "B : "; print_vec(local_B);
        std::cerr << "SA: "; print_vec(local_SA);
#endif
        if (unfinished_buckets == 0)
            break;
    }

    // now local_SA is actual SA and local_B is actual ISA
}

inline char rand_dna_char()
{
    char DNA[4] = {'A', 'C', 'G', 'T'};
    return DNA[rand() % 4];
}

std::string rand_dna(std::size_t size, int seed)
{
    srand(1337*seed);
    std::string str;
    str.resize(size, ' ');
    for (std::size_t i = 0; i < size; ++i)
    {
        str[i] = rand_dna_char();
    }
    return str;
}

void test_sa(MPI_Comm comm, std::size_t input_size, bool test_correct = false)
{
    // get comm parameters
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    //std::string local_str = "missisippi";
    std::string local_str = rand_dna(input_size, rank);

    std::vector<index_t> local_SA;
    std::vector<index_t> local_ISA;

    // construct local SA for input string
    sa_construction(local_str, local_SA, local_ISA, comm);

    // final SA and ISA
    if (test_correct)
    {
        // gather SA and ISA to local
        std::vector<index_t> global_SA = gather_vectors(local_SA, comm);
        std::vector<index_t> global_ISA = gather_vectors(local_ISA, comm);
        std::vector<char> global_str_vec = gather_range(local_str.begin(), local_str.end(), comm);
        std::string global_str(global_str_vec.begin(), global_str_vec.end());
        if (rank == 0)
        {
#ifndef NDEBUG
            std::cerr << "##################################################" << std::endl;
            std::cerr << "#               Final SA and ISA                 #" << std::endl;
            std::cerr << "##################################################" << std::endl;
            std::cerr << "STR: " << global_str << std::endl;
            std::cerr << "SA : "; print_vec(global_SA);
            std::cerr << "ISA: "; print_vec(global_ISA);
#endif

            // check if correct
            if (!gl_check_correct_SA(global_SA, global_ISA, global_str))
            {
                std::cerr << "[ERROR] Test unsuccessful" << std::endl;
                exit(1);
            }
            else
            {
                std::cerr << "[SUCCESS]" << std::endl;
            }
        }
    }
    std::cerr << " === Rank " << rank << " is finished === " << std::endl;
}


void my_mpi_errorhandler(MPI_Comm* comm, int* errorcode, ...)
{
    // throw exception, enables gdb stack trace analysis
    throw std::runtime_error("Shit: mpi fuckup");
}

int main(int argc, char *argv[])
{
    // set up MPI
    MPI_Init(&argc, &argv);


    // get communicator size and my rank
    MPI_Comm comm = MPI_COMM_WORLD;
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    // set custom error handler (for debugging with working stack-trace on gdb)
    //MPI_Errhandler errhandler;
    //MPI_Errhandler_create(&my_mpi_errorhandler, &errhandler);
    //MPI_Errhandler_set(comm, errhandler);

    // attach to process 0
    //wait_gdb_attach(0, comm);
    test_sa(comm, 20000000);
    //test_sa(comm, 100, true);

    // finalize MPI
    MPI_Finalize();
    return 0;
}
