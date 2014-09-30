
#include <mpi.h>


/*
 * TODO;
 * - [ ] read string and distribute by block decomposition
 * - [ ] initial bucketing (local -> distributed)
 * - [ ] code for transfering SA -> ISA
 * - [ ] code for transfering ISA -> SA
 * - [ ] prefix sum for backet numbers
 * - [ ] prefix doubling
 */


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


#include <vector>

#include <assert.h>

#include "algos.hpp"
#include "parallel_utils.hpp"
#include "mpi_utils.hpp"

#define PSAC_TAG_EDGE_KMER 1

typedef int kmer_t;
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



void initial_bucketing(const index_t n, const std::basic_string<char>& local_str, const MPI_Comm comm, std::vector<index_t>& local_B, std::vector<index_t>& local_SA)
{
    // TODO: based on alphabet size and kmer size and k
    static constexpr unsigned int k = 3;

    // sigma^k
    // assumption: sigma = 2^l for some l, TODO: otherwise set l=ceil(log(sigma))
    // then: histsize = 2^(l*k)
    // TODO: choose k dynamically based on l and the input size !?
    // TODO: bucket count is only efficient as long as the histrogram table
    //       does not exceed a certain size, this has to be tested
    //       experimentally
    //       as soon as it gets inefficient, normal sorting or bucket sorting
    //       ( non exact bucketing) might get much more efficient
    static constexpr unsigned int l = 8; // bits per character
    static constexpr std::size_t hist_size = static_cast<std::size_t>(1) << (l*k);
    static constexpr kmer_t kmer_mask = ((static_cast<kmer_t>(1) << (l*k)) - static_cast<kmer_t>(1));
    std::vector<count_t> hist(hist_size, 0);

    // sliding window k-mer (for prototype only using ASCII alphabet)

    // fill first k-mer (until k-1 positions) and send to left processor
    // filling k-mer with first character = MSBs for lexicographical ordering
    auto str_it = local_str.begin();
    kmer_t kmer = 0;
    for (unsigned int i = 0; i < k-1; ++i)
    {
        kmer <<= l;
        kmer |= static_cast<kmer_t>(*str_it);
        ++str_it;
    }

    // send this to left processor, start async receive from right processor
    // for last in seq
    int rank, p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);


    // start receiving for end
    kmer_t last_kmer = 0;
    MPI_Request recv_req;
    if (rank < p-1) // if not last processor
    {
        MPI_Irecv(&last_kmer, 1, MPI_UINT64_T, rank+1, PSAC_TAG_EDGE_KMER,
                  comm, &recv_req);
    }
    if (rank > 0) // if not first processor
    {
        // TODO: [ENH] use async send as well and start with the computation
        //             immediately
        MPI_Send(&kmer, 1,MPI_UINT64_T, rank-1, PSAC_TAG_EDGE_KMER, comm);
    }

    std::vector<index_t> buckets(local_str.size());

    auto buk_it = buckets.begin();
    // continue to create all k-mers and add into histogram count
    while (str_it != local_str.end())
    {
        // get next kmer
        kmer <<= l;
        kmer |= static_cast<kmer_t>(*str_it);
        kmer &= kmer_mask;
        // add to bucket number array
        *buk_it = kmer;
        // count in histogram
        ++hist[kmer];
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
        // DNA strings? (5 character alphabet seems dumb)
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
        // count in histogram
        ++hist[kmer];
    }


    // TODO:
    // - local bucketing tuples (B[i], i) [needs internal excl_prefix_sum !?]
    //   -> kinda is the local SA which then gets distributed
    //   -> can I do this in-place by switching around buckets while keeping
    //      track of all permutations with a permutation array (i.e., the SA)?
    // - get borders for sending
    // - all2all

    // local_hist_prefix
    // TODO: [ENH] instead of copying, can revert prefix sum in additional O(n) pass
    //             this comes down to a mem vs. computation tradeoff, since both
    //             are only constant factors
    std::vector<count_t> local_hist_prefix(hist);
    excl_prefix_sum(local_hist_prefix.begin(), local_hist_prefix.end());

    // for local SA
    local_SA = std::vector<index_t>(local_str.size());
    // for local buckets
    // TODO: [ENH] could scan input string again instead of needing another
    //             copy of B for reordering, this would remove the whole
    //             B array (-> comp vs mem tradeoff)
    local_B = std::vector<index_t>(local_str.size());
    index_t str_off = block_partition_excl_prefix_size(n, p, rank);

    for (std::size_t i = 0; i < buckets.size(); ++i)
    {
        kmer_t bucket_no = buckets[i];
        index_t out_pos = local_hist_prefix[bucket_no]++;

        // save buckets in sorted order together with their original
        // position in the string (~> incomplete SA)
        // TODO: this is not really cache efficient, could be done better
        local_B[out_pos] = bucket_no;
        local_SA[out_pos] = i + str_off;
        // for each bucket, the elements are now sorted by (bucket-no,
        // str-index) where `str-index` is the SA number, since we linearly
        // scanned the input
    }

    // print out all:



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
/*
#ifndef NDEBUG
    std::cerr << "Send count on rank = " << rank << ": "; print_vec(send_counts);
    std::cerr << "Recv count on rank = " << rank << ": "; print_vec(recv_counts);
    std::cerr << "send displs on rank = " << rank << ": "; print_vec(send_displs);
    std::cerr << "Recv displs on rank = " << rank << ": "; print_vec(recv_displs);
#endif
*/
    // allocate receive buffers (TODO [ENH]: receive in place and then do the
    // bucket sort below also in-place)
    std::vector<index_t> recv_B(local_B.size());// = buckets; // TODO: assert that all those are actually equal to the local block partition size
    std::vector<index_t> recv_SA(local_SA.size());
    // TODO: correct MPI_Datatype
    MPI_Alltoallv(&local_B[0], &send_counts[0], &send_displs[0], MPI_INT,
                  &recv_B[0], &recv_counts[0], &recv_displs[0], MPI_INT,
                  comm);
    MPI_Alltoallv(&local_SA[0], &send_counts[0], &send_displs[0], MPI_INT,
                  &recv_SA[0], &recv_counts[0], &recv_displs[0], MPI_INT,
                  comm);

/*
#ifndef NDEBUG
    std::cerr << "================= After All2All ===============" << std::endl;
    std::cerr << "Send count on rank = " << rank << ": "; print_vec(send_counts);
    std::cerr << "Recv count on rank = " << rank << ": "; print_vec(recv_counts);
    std::cerr << "send displs on rank = " << rank << ": "; print_vec(send_displs);
    std::cerr << "Recv displs on rank = " << rank << ": "; print_vec(recv_displs);
    std::cerr << "local B :"; print_vec(local_B);
    std::cerr << "local SA:"; print_vec(local_SA);
    std::cerr << "recv  B :"; print_vec(recv_B);
    std::cerr << "recv  SA:"; print_vec(recv_SA);
#endif
*/


    /* local rearranging */

    /*
    // - local rearrange (stable bucket sort!?)
    // 1.) one pass to create new local hist
    //  [ TODO: how do i do this in-place? ]
    // reset histogram
    std::fill(hist.begin(), hist.end(), 0);
    // refill histogram with new content
    // TODO: could avoid full scan by reusing previous histograms and doing some
    //       math
    for (index_t b : recv_B)
    {
      assert(b < hist.size());
        ++hist[b];
    }
    */
    // since using the recv_displs, we don't need to re-create the histogram

    // two possible orders of iterating:
    //   1) linear through input and write "randomly
    //   2) linear for output/writes, read "randomly" (implementing this currently)
    // write sorted order (linear write, jump between recv_displacements in input)
    auto outit_B = local_B.begin();
    auto outit_SA = local_SA.begin();

    for (index_t b = 0; b < hist.size(); ++b)
    {
        for (int i = 0; i < p; ++i)
        {
            int recv_pos = recv_displs[i];
            while ((i+1 == p || recv_pos != recv_displs[i+1]) && recv_pos < recv_B.size() && recv_B[recv_pos] == b)
            {
                *(outit_B++) = recv_B[recv_pos];
                *(outit_SA++) = recv_SA[recv_pos];
                ++recv_pos;
            }
            recv_displs[i] = recv_pos;
        }
    }
}


// assumed sorted order (globally) by tuple (B1[i], B2[i])
// this reassigns new, unique bucket numbers in {1,...,n} globally
void rebucket(std::vector<index_t>& local_B1, std::vector<index_t>& local_B2, MPI_Comm comm)
{
    // assert inputs are of equal size
    assert(local_B1.size() == local_B2.size() && local_B1.size() > 0);

    // get processor id
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);


    /*
     * send right most element to one processor to the right
     * so that that processor can determine whether the same bucket continues
     * or a new bucket starts with it's first element
     */
    MPI_Request recv_req;
    index_t prevRight[2];
    if (rank > 0) // if not last processor
    {
        // TODO: fix MPI datatype
        MPI_Irecv(prevRight, 2, MPI_INT, rank-1, PSAC_TAG_EDGE_KMER,
                  comm, &recv_req);
    }
    if (rank < p-1) // if not first processor
    {
        // send my most right element to the right
        index_t myRight[2] = {local_B1.back(), local_B2.back()};
        // TODO: [ENH] use async send as well and start with the computation
        //             immediately
        MPI_Send(myRight, 2, MPI_INT, rank+1, PSAC_TAG_EDGE_KMER, comm);
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
    bool nextDiff = false;
    if (prevRight[0] != local_B1[0] || prevRight[1] != local_B2[0])
    {
      nextDiff = true;
    }

    for (std::size_t i = 0; i+1 < local_B1.size(); ++i)
    {
        bool setOne = nextDiff;
        nextDiff = (local_B1[i] != local_B1[i+1] || local_B2[i] != local_B2[i+1]);
        local_B1[i] = setOne ? 1 : 0;
    }

    local_B1.back() = nextDiff ? 1 : 0;

    /*
     * run global prefix sum on local_B1
     * this will result in the new bucket numbers in B1
     */
    global_prefix_sum(local_B1.begin(), local_B1.end(), comm);
}

// TODO:
// function for shift in ISA/B space, tupleling, reorder to SA space by first element
// of each tuple, then localized sorting of each bucket (equivalent to global sort)
// this could be done similar to the quicksort from the HPC lab (additional communicators)
// after local reordering: mark bucket starts and prefix sum for new bucket numbers
// reorder to ISA/B space (only new bucket numbers)

void reorder_sa_to_isa(std::size_t n, std::vector<index_t>& local_SA, std::vector<index_t>& local_B, MPI_Comm comm)
{
    // get processor id
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    // 1.) local bucketing for each processor
    std::vector<count_t> send_counts(p, 0);
    for (

    //   -> might be tricky [custom swap?]
    // 2.) get send_count (boundaries) -> easy
    // 3.)
}


void test_sa(MPI_Comm comm)
{
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    std::string local_str = "missisippi";

    std::vector<index_t> local_B;
    std::vector<index_t> local_SA;

    std::size_t n = p*local_str.size();

    initial_bucketing(n, local_str, comm, local_B, local_SA);

    std::cerr << "On processor rank = " << rank << std::endl;
    std::cerr << "B : "; print_vec(local_B);
    std::cerr << "SA: "; print_vec(local_SA);

    // re-assign new bucket numbers
    rebucket(local_B, local_B, comm);
    std::cerr << "B0: "; print_vec(local_B);
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

    // attach to process 0
    //wait_gdb_attach(0, comm);
    test_sa(comm);

    // finalize MPI
    MPI_Finalize();
    return 0;
}
