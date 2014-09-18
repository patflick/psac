
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
#define PSAC_TAG_EDGE_KMER 1

typedef uint64_t kmer_t;
// must be 64 bit int for strings > 4GB
// those need special handleing with MPI anyway, since MPI only works with ints
typedef uint32_t index_t; // content of SA and ISA and Bucket-numbers
typedef index_t count_t; // bucket counts for histogram

void kmer_gen(const std::basic_string<char>& local_str, std::vector<index_t>& B, const MPI_Comm comm)
{
    // TODO: resize B if it is not initialized


    // string needs to be overlapping by at least k-1 on each processor,
    // otherwise, send those ends (easy!)
    
    // TODO: based on alphabet size and kmer size and k
    static constexpr unsigned int k = 2;

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
    std::vector<count_t> hist(static_cast<std::size_t>(1) << (l*k), 0);

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

    auto buk_it = B.begin();
    // continue to create all k-mers and add into histogram count
    while (str_it != local_str.end())
    {
        // get next kmer
        kmer <<= l;
        kmer |= static_cast<kmer_t>(*str_it);
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


    // construct last k-mers
    for (unsigned int i = 0; i < k-1; ++i)
    {
        kmer <<= l;
        kmer |= (((1<<l) - 1) & (last_kmer >> (l*(k-i-2))));

        // add to bucket number array
        *buk_it = kmer;
        ++buk_it;
        // count in histogram
        ++hist[kmer];
    }

    // all_reduce the histogram
    MPI_Allreduce(&hist[0], &hist[0], hist.size(), MPI_UINT32_T, MPI_SUM, comm);

    // TODO: bucketing and all2all
    // - this could get tricky, since it isn't immediately clear which target
    //   processor each element will go to, should remain sorted by
    //   the tuple (bucket-no, string-index)
}

void initial_bucketing(local_string, SA, ISA)
{
    // sort k-mers of small size (fit into bucket no)
    // - fixed lookup table on each processor of size $\sigma^k$
    // - count by linear scan over string with window
    // - MPI reduction
    // - assign in order (bucket-no, 
    
    
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

    /* code */
    /* ... */

    // finalize MPI
    MPI_Finalize();
    return 0;
}
