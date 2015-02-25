#ifndef SUFFIX_ARRAY_HPP
#define SUFFIX_ARRAY_HPP

#include <mpi.h>
#include <vector>

#include "timer.hpp"
#include "partition.hpp"
#include "alphabet.hpp"
#include "mpi_samplesort.hpp"

// TODO: move these macros to somewhere else
/*********************************************************************
 *                 macros for debugging with distributed vectors     *
 *********************************************************************/
// whether to gather all vectors to rank 0 prior to debug output
// set both to `0` to disable debug output
#define DO_DEBUG_GLOBAL_VEC 0
#define DO_DEBUG_LOCAL_VEC 0

// print vector helpers
#define DEBUG_PRINT_STAGE(stage) \
    std::cerr << "========  " << stage << "  =======" << std::endl;\

#define DEBUG_PRINT_LOCAL_VEC(vec) \
    fprintf(stderr, "%-10s: ",#vec);print_vec(vec);

#define DEBUG_PRINT_GLOBAL_VEC(vec) \
    {\
        std::vector<index_t> gl_##vec = gather_vectors(vec,comm);\
        if (rank == 0) {\
            fprintf(stderr, "GLOBAL %-10s: ",#vec);print_vec(gl_##vec);\
    }}

// defining common macros for stage vector output
#if DO_DEBUG_GLOBAL_VEC
#define DEBUG_STAGE_VEC(stage, vec)\
        DEBUG_PRINT_STAGE(stage)\
        DEBUG_PRINT_GLOBAL_VEC(vec)
#define DEBUG_STAGE_VEC2(stage, vec, vec2)\
        DEBUG_PRINT_STAGE(stage)\
        DEBUG_PRINT_GLOBAL_VEC(vec)\
        DEBUG_PRINT_GLOBAL_VEC(vec2)
#define DEBUG_STAGE_VEC3(stage, vec, vec2, vec3)\
        DEBUG_PRINT_STAGE(stage)\
        DEBUG_PRINT_GLOBAL_VEC(vec)\
        DEBUG_PRINT_GLOBAL_VEC(vec2)\
        DEBUG_PRINT_GLOBAL_VEC(vec3)
#elif DO_DEBUG_LOCAL_VEC
#define DEBUG_STAGE_VEC(stage, vec)\
        DEBUG_PRINT_STAGE(stage)\
        DEBUG_PRINT_LOCAL_VEC(vec)
#define DEBUG_STAGE_VEC2(stage, vec, vec2)\
        DEBUG_PRINT_STAGE(stage)\
        DEBUG_PRINT_LOCAL_VEC(vec)\
        DEBUG_PRINT_LOCAL_VEC(vec2)
#define DEBUG_STAGE_VEC3(stage, vec, vec2, vec3)\
        DEBUG_PRINT_STAGE(stage)\
        DEBUG_PRINT_LOCAL_VEC(vec)\
        DEBUG_PRINT_LOCAL_VEC(vec2)\
        DEBUG_PRINT_LOCAL_VEC(vec3)
#else
#define DEBUG_STAGE_VEC(stage, vec)
#define DEBUG_STAGE_VEC2(stage, vec, vec2)
#define DEBUG_STAGE_VEC3(stage, vec, vec2, vec3)
#endif


/*********************************************************************
 *              Macros for timing sections in the code               *
 *********************************************************************/

#define SAC_ENABLE_TIMER 1
#if SAC_ENABLE_TIMER
#define SAC_TIMER_START() TIMER_START()
#define SAC_TIMER_END_SECTION(str) TIMER_END_SECTION(str)
#define SAC_TIMER_LOOP_START() TIMER_LOOP_START()
#define SAC_TIMER_END_LOOP_SECTION(iter, str) TIMER_END_LOOP_SECTION(iter, str)
#else
#define SAC_TIMER_START()
#define SAC_TIMER_END_SECTION(str)
#define SAC_TIMER_LOOP_START()
#define SAC_TIMER_END_LOOP_SECTION(iter, str)
#endif

template <typename T>
struct TwoBSA
{
    T B1;
    T B2;
    T SA;

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
MPI_Datatype get_mpi_dt<TwoBSA<unsigned int>>()
{
    // keep only one instance around
    // NOTE: this memory will not be destructed.
    MPI_Datatype dt;
    MPI_Datatype element_t = get_mpi_dt<unsigned int>();
    MPI_Type_contiguous(3, element_t, &dt);
    return dt;
}
template<>
MPI_Datatype get_mpi_dt<TwoBSA<int>>()
{
    // keep only one instance around
    // NOTE: this memory will not be destructed.
    MPI_Datatype dt;
    MPI_Datatype element_t = get_mpi_dt<int>();
    MPI_Type_contiguous(3, element_t, &dt);
    return dt;
}
template<>
MPI_Datatype get_mpi_dt<TwoBSA<std::size_t>>()
{
    // keep only one instance around
    // NOTE: this memory will not be destructed.
    MPI_Datatype dt;
    MPI_Datatype element_t = get_mpi_dt<std::size_t>();
    MPI_Type_contiguous(3, element_t, &dt);
    return dt;
}


// distributed suffix array
template <typename InputIterator, typename index_t = std::size_t>
class suffix_array
{
private:
public:
    suffix_array(InputIterator begin, InputIterator end, MPI_Comm comm = MPI_COMM_WORLD)
        : comm(comm), input_begin(begin), input_end(end)
    {
        // get communcation parameters
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &p);

        // the local size of the input
        local_size = std::distance(begin, end);

        // get local and global size by reduction
        MPI_Datatype mpi_size_t = get_mpi_dt<std::size_t>();
        MPI_Allreduce(&local_size, &n, 1, mpi_size_t, MPI_SUM, comm);
        part = partition::block_decomposition_buffered<index_t>(n, p, rank);

        // assert a block decomposition
        if (part.local_size() != local_size)
            throw std::runtime_error("The input string must be equally block decomposed accross all MPI processes.");

        // get mpi data type TODO: replace with new call once refactored
        mpi_index_t = get_mpi_dt<index_t>();
    }
    virtual ~suffix_array() {}
private:
    /// The global size of the input string and suffix array
    std::size_t n;
    /// The local size of the input string and suffix array
    /// is either floor(n/p) or ceil(n/p) and based on a equal block
    /// distribution
    std::size_t local_size;

    /// The MPI communicator to use for the parallel suffix array construction
    MPI_Comm comm;
    /// The number of processors in the communicator
    int p;
    /// The local processors rank among the processors in the MPI communicator
    int rank;
    /// The MPI datatype for the templated type `index_t`.
    MPI_Datatype mpi_index_t;

    /// Iterators over the local input string
    InputIterator input_begin;
    /// End iterator for local input string
    InputIterator input_end;

    // The block decomposition for the suffix array
    partition::block_decomposition_buffered<index_t> part;

public: // TODO: make private again and provide some iterator and query access
    /// The local suffix array
    std::vector<index_t> local_SA;
    /// The local inverse suffix array (TODO: rename?)
    std::vector<index_t> local_B;
    /// The local LCP array (remains empty if no LCP is constructed)
    std::vector<index_t> local_LCP;

private:

    // MPI tags used in constructing the suffix array
    static const int PSAC_TAG_EDGE_KMER = 1;
    static const int PSAC_TAG_SHIFT = 2;

public:
template <bool _CONSTRUCT_LCP=false>
void construct() {
    SAC_TIMER_START();
    // TODO: where is `n` and `local_size` initialized?

    /***********************
     *  Initial bucketing  *
     ***********************/

    // create initial k-mers and use these as the initial bucket numbers
    // for each character position
    // `k` depends on the alphabet size and the word size of each suffix array
    // element. `k` is choosen to maximize the number of alphabet characters
    // that fit into one machine word
    unsigned int k;
    unsigned int bits_per_char;
    std::tie(k, bits_per_char) = initial_bucketing();
    DEBUG_STAGE_VEC("after initial bucketing", local_B);
#if 0
    std::cerr << "========  After initial bucketing  ========" << std::endl;
    std::cerr << "On processor rank = " << rank << std::endl;
    std::cerr << "B : "; print_vec(local_B);
#endif

    SAC_TIMER_END_SECTION("initial-bucketing");

    // init local_SA
    if (local_SA.size() != local_B.size())
    {
        local_SA.resize(local_B.size());
    }

    std::vector<index_t> local_B_SA;
    std::size_t unfinished_buckets = 1<<k;
    std::size_t shift_by;

    /*******************************
     *  Prefix Doubling main loop  *
     *******************************/
    for (shift_by = k; shift_by < n; shift_by <<= 1)
    {
        SAC_TIMER_LOOP_START();
        /**************************************************
         *  Pairing buckets by shifting `shift_by` = 2^k  *
         **************************************************/
        // shift the B1 buckets by 2^k to the left => equals B2
        std::vector<index_t> local_B2;
        shift_buckets(shift_by, local_B2);
        DEBUG_STAGE_VEC2("after shift by " << shift_by, local_B, local_B2);
#if 0
        std::cerr << "========  After shift by " << shift_by << "  ========" << std::endl;
        std::cerr << "On processor rank = " << rank << std::endl;
        std::cerr << "B : "; print_vec(local_B);
        std::cerr << "B2: "; print_vec(local_B2);
#endif
        SAC_TIMER_END_LOOP_SECTION(shift_by, "shift-buckets");

        /*************
         *  ISA->SA  *
         *************/
        // by using sample sort on tuples (B1,B2)
        isa_2b_to_sa(local_B2);
        DEBUG_STAGE_VEC3("after reorder ISA->SA", local_B, local_B2, local_SA);
        SAC_TIMER_END_LOOP_SECTION(shift_by, "ISA-to-SA");

        /****************
         *  Update LCP  *
         ****************/
        // if this is the first iteration: create LCP, otherwise update
        /*
         * TODO: LCP!
        if (_CONSTRUCT_LCP)
        {
            if (shift_by == k) {
                initial_kmer_lcp(n, k, bits_per_char, local_B, local_B2, local_LCP, comm);
                SAC_TIMER_END_LOOP_SECTION(shift_by, "init-lcp");
            } else {
                resolve_next_lcp(n, shift_by, local_B, local_B2, local_LCP, comm);
                SAC_TIMER_END_LOOP_SECTION(shift_by, "update-lcp");
            }
            DEBUG_STAGE_VEC("LCP update", local_LCP);
        }
        */

        /*******************************
         *  Assign new bucket numbers  *
         *******************************/
        unfinished_buckets = rebucket(local_B2, true);
        if (rank == 0)
            std::cerr << "iteration " << shift_by << ": unfinished buckets = " << unfinished_buckets << std::endl;
        DEBUG_STAGE_VEC("after rebucket", local_B);
        SAC_TIMER_END_LOOP_SECTION(shift_by, "rebucket");

        /*************
         *  SA->ISA  *
         *************/
        // by bucketing to correct target processor using the `SA` array
        // // TODO by number of unresolved elements rather than buckets!!
        if(false)
        //if (unfinished_buckets < n/10)
        {
            // prepare for bucket chaising (needs SA, and bucket arrays in both
            // SA and ISA order)
            std::vector<index_t> cpy_SA(local_SA);
            local_B_SA = local_B; // copy
            reorder_sa_to_isa(cpy_SA);
            SAC_TIMER_END_LOOP_SECTION(shift_by, "SA-to-ISA");
            break;
        }
        else if ((shift_by << 1) >= n || unfinished_buckets == 0)
        {
            // if last iteration, use copy of local_SA for reorder and keep
            // original SA
            std::vector<index_t> cpy_SA(local_SA);
            reorder_sa_to_isa(cpy_SA);
            SAC_TIMER_END_LOOP_SECTION(shift_by, "SA-to-ISA");
        }
        else
        {
            reorder_sa_to_isa();
            SAC_TIMER_END_LOOP_SECTION(shift_by, "SA-to-ISA");
        }
        DEBUG_STAGE_VEC2("after reorder SA->ISA", local_B, local_SA);

        // end iteratior
        SAC_TIMER_END_SECTION("sac-iteration");

        // check for termination condition
        if (unfinished_buckets == 0)
            break;
    }

    if (unfinished_buckets > 0)
    {
        if (rank == 0)
            std::cerr << "Starting Bucket chasing algorithm" << std::endl;
        // TODO: enable bucket chaising construction
        return;
        //sa_bucket_chaising_constr(n, local_SA, local_B_SA, local_B, comm, shift_by);
    }

    // now local_SA is actual block decomposed SA and local_B is actual ISA with an offset of one
    for (std::size_t i = 0; i < local_B.size(); ++i)
    {
        // the buffer indeces are `1` based indeces, but the ISA should be
        // `0` based indeces
        local_B[i] -= 1;
    }
}


private:

/*********************************************************************
 *                         Initial Bucketing                         *
 *********************************************************************/
// TODO: externalize some code as "k-mer generation"
std::pair<unsigned int, unsigned int> initial_bucketing()
{
    std::size_t min_local_size = part.local_size(p-1);

    // get global alphabet histogram
    std::vector<index_t> alphabet_hist = alphabet_histogram<InputIterator, index_t>(input_begin, input_end, comm);
    // get mapping table and alphabet sizes
    std::vector<char> alphabet_mapping = alphabet_mapping_tbl(alphabet_hist);
    unsigned int sigma = alphabet_unique_chars(alphabet_hist);
    // bits per character: set l=ceil(log(sigma))
    unsigned int l = alphabet_bits_per_char(sigma);
    // number of characters per word => the `k` in `k-mer`
    unsigned int k = alphabet_chars_per_word<index_t>(l);

    // TODO: during current debugging:
    k = 1;
    // if the input is too small for `k`, choose a smaller `k`
    if (k > min_local_size)
    {
        k = min_local_size;
    }

    if (rank == 0)
        std::cerr << "Detecting sigma=" << sigma << " => l=" << l << ", k=" << k
                  << std::endl;

    // get k-mer mask
    index_t kmer_mask = ((static_cast<index_t>(1) << (l*k)) - static_cast<index_t>(1));
    if (kmer_mask == 0)
        kmer_mask = ~static_cast<index_t>(0);

    // sliding window k-mer (for prototype only using ASCII alphabet)

    // fill first k-mer (until k-1 positions) and send to left processor
    // filling k-mer with first character = MSBs for lexicographical ordering
    InputIterator str_it = input_begin;
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
    // TODO: replace by some utility linear shift communcation function!!
    index_t last_kmer = 0;
    MPI_Request recv_req;
    if (rank < p-1) // if not last processor
    {
        MPI_Irecv(&last_kmer, 1, mpi_index_t, rank+1, PSAC_TAG_EDGE_KMER,
                  comm, &recv_req);
    }
    if (rank > 0) // if not first processor
    {
        MPI_Send(&kmer, 1, mpi_index_t, rank-1, PSAC_TAG_EDGE_KMER, comm);
    }


    // init output
    if (local_B.size() != local_size)
        local_B.resize(local_size);
    auto buk_it = local_B.begin();
    // continue to create all k-mers and add into histogram count
    while (str_it != input_end)
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

    // return the number of characters which are part of each bucket number
    // (i.e., k-mer)
    return std::make_pair(k, l);
}

// in: 2^m, B1
// out: B2
void shift_buckets(std::size_t dist, std::vector<index_t>& local_B2)
{
    // get # elements to the left
    std::size_t prev_size = part.excl_prefix_size();
    assert(local_size == local_B.size());

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
        int p1 = part.target_processor(right_first_gl_idx);

        std::size_t p1_gl_end = part.prefix_size(p1);
        std::size_t p1_recv_cnt = p1_gl_end - right_first_gl_idx;

        if (p1 != rank)
        {
            // only receive if the source is not myself (i.e., `rank`)
            // [otherwise results are directly written instead of MPI_Sended]
            assert(p1_recv_cnt < std::numeric_limits<int>::max());
            int recv_cnt = p1_recv_cnt;
            MPI_Irecv(&local_B2[0],recv_cnt, mpi_index_t, p1,
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
            MPI_Irecv(&local_B2[0] + p1_recv_cnt, recv_cnt, mpi_index_t, p2,
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
            p1 = part.target_processor(first_gl_idx);
        }
        std::size_t last_gl_idx = prev_size + local_size - 1 - dist;
        int p2 = part.target_processor(last_gl_idx);

        std::size_t local_split;
        if (p1 != p2)
        {
            // local start index of area for second processor
            if (p1 >= 0)
            {
                local_split = part.prefix_size(p1) + dist - prev_size;
                // send to first processor
                assert(p1 != rank);
                MPI_Send(&local_B[0], local_split,
                         mpi_index_t, p1, PSAC_TAG_SHIFT, comm);
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
            MPI_Send(&local_B[0] + local_split, local_size - local_split,
                     mpi_index_t, p2, PSAC_TAG_SHIFT, comm);
        }
        else
        {
            // in this case the split should be exactly at `dist`
            assert(local_split == dist);
            // locally reassign
            for (std::size_t i = local_split; i < local_size; ++i)
            {
                local_B2[i-local_split] = local_B[i];
            }
        }
    }

    // wait for successful receive:
    MPI_Waitall(n_irecvs, recv_reqs, MPI_STATUS_IGNORE);
}


void isa_2b_to_sa(std::vector<index_t>& local_B2)
{
    assert(local_B2.size() == local_size);
    SAC_TIMER_START();

    // initialize tuple array
    std::vector<TwoBSA<index_t> > tuple_vec(local_size);

    // get global index offset
    std::size_t str_offset = part.excl_prefix_size();

    // fill tuple vector
    for (std::size_t i = 0; i < local_size; ++i)
    {
        tuple_vec[i].B1 = local_B[i];
        tuple_vec[i].B2 = local_B2[i];
        assert(str_offset + i < std::numeric_limits<index_t>::max());
        tuple_vec[i].SA = str_offset + i;
    }

    // release memory of input (to remain at the minimum 6x words memory usage)
    local_B.clear(); local_B.shrink_to_fit();
    local_B2.clear(); local_B2.shrink_to_fit();
    local_SA.clear(); local_SA.shrink_to_fit();

    SAC_TIMER_END_SECTION("isa2sa_tupleize");

    // parallel, distributed sample-sorting of tuples (B1, B2, SA)
    if(rank == 0)
        std::cerr << "  sorting local size = " << tuple_vec.size() << std::endl;
    samplesort(tuple_vec.begin(), tuple_vec.end(), std::less<TwoBSA<index_t> >());

    SAC_TIMER_END_SECTION("isa2sa_samplesort");

    // reallocate output
    local_B.resize(local_size);
    local_B2.resize(local_size);
    local_SA.resize(local_size);

    // read back into input vectors
    for (std::size_t i = 0; i < local_size; ++i)
    {
        local_B[i] = tuple_vec[i].B1;
        local_B2[i] = tuple_vec[i].B2;
        local_SA[i] = tuple_vec[i].SA;
    }
    SAC_TIMER_END_SECTION("isa2sa_untupleize");
}

// assumed sorted order (globally) by tuple (B1[i], B2[i])
// this reassigns new, unique bucket numbers in {1,...,n} globally
std::size_t rebucket(std::vector<index_t>& local_B2, bool count_unfinished)
{
    /*
     * NOTE: buckets are indexed by the global index of the first element in
     *       the bucket with a ONE-BASED-INDEX (since bucket number `0` is
     *       reserved for out-of-bounds)
     */
    // assert inputs are of equal size
    assert(local_B.size() == local_B2.size() && local_B.size() > 0);

    // init result
    std::size_t result = 0;

    /*
     * send right-most element to one processor to the right
     * so that that processor can determine whether the same bucket continues
     * or a new bucket starts with it's first element
     */
    MPI_Request recv_req;
    index_t prevRight[2];
    if (rank > 0) // if not last processor
    {
        MPI_Irecv(prevRight, 2, mpi_index_t, rank-1, PSAC_TAG_EDGE_KMER,
                  comm, &recv_req);
    }
    if (rank < p-1) // if not first processor
    {
        // send my most right element to the right
        index_t myRight[2] = {local_B.back(), local_B2.back()};
        MPI_Send(myRight, 2, mpi_index_t, rank+1, PSAC_TAG_EDGE_KMER, comm);
    }
    if (rank > 0)
    {
        // wait for the async receive to finish
        MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
    }

    // get my global starting index
    std::size_t prefix = part.excl_prefix_size();

    /*
     * assign local zero or one, depending on whether the bucket is the same
     * as the previous one
     */
    bool firstDiff = false;
    if (rank == 0)
    {
        firstDiff = true;
    }
    else if (prevRight[0] != local_B[0] || prevRight[1] != local_B2[0])
    {
        firstDiff = true;
    }

    // set local_B1 to `1` if previous entry is different:
    // i.e., mark start of buckets
    bool nextDiff = firstDiff;
    for (std::size_t i = 0; i+1 < local_B.size(); ++i)
    {
        bool setOne = nextDiff;
        nextDiff = (local_B[i] != local_B[i+1] || local_B2[i] != local_B2[i+1]);
        local_B[i] = setOne ? prefix+i+1 : 0;
    }

    local_B.back() = nextDiff ? prefix+(local_size-1)+1 : 0;

    // count unfinished buckets
    if (count_unfinished)
    {
        // mark 1->0 transitions with 1, if i am the zero and previous is 1
        // (i.e. identical)
        // (i.e. `i` is the second equal element in a bucket)
        // which means counting unfinished buckets, then allreduce
        index_t local_unfinished_buckets;
        index_t prev_right;
        // TODO: replace by a common shift communcation function!
        if (rank > 0) // if not last processor
        {
            MPI_Irecv(&prev_right, 1, mpi_index_t, rank-1, PSAC_TAG_EDGE_KMER,
                      comm, &recv_req);
        }
        if (rank < p-1) // if not first processor
        {
            // send my most right element to the right
            MPI_Send(&local_B.back(), 1, mpi_index_t, rank+1, PSAC_TAG_EDGE_KMER, comm);
        }
        if (rank > 0)
        {
            // wait for the async receive to finish
            MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
        }
        if (rank == 0)
            local_unfinished_buckets = 0;
        else
            local_unfinished_buckets = (prev_right > 0 && local_B[0] == 0) ? 1 : 0;
        for (std::size_t i = 1; i < local_B.size(); ++i)
        {
            if(local_B[i-1] > 0 && local_B[i] == 0)
                ++local_unfinished_buckets;
        }

        index_t total_unfinished;
        MPI_Allreduce(&local_unfinished_buckets, &total_unfinished, 1,
                      mpi_index_t, MPI_SUM, comm);
        result = total_unfinished;
    }

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
    auto rev_it = local_B.rbegin();
    std::size_t local_max = 0;
    while (rev_it != local_B.rend() && (local_max = *rev_it) == 0)
        ++rev_it;

    // 2.) distributed scan with max() to get starting max for each sequence
    std::size_t pre_max;
    MPI_Datatype mpi_size_t = get_mpi_dt<std::size_t>();
    MPI_Exscan(&local_max, &pre_max, 1, mpi_size_t, MPI_MAX, comm);
    if (rank == 0)
        pre_max = 0;

    // 3.) linear scan and assign bucket numbers
    for (std::size_t i = 0; i < local_B.size(); ++i)
    {
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


void reorder_sa_to_isa(std::vector<index_t>& SA)
{
    assert(SA.size() == local_B.size());

    SAC_TIMER_START();
    // 1.) local bucketing for each processor
    //
    // counting the number of elements for each processor
    std::vector<int> send_counts(p, 0);
    for (index_t sa : SA)
    {
        int target_p = part.target_processor(sa);
        assert(0 <= target_p && target_p < p);
        ++send_counts[target_p];
    }
    std::vector<int> send_displs = get_displacements(send_counts);
    std::vector<index_t> send_SA(SA.size());
    std::vector<index_t> send_B(local_B.size());
    // Reorder the SA and B arrays into buckets, one for each target processor.
    // The target processor is given by the value in the SA.
    for (std::size_t i = 0; i < SA.size(); ++i)
    {
        int target_p = part.target_processor(SA[i]);
        assert(target_p < p && target_p >= 0);
        std::size_t out_idx = send_displs[target_p]++;
        assert(out_idx < SA.size());
        send_SA[out_idx] = SA[i];
        send_B[out_idx] = local_B[i];
    }
    SAC_TIMER_END_SECTION("sa2isa_bucketing");

    // get displacements again (since they were modified above)
    send_displs = get_displacements(send_counts);
    // get receive information
    std::vector<int> recv_counts = all2allv_get_recv_counts(send_counts, comm);
    std::vector<int> recv_displs = get_displacements(recv_counts);

    // perform the all2all communication
    MPI_Alltoallv(&send_B[0], &send_counts[0], &send_displs[0], mpi_index_t,
                  &local_B[0], &recv_counts[0], &recv_displs[0], mpi_index_t,
                  comm);
    MPI_Alltoallv(&send_SA[0], &send_counts[0], &send_displs[0], mpi_index_t,
                  &SA[0], &recv_counts[0], &recv_displs[0], mpi_index_t,
                  comm);
    SAC_TIMER_END_SECTION("sa2isa_all2all");

    // rearrange locally
    // TODO [ENH]: more cache efficient by sorting rather than random assignment
    for (std::size_t i = 0; i < SA.size(); ++i)
    {
        index_t out_idx = SA[i] - part.excl_prefix_size();
        assert(0 <= out_idx && out_idx < SA.size());
        send_B[out_idx] = local_B[i];
    }

    // output is now in send_B -> swap vectors
    local_B.swap(send_B);

    // reassign the SA
    std::size_t global_offset = part.excl_prefix_size();
    for (std::size_t i = 0; i < SA.size(); ++i)
    {
        SA[i] = global_offset + i;
    }
    SAC_TIMER_END_SECTION("sa2isa_rearrange");
}

void reorder_sa_to_isa()
{
    reorder_sa_to_isa(local_SA);
}

};


#endif // SUFFIX_ARRAY_HPP





