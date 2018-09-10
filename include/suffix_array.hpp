/*
 * Copyright 2015 Georgia Institute of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SUFFIX_ARRAY_HPP
#define SUFFIX_ARRAY_HPP

#include <mpi.h>
#include <vector>
#include <cstring> // memcmp

#include "alphabet.hpp"
#include "kmer.hpp"
#include "par_rmq.hpp"
#include "shifting.hpp"
#include "bucketing.hpp"
#include "stringset.hpp"
#include "bulk_permute.hpp"
#include "bulk_rma.hpp"
#include "idxsort.hpp"

#include <mxx/datatypes.hpp>
#include <mxx/shift.hpp>
#include <mxx/partition.hpp>
#include <mxx/sort.hpp>
#include <mxx/collective.hpp>
#include <mxx/timer.hpp>

#include <prettyprint.hpp>


/*********************************************************************
 *              Macros for timing sections in the code               *
 *********************************************************************/

// TODO: use a proper logging engine!
#define INFO(msg) {std::cerr << msg << std::endl;}
//#define INFO(msg) {}

#define SAC_ENABLE_TIMER 1
#if SAC_ENABLE_TIMER
#define SAC_TIMER_START() mxx::section_timer timer(std::cerr, this->comm);
#define SAC_TIMER_END_SECTION(str) timer.end_section(str);
#define SAC_TIMER_LOOP_START() mxx::section_timer looptimer(std::cerr, this->comm);
#define SAC_TIMER_END_LOOP_SECTION(iter, str) looptimer.end_section(str);
#else
#define SAC_TIMER_START()
#define SAC_TIMER_END_SECTION(str)
#define SAC_TIMER_LOOP_START()
#define SAC_TIMER_END_LOOP_SECTION(iter, str)
#endif

template <typename T>
struct TwoBSA {
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

template <typename T>
std::ostream& operator<<(std::ostream& os, const TwoBSA<T>& t){
    return os << "{" << t.SA << "," << t.B1 << "," << t.B2 << "}";
}


// specialize MPI datatype (mxx)
namespace mxx {
template <typename T>
class datatype_builder<TwoBSA<T> > : public datatype_contiguous<T, 3> {};
}


// pair of two same element
template <typename T>
struct mypair
{
    T first;
    T second;
};

// partial template specialization for mypair
namespace mxx {
template <typename T>
class datatype_builder<mypair<T> > : public datatype_contiguous<T, 2> {};
}



// distributed suffix array
template <typename char_t, typename index_t = std::size_t, bool _CONSTRUCT_LCP = false>
class suffix_array {
private:
public:
    suffix_array(const mxx::comm& _comm) : comm(_comm.copy()) {
    }
    /*
    suffix_array(InputIterator begin, InputIterator end, const mxx::comm& _comm)
        : comm(_comm.copy()), p(comm.size()),
        input_begin(begin), input_end(end)
    {
        // the local size of the input
        local_size = std::distance(begin, end);
        n = mxx::allreduce(local_size, this->comm);
        // get distribution
        part = mxx::blk_dist(n, comm.size(), comm.rank());

        // assert a block decomposition
        if (part.local_size() != local_size)
            throw std::runtime_error("The input string must be equally block decomposed accross all MPI processes.");
    }
    */
    virtual ~suffix_array() {}

private:
    /// The global size of the input string and suffix array
    std::size_t n;

    /// The local size of the input string and suffix array
    /// is either floor(n/p) or ceil(n/p) and based on a equal block
    /// distribution
    std::size_t local_size;

    /// The MPI communicator to use for the parallel suffix array construction
    mxx::comm comm;

    /// number of processes = size of the communicator
    int p;


public:

    // The block decomposition for the suffix array
    mxx::blk_dist part;

    /// Iterators over the local input string
    //InputIterator input_begin;
    /// End iterator for local input string
    //InputIterator input_end;

    //using char_type = typename std::iterator_traits<InputIterator>::value_type;
    using char_type = char_t;
    using alphabet_type = alphabet<char_type>;
    alphabet_type alpha;

public:
    /// The local suffix array
    std::vector<index_t> local_SA;
    /// The local inverse suffix array (TODO: rename?)
    std::vector<index_t> local_B;
    /// The local LCP array (remains empty if no LCP is constructed)
    std::vector<index_t> local_LCP;

private:

    // MPI tags used in constructing the suffix array
    static const int PSAC_TAG_SHIFT = 2;

public:

void init_size(size_t lsize) {
    local_size = lsize;
    n = mxx::allreduce(local_size, this->comm);
    // get distribution
    part = mxx::blk_dist(n, comm.size(), comm.rank());

    p = comm.size();

    // assert a block decomposition
    if (part.local_size() != local_size)
        throw std::runtime_error("The input string must be equally block decomposed accross all MPI processes.");
}

//template <typename StringSet> TODO template later
void construct_ss(simple_dstringset& ss, const alphabet_type& alpha) {
    SAC_TIMER_START();
    /***********************
     *  Initial bucketing  *
     ***********************/
    unsigned int k = get_optimal_k<index_t>(alpha, ss.sum_sizes, comm);
    if(comm.rank() == 0) {
        INFO("Alphabet: " << alpha);
    }
    SAC_TIMER_END_SECTION("alphabet detection");

    // create distributed sequences helper structure for the distributed stringset
    dist_seqs ds = dist_seqs::from_dss(ss, comm);

    // create all kmers
    local_B = kmer_gen_stringset<index_t>(ss, k, alpha, comm);
    // equally distribute kmers
    mxx::stable_distribute_inplace(local_B, comm);
    init_size(local_B.size());
    SAC_TIMER_END_SECTION("kmer generation");

    size_t shift_by;
    std::vector<index_t> local_B_SA;
    size_t unfinished_buckets, unfinished_elements;
    for (shift_by = k; shift_by < n; shift_by <<= 1) {
        SAC_TIMER_LOOP_START();

        // 1) doubling by shifting into tuples (2BSA kind of structure)
        std::vector<index_t> B2 = shift_buckets_ds(ds, local_B, shift_by, comm);
        SAC_TIMER_END_LOOP_SECTION(shift_by, "shift-buckets-ds");

        // 2) sort by (B1, B2)
        local_SA = idxsort_vectors<index_t, index_t, true>(local_B, B2, comm);
        SAC_TIMER_END_LOOP_SECTION(shift_by, "SA2ISA-idxsort");

        // 4) rebucket (B1, B2) -> B1 and LCP contruction
        if (shift_by == k) {
            if (_CONSTRUCT_LCP) {
                initial_kmer_lcp_gsa(k, alpha.bits_per_char(), B2);
                SAC_TIMER_END_LOOP_SECTION(shift_by, "init-lcp");
            }
            std::tie(unfinished_buckets, unfinished_elements) = rebucket_gsa_kmers(local_B, B2, true, comm, alpha.bits_per_char());
        } else {
            if (_CONSTRUCT_LCP) {
                resolve_next_lcp(shift_by, B2);
                SAC_TIMER_END_LOOP_SECTION(shift_by, "resove-lcp");
            }
            std::tie(unfinished_buckets, unfinished_elements) = rebucket_gsa(local_B, B2, true, comm);
        }
        if (comm.rank() == 0) {
            INFO("iteration " << shift_by << ": unfinished buckets = " << unfinished_buckets << ", unfinished elements = " << unfinished_elements);
        }

        // 5) reverse order to SA order
        if ((shift_by << 1) >= n || unfinished_buckets == 0) {
            // if last iteration, use copy of local_SA for reorder and keep
            // original SA
            std::vector<index_t> cpy_SA(local_SA);
            bulk_permute_inplace(local_B, cpy_SA, part, comm);
            SAC_TIMER_END_LOOP_SECTION(shift_by, "SA2ISA-bulk-permute");
        //} else if (unfinished_elements < n/10) {
        } else if (true) {
            // switch to A2: bucket chaising
            // prepare for bucket chaising (needs SA, and bucket arrays in both
            // SA and ISA order)
            std::vector<index_t> cpy_SA(local_SA);
            local_B_SA = local_B; // copy
            bulk_permute_inplace(local_B, cpy_SA, part, comm);
            SAC_TIMER_END_LOOP_SECTION(shift_by, "SA2ISA-bulk-permute");
            break;
        } else {
            bulk_permute_inplace(local_B, local_SA, part, comm);
            SAC_TIMER_END_LOOP_SECTION(shift_by, "SA2ISA-bulk-permute");
            //SAC_TIMER_END_LOOP_SECTION(shift_by, "SA-to-ISA");
        }
        // end iteratior
        SAC_TIMER_END_SECTION("sac-iteration");

        if (unfinished_buckets == 0)
            break;
    }

    if (unfinished_buckets > 0) {
        if (comm.rank() == 0)
            INFO("Starting Bucket chasing algorithm");
        construct_msgs_gsa(ds, local_B_SA, local_B, 2*shift_by);
    }

    SAC_TIMER_END_SECTION("construct-msgs");
    for (std::size_t i = 0; i < local_B.size(); ++i) {
        // the buffer indeces are `1` based indeces, but the ISA should be
        // `0` based indeces
        local_B[i] -= 1;
    }
}

template <typename Iterator>
void construct(Iterator begin, Iterator end, bool fast_resolval, const alphabet_type& alpha, unsigned int k) {
    SAC_TIMER_START();
    // create initial k-mers and use these as the initial bucket numbers
    // for each character position
    local_B = kmer_generation<index_t>(begin, end, k, alpha, comm);
    SAC_TIMER_END_SECTION("kmer-gen");

    std::vector<index_t> local_B_SA;
    std::size_t unfinished_buckets = 1<<k;
    std::size_t unfinished_elements = n;
    std::size_t shift_by;

    /*******************************
     *  Prefix Doubling main loop  *
     *******************************/
    for (shift_by = k; shift_by < n; shift_by <<= 1) {
        SAC_TIMER_LOOP_START();
        /**************************************************
         *  Pairing buckets by shifting `shift_by` = 2^i  *
         **************************************************/
        // shift the B1 buckets by 2^i to the left => equals B2
        std::vector<index_t> local_B2 = shift_vector(local_B, part, shift_by, comm);
        SAC_TIMER_END_LOOP_SECTION(shift_by, "shift-buckets");

        /*************
         *  ISA->SA  *
         *************/
        // by using sample sort on tuples (B1,B2)
        local_SA = idxsort_vectors<index_t, index_t>(local_B, local_B2, comm);
        SAC_TIMER_END_LOOP_SECTION(shift_by, "ISA-to-SA");

        /****************
         *  Update LCP  *
         ****************/
        // if this is the first iteration: create LCP, otherwise update
        if (_CONSTRUCT_LCP) {
            if (shift_by == k) {
                initial_kmer_lcp(k, alpha.bits_per_char(), local_B2);
                SAC_TIMER_END_LOOP_SECTION(shift_by, "init-lcp");
            } else {
                resolve_next_lcp(shift_by, local_B2);
                SAC_TIMER_END_LOOP_SECTION(shift_by, "update-lcp");
            }
        }

        /*******************************
         *  Assign new bucket numbers  *
         *******************************/
        std::tie(unfinished_buckets, unfinished_elements) = rebucket(local_B, local_B2, true, comm);
        if (comm.rank() == 0) {
            INFO("iteration " << shift_by << ": unfinished buckets = " << unfinished_buckets << ", unfinished elements = " << unfinished_elements);
        }
        SAC_TIMER_END_LOOP_SECTION(shift_by, "rebucket");

        /*************
         *  SA->ISA  *
         *************/
        // by bucketing to correct target processor using the `SA` array
        if (fast_resolval && unfinished_elements < n/10) {
            // prepare for bucket chaising (needs SA, and bucket arrays in both
            // SA and ISA order)
            std::vector<index_t> cpy_SA(local_SA);
            local_B_SA = local_B; // copy
            bulk_permute_inplace(local_B, cpy_SA, part, comm);
            SAC_TIMER_END_LOOP_SECTION(shift_by, "SA-to-ISA");
            SAC_TIMER_END_SECTION("sac-iteration");
            break;
        } else if ((shift_by << 1) >= n || unfinished_buckets == 0) {
            // if last iteration, use copy of local_SA for reorder and keep
            // original SA
            std::vector<index_t> cpy_SA(local_SA);
            bulk_permute_inplace(local_B, cpy_SA, part, comm);
            SAC_TIMER_END_LOOP_SECTION(shift_by, "SA-to-ISA");
        } else {
            bulk_permute_inplace(local_B, local_SA, part, comm);
            SAC_TIMER_END_LOOP_SECTION(shift_by, "SA-to-ISA");
        }

        // end iteratior
        SAC_TIMER_END_SECTION("sac-iteration");

        // check for termination condition
        if (unfinished_buckets == 0)
            break;
    }

    if (unfinished_buckets > 0) {
        if (comm.rank() == 0)
            INFO("Starting Bucket chasing algorithm");
        construct_msgs(local_B_SA, local_B, 2*shift_by);
    }
    SAC_TIMER_END_SECTION("construct-msgs");

    // now local_SA is actual block decomposed SA and local_B is actual ISA with an offset of one
    for (std::size_t i = 0; i < local_B.size(); ++i) {
        // the buffer indeces are `1` based indeces, but the ISA should be
        // `0` based indeces
        local_B[i] -= 1;
    }
    SAC_TIMER_END_SECTION("fix-isa");
}


template <typename Iterator>
void construct(Iterator begin, Iterator end, bool fast_resolval = true, unsigned int k = 0) {

    SAC_TIMER_START();
    /* get sizes */
    // the local size of the input
    init_size(std::distance(begin, end));

    /***********************
     *  Initial bucketing  *
     ***********************/

    // detect alphabet and get encoding
    alpha = alphabet_type::from_sequence(begin, end, comm);
    k = get_optimal_k<index_t>(alpha, local_size, comm, k);
    if(comm.rank() == 0) {
        INFO("Alphabet: " << alpha);
    }

    construct(begin, end, fast_resolval, alpha, k);
}

// generalized to more than "doubling" (e.g. prefix-trippling with L=3)
// NOTE: this implementation doesn't support building the LCP (SA + ISA only)
template <std::size_t L, typename Iterator>
void construct_arr(Iterator begin, Iterator end, bool fast_resolval = true) {
    SAC_TIMER_START();
    init_size(std::distance(begin, end));

    /***********************
     *  Initial bucketing  *
     ***********************/

    // detect alphabet and get encoding
    alpha = alphabet_type::from_sequence(begin, end, comm);
    unsigned int k = get_optimal_k<index_t>(alpha, local_size, comm);
    if(comm.rank() == 0) {
        INFO("Alphabet: " << alpha.unique_chars());
        INFO("Detecting sigma=" << alpha.sigma() << " => l=" << alpha.bits_per_char() << ", k=" << k);
    }

    // create initial k-mers and use these as the initial bucket numbers
    // for each character position
    local_B = kmer_generation<index_t>(begin, end, k, alpha, comm);
    SAC_TIMER_END_SECTION("initial-bucketing");


    std::vector<index_t> local_B_SA;
    std::size_t unfinished_buckets = 1<<k;
    std::size_t unfinished_elements = n;
    std::size_t shift_by;

    /*******************************
     *  Prefix Doubling main loop  *
     *******************************/
    for (shift_by = k; shift_by < n; shift_by*=L) {
        SAC_TIMER_LOOP_START();

        /*****************
         *  fill tuples  *
         *****************/
        std::vector<std::array<index_t, L+1> > tuples(local_size);
        std::size_t offset = part.eprefix_size();
        for (std::size_t i = 0; i < local_size; ++i) {
            tuples[i][0] = i + offset;
            tuples[i][1] = local_B[i];
        }
        SAC_TIMER_END_LOOP_SECTION(shift_by, "arr-tupelize");

        /**************************************************
         *  Pairing buckets by shifting `shift_by` = 2^k  *
         **************************************************/
        // shift the B1 buckets by 2^k to the left => equals B2
        multi_shift_inplace<index_t, L>(tuples, part, shift_by, comm);
        SAC_TIMER_END_LOOP_SECTION(shift_by, "shift-buckets");


        /*************
         *  ISA->SA  *
         *************/
        // by using sample sort on tuples (B1,B2)
        sort_array_tuples<L>(tuples);
        SAC_TIMER_END_LOOP_SECTION(shift_by, "ISA-to-SA");


        /****************
         *  Update LCP  *
         ****************/
        // if this is the first iteration: create LCP, otherwise update
        // TODO: LCP construciton is not (yet) implemented for std::array based construction
        /*
        if (_CONSTRUCT_LCP)
        {
            if (shift_by == k) {
                initial_kmer_lcp(k, bits_per_char, local_B2);
                SAC_TIMER_END_LOOP_SECTION(shift_by, "init-lcp");
            } else {
                resolve_next_lcp(shift_by, local_B2);
                SAC_TIMER_END_LOOP_SECTION(shift_by, "update-lcp");
            }
        }
        */

        /*******************************
         *  Assign new bucket numbers  *
         *******************************/
        std::tie(unfinished_buckets,unfinished_elements) = rebucket_arr<L>(tuples, local_B, true, comm);
        if (comm.rank() == 0) {
            INFO("iteration " << shift_by << ": unfinished buckets = " << unfinished_buckets << ", unfinished elements = " << unfinished_elements);
        }
        SAC_TIMER_END_LOOP_SECTION(shift_by, "rebucket");


        /**************************************
         *  Reset local_SA array from tuples  *
         **************************************/

        // init local_SA
        local_SA.resize(local_size);
        for (std::size_t i = 0; i < local_size; ++i) {
            local_SA[i] = tuples[i][0];
        }

        // deallocate all memory
        tuples.clear();
        tuples.shrink_to_fit();
        SAC_TIMER_END_LOOP_SECTION(shift_by, "arr-untupelize");


        /*************
         *  SA->ISA  *
         *************/
        // by bucketing to correct target processor using the `SA` array

        if (fast_resolval && unfinished_elements < n/10) {
            // prepare for bucket chaising (needs SA, and bucket arrays in both
            // SA and ISA order)
            std::vector<index_t> cpy_SA(local_SA);
            local_B_SA = local_B; // copy
            bulk_permute_inplace(local_B, cpy_SA, part, comm);
            SAC_TIMER_END_LOOP_SECTION(shift_by, "SA-to-ISA");
            break;
        } else if ((shift_by * L) >= n || unfinished_buckets == 0) {
            // if last iteration, use copy of local_SA for reorder and keep
            // original SA
            std::vector<index_t> cpy_SA(local_SA);
            bulk_permute_inplace(local_B, cpy_SA, part, comm);
            SAC_TIMER_END_LOOP_SECTION(shift_by, "SA-to-ISA");
        } else {
            bulk_permute_inplace(local_B, local_SA, part, comm);
            SAC_TIMER_END_LOOP_SECTION(shift_by, "SA-to-ISA");
        }

        // end iteratior
        SAC_TIMER_END_SECTION("sac-iteration");

        // check for termination condition
        if (unfinished_buckets == 0)
            break;
    }

    if (unfinished_buckets > 0) {
        if (comm.rank() == 0)
            INFO("Starting Bucket chasing algorithm");
        construct_msgs(local_B_SA, local_B, L*shift_by);
    }
    SAC_TIMER_END_SECTION("construct-msgs");

    // now local_SA is actual block decomposed SA and local_B is actual ISA with an offset of one
    for (std::size_t i = 0; i < local_B.size(); ++i) {
        // the buffer indeces are `1` based indeces, but the ISA should be
        // `0` based indeces
        local_B[i] -= 1;
    }
    SAC_TIMER_END_SECTION("fix-isa");
}

#if 0
void construct_fast() {
    SAC_TIMER_START();

    /***********************
     *  Initial bucketing  *
     ***********************/

    // detect alphabet and get encoding
    alpha = alphabet_type::from_sequence(input_begin, input_end, comm);
    unsigned int bits_per_char = alpha.bits_per_char();
    unsigned int k = get_optimal_k<index_t>(alpha, local_size, comm);
    if(comm.rank() == 0) {
        INFO("Alphabet: " << alpha.unique_chars());
        INFO("Detecting sigma=" << alpha.sigma() << " => l=" << bits_per_char << ", k=" << k);
    }

    // create initial k-mers and use these as the initial bucket numbers
    // for each character position
    local_B = kmer_generation<index_t>(input_begin, input_end, k, alpha, comm);
    SAC_TIMER_END_SECTION("initial-bucketing");

    // init local_SA
    if (local_SA.size() != local_B.size()) {
        local_SA.resize(local_B.size());
    }

    kmer_sorting();
    SAC_TIMER_END_SECTION("kmer-sorting");

    if (_CONSTRUCT_LCP) {
        initial_kmer_lcp(k, bits_per_char);
        SAC_TIMER_END_SECTION("initial-kmer-lcp");
    }

    rebucket_kmer();
    SAC_TIMER_END_SECTION("rebucket-kmer");

    std::vector<index_t> cpy_SA(local_SA);
    std::vector<index_t> local_B_SA(local_B); // copy
    bulk_permute_inplace(local_B, cpy_SA, part, comm);
    SAC_TIMER_END_SECTION("sa2isa");

    cpy_SA.clear();
    cpy_SA.shrink_to_fit();

    if (comm.rank() == 0)
        INFO("Starting Bucket chasing algorithm");
    construct_msgs(local_B_SA, local_B, k);
    SAC_TIMER_END_SECTION("construct-msgs");

    // now local_SA is actual block decomposed SA and local_B is actual ISA with an offset of one
    for (std::size_t i = 0; i < local_B.size(); ++i) {
        // the buffer indeces are `1` based indeces, but the ISA should be
        // `0` based indeces
        local_B[i] -= 1;
    }
    SAC_TIMER_END_SECTION("fix-isa");
}

#endif


private:



/*********************************************************************
 *                     ISA -> SA (sort buckets)                      *
 *********************************************************************/


template <std::size_t L>
void sort_array_tuples(std::vector<std::array<index_t, L+1> >& tuples) {
    assert(tuples.size() == local_size);
    SAC_TIMER_START();

    // parallel, distributed sample-sorting of tuples (B1, B2, SA)
    mxx::sort(tuples.begin(), tuples.end(),
    [] (const std::array<index_t, L+1>& x, const std::array<index_t, L+1>& y) {
        for (unsigned int i = 1; i < L+1; ++i) {
            if (x[i] != y[i])
                return x[i] < y[i];
        }
        return false;
    }, comm);

    SAC_TIMER_END_SECTION("isa2sa_samplesort");
}

void kmer_sorting() {
    SAC_TIMER_START();

    // initialize tuple array
    std::vector<mypair<index_t> > tuple_vec(local_size);

    // get global index offset
    std::size_t str_offset = part.eprefix_size();

    // fill tuple vector
    for (std::size_t i = 0; i < local_size; ++i) {
        tuple_vec[i].first = local_B[i];
        assert(str_offset + i < std::numeric_limits<index_t>::max());
        tuple_vec[i].second = str_offset + i;
    }

    // release memory of input (to remain at the minimum 6x words memory usage)
    local_B.clear(); local_B.shrink_to_fit();
    local_SA.clear(); local_SA.shrink_to_fit();

    SAC_TIMER_END_SECTION("isa2sa_pairize");

    // parallel, distributed sample-sorting of tuples (B1, B2, SA)
    mxx::sort(tuple_vec.begin(), tuple_vec.end(),
              [](const mypair<index_t>& x, const mypair<index_t>& y) {
                    return x.first < y.first;
              }, comm);

    SAC_TIMER_END_SECTION("isa2sa_samplesort_pairs");

    // reallocate output
    local_B.resize(local_size);
    local_SA.resize(local_size);

    // read back into input vectors
    for (std::size_t i = 0; i < local_size; ++i) {
        local_B[i] = tuple_vec[i].first;
        local_SA[i] = tuple_vec[i].second;
    }
    SAC_TIMER_END_SECTION("isa2sa_unpairize");
}


/*********************************************************************
 *              Rebucket tuples into new bucket numbers              *
 *********************************************************************/

// assumed sorted order (globally) by local_B
// this reassigns new, unique bucket numbers in {1,...,n} globally
void rebucket_kmer() {
    /*
     * NOTE: buckets are indexed by the global index of the first element in
     *       the bucket with a ONE-BASED-INDEX (since bucket number `0` is
     *       reserved for out-of-bounds)
     */


    // get my global starting index
    size_t prefix = part.eprefix_size();
    size_t local_max = 0;

    /*
     * assign local zero or one, depending on whether the bucket is the same
     * as the previous one
     */
    foreach_pair(local_B.begin(), local_B.end(), [&](index_t prev, index_t& cur, size_t i) {
        if (prev == cur) {
            cur = prefix + i + 1;
            local_max = cur;
        } else {
            cur = 0;
        }
    }, comm);
    if (comm.rank() == 0) {
        local_B[0] = 1;
        if (local_max == 0)
            local_max = 1;
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

    // 3.) linear scan and assign bucket numbers
    for (std::size_t i = 0; i < local_B.size(); ++i) {
        if (local_B[i] == 0)
            local_B[i] = pre_max;
        else
            pre_max = local_B[i];
        assert(local_B[i] <= i+prefix+1);
        // first element of bucket has id of it's own global index:
        assert(i == 0 || (local_B[i-1] ==  local_B[i] || local_B[i] == i+prefix+1));
    }
}

// same function as before, but this one assumes tuples instead of
// two arrays
// This is used in the bucket chaising construction. The MPI_Comm will most
// of the time be a subcommunicator (so do not use the member `comm`)
template <typename EqualFunc>
void rebucket_bucket(std::vector<TwoBSA<index_t> >& bucket, const mxx::comm& comm, std::size_t bucket_offset, std::vector<std::tuple<index_t, index_t, index_t> >& minqueries, size_t shift_by, EqualFunc eq) {
    /*
     * NOTE: buckets are indexed by the global index of the first element in
     *       the bucket with a ONE-BASED-INDEX (since bucket number `0` is
     *       reserved for out-of-bounds)
     */
    // inputs can be of different size, since buckets can span bucket boundaries
    std::size_t local_size = bucket.size();
    std::size_t prefix = mxx::exscan(local_size, std::plus<size_t>(), comm);
    size_t local_max = 0;
    prefix += bucket_offset;

    // iterate through all pairs of elements (spanning across processors)
    foreach_pair(bucket.begin(), bucket.end(), [&](const TwoBSA<index_t>& prev, TwoBSA<index_t>& cur, size_t i){
        // if this is a new bucket boundary: set current to prefix+index and update LCP
        if (!eq(prev, cur)) {
            cur.B1 = prefix+i+1;
            local_max = cur.B1;
        } else {
            cur.B1 = 0;
        }
        if (_CONSTRUCT_LCP) {
            if (prev.B2 == 0 || cur.B2 == 0) {
                if (local_LCP[i+prefix - part.eprefix_size()] == n)
                    local_LCP[i+prefix - part.eprefix_size()] = shift_by;
            } else if (prev.B2 != cur.B2) {
                index_t left_b  = std::min(prev.B2, cur.B2);
                index_t right_b = std::max(prev.B2, cur.B2);
                assert(0 < left_b && left_b < n);
                assert(0 < right_b && right_b <= n);
                minqueries.emplace_back(i + prefix, left_b, right_b);
            }
        }
    }, comm);

    // specially handle first element of first process
    if (comm.rank() == 0) {
        bucket[0].B1 = prefix + 1;
        if (local_max == 0)
            local_max = prefix + 1;
    }

    /*
     * Global prefix MAX:
     *  - such that for every item we have it's bucket number, where the
     *    bucket number is equal to the first index in the bucket
     *    this way buckets who are finished, will never receive a new
     *    number.
     */
    // 2.) distributed scan with max() to get starting max for each sequence
    std::size_t pre_max = mxx::exscan(local_max, mxx::max<size_t>(), comm);
    if (comm.rank() == 0)
        pre_max = 0;

    // 3.) linear scan and assign bucket numbers
    for (std::size_t i = 0; i < local_size; ++i) {
        if (bucket[i].B1 == 0)
            bucket[i].B1 = pre_max;
        else
            pre_max = bucket[i].B1;
        assert(bucket[i].B1 <= i+prefix+1);
        // first element of bucket has id of it's own global index:
        assert(i == 0 || (bucket[i-1].B1 ==  bucket[i].B1 || bucket[i].B1 == i+prefix+1));
    }
}

void rebucket_bucket(std::vector<TwoBSA<index_t> >& bucket, const mxx::comm& comm, std::size_t gl_offset, std::vector<std::tuple<index_t, index_t, index_t> >& minqueries, size_t shift_by) {
    rebucket_bucket(bucket, comm, gl_offset, minqueries, shift_by, [](const TwoBSA<index_t>& x, const TwoBSA<index_t>& y){
        return x.B1 == y.B1 && x.B2 == y.B2;
    });
}

void rebucket_bucket_gsa(std::vector<TwoBSA<index_t> >& bucket, const mxx::comm& comm, std::size_t gl_offset, std::vector<std::tuple<index_t, index_t, index_t> >& minqueries, size_t shift_by) {
    rebucket_bucket(bucket, comm, gl_offset, minqueries, shift_by, [](const TwoBSA<index_t>& x, const TwoBSA<index_t>& y){
        return x.B1 == y.B1 && x.B2 == y.B2 && x.B2 != 0;
    });
}

/*********************************************************************
 *          Faster construction for fewer remaining buckets          *
 *********************************************************************/

std::vector<index_t> get_active(const std::vector<index_t>& B, const std::vector<index_t>& active, const mxx::comm& comm, bool print_stats = false) {
    // get next element from right
    index_t right_B = mxx::left_shift(B[0], comm);
    // get global offset
    size_t prefix = part.eprefix_size();

    size_t unresolved_els = 0;
    size_t unfinished_b = 0;

    std::vector<index_t> active_elements;
    bool use_b = active.empty();
    size_t loopn = use_b ? B.size() : active.size();
    for (index_t g = 0; g < loopn; ++g) {
        size_t j = use_b ? g : active[g];
        // get global index for each local index
        size_t i =  prefix + j;
        // check if this is a unresolved bucket
        // relying on the property that for resolved buckets:
        //   B[i] == i+1 and B[i+1] == i+2
        //   (where `i' is the global index)
        if (B[j] != i+1) {
            // save local active indexes
            active_elements.push_back(j);
            unresolved_els++;
        } else if (B[j] == i+1 && ((j < local_size-1 && B[j+1] == i+1)
                        || (j == local_size-1 && comm.rank() < p-1 && right_B == i+1))) {
            active_elements.push_back(j);
            unresolved_els++;
            unfinished_b++;
        }
    }

    if (print_stats) {
        size_t gl_unresolved = mxx::allreduce(unresolved_els, comm);
        size_t gl_unfinished = mxx::allreduce(unfinished_b, comm);
        if (comm.rank() == 0) {
            INFO("unresolved = " << gl_unresolved << ", unfinished = " << gl_unfinished);
        }
    }
    return active_elements;
}

std::vector<index_t> get_active(const std::vector<index_t>& B, const mxx::comm& comm, bool print_stats = false) {
    std::vector<index_t> empty;
    return get_active(B, empty, comm, print_stats);
}

std::vector<index_t> sparse_get_b2(const std::vector<index_t>& active, const std::vector<index_t>& B, const std::vector<index_t>& SA, size_t shift_by, const mxx::comm& comm) {
    std::vector<size_t> rma_reqs;
    std::vector<index_t> b2(active.size());
    for (size_t ai = 0; ai < active.size(); ++ai) {
        size_t j = active[ai];
        if (SA[j] + shift_by < n) {
            rma_reqs.push_back(SA[j]+shift_by);
        }
    }

    // use bulk RMA to request the values of B at doubled (+shift_by) location for
    // each active suffix
    std::vector<index_t> rma_b2 = bulk_rma(B.begin(), B.end(), rma_reqs, comm);

    auto b2in = rma_b2.begin();
    for (size_t i = 0; i < active.size(); ++i) {
        size_t j = active[i];
        if (SA[j] + shift_by < n) {
            b2[i] = *b2in;
            ++b2in;
        }
    }

    return b2;
}

/*
std::vector<index_t> local_get_sparse_b2(const dist_seqs& ds, const std::vector<index_t>& B, const std::vector<size_t>& local_queries, size_t shift_by) {
    // argsort the local_queries
    std::vector<size_t> argsort(local_queries.size());
    std::iota(argsort.begin(), argsort.end(), 0);
    std::sort(argsort.begin(), argsort.end(), [&local_queries](size_t i, size_t j) { return local_queries[i] < local_queries[j]; });

    // scan local queries in argsorted order and save the B2 if the string size permits
    std::vector<index_t> results(local_queries.size(), 9999999999ull);
    // each query is {SA[j] + shift_by}
    size_t i = 0;
    // linear in number of local strings + number of queries
    ds.for_each_seq([&](size_t str_beg, size_t str_end) {
        while (i < local_queries.size() && local_queries[argsort[i]] < str_end) {
            size_t qi = argsort[i];
            if (local_queries[qi] - shift_by >= str_beg) {
                results[qi] = B[local_queries[qi] - part.eprefix_size()];
            } else {
                results[qi] = 0;
            }
            ++i;
        }
    });
    for (size_t i = 0; i < results.size(); ++i) {
        assert(results[i] <= n);
    }
    return results;
}

template <typename T>
std::vector<T> sparse_doubling(const dist_seqs& ds, const std::vector<T>& vec, const std::vector<size_t>& rma_reqs, size_t shift_by, const mxx::comm& comm) {
    size_t local_size = vec.size();
    size_t global_size = mxx::allreduce(local_size, comm);
    mxx::blk_dist part(global_size, comm.size(), comm.rank());

    std::vector<size_t> original_pos;
    std::vector<size_t> bucketed_rma;
    std::vector<size_t> send_counts = idxbucketing(rma_reqs, [&part](size_t gidx) { return part.rank_of(gidx); }, comm.size(), bucketed_rma, original_pos);

    std::vector<size_t> recv_counts = mxx::all2all(send_counts, comm);

    // send all queries via all2all
    std::vector<size_t> local_queries = mxx::all2allv(bucketed_rma, send_counts, recv_counts, comm);

    std::vector<index_t> results = local_get_sparse_b2(ds, vec, local_queries, shift_by);
    results = mxx::all2allv(results, recv_counts, send_counts, comm);

    std::vector<index_t> rma_b2 = permute(results, original_pos);
    return rma_b2;
}
*/

std::vector<index_t> sparse_get_b2(const dist_seqs& ds, const std::vector<index_t>& active, const std::vector<index_t>& B, const std::vector<index_t>& SA, size_t shift_by, const mxx::comm& comm) {
    // create RMA requests for the doubled positions
    std::vector<size_t> rma_reqs;
    for (size_t ai = 0; ai < active.size(); ++ai) {
        size_t j = active[ai];
        if (SA[j] + shift_by < n) {
            rma_reqs.push_back(SA[j]+shift_by);
        }
    }

    //std::vector<index_t> rma_b2 = sparse_doubling(ds, B, rma_reqs, shift_by, comm);
    std::vector<index_t> rma_b2 = bulk_query_ds(ds, B, rma_reqs, comm, [&](size_t gidx, size_t str_beg, size_t) {
            if (gidx - shift_by >= str_beg) {
                return B[gidx - part.eprefix_size()];
            } else {
                return static_cast<index_t>(0);
            }
    });

    auto b2in = rma_b2.begin();
    assert(rma_b2.size() == rma_reqs.size());
    std::vector<index_t> b2(active.size());
    for (size_t i = 0; i < active.size(); ++i) {
        size_t j = active[i];
        if (SA[j] + shift_by < n) {
            b2[i] = *b2in;
            assert(b2[i] <= n);
            ++b2in;
        }
    }
    return b2;
}


template <typename Func>
void construct_msgs(std::vector<index_t>& local_B, std::vector<index_t>& local_ISA, int dist, Func sparse_b2_func) {
    /*
     * Algorithm for few remaining buckets (more communication overhead per
     * element but sends only unfinished buckets -> less data in total if few
     * buckets remaining)
     *
     * INPUT:
     *  - SA in SA order
     *  - B in SA order
     *  - B in ISA order
     *  - dist: the current dist=2^k, gets doubled after every iteration
     *
     * ALGO:
     * 1.) on i:            send tuple (`to:` Sa[i]+2^k, `from:` i)
     * 2.) on SA[i]+2^k:    return tuple (`to:` i, ISA[SA[i]+2^k])
     * 3.) on i:            for each unfinished bucket:
     *                          sort by new bucket index (2-stage across
     *                          processor boundaries using MPI subcommunicators)
     *                          rebucket into `B`
     * 4.) on i:            send tuple (`to:` SA[i], B[i]) // update bucket numbers in ISA order
     * 5.) on SA[i]:        update ISA[SA[i]] to new B[i]
     *
     */

    SAC_TIMER_START();

    // get global offset
    size_t prefix = part.eprefix_size();

    // get active elements
    std::vector<index_t> active = get_active(local_B, comm, true);
    SAC_TIMER_END_SECTION("get active elements");


    for (index_t shift_by = dist; shift_by < n; shift_by <<= 1) {
        // check for termination
        if (mxx::allreduce(active.size()) == 0)
            // finished!
            break;

        std::vector<index_t> b2 = sparse_b2_func(active, local_ISA, local_SA, shift_by, comm);

        // prepare LCP queries vector
        std::vector<std::tuple<index_t, index_t, index_t> > minqueries;

        dist_seqs_buckets db = dist_seqs_buckets::from_func(local_B, comm);

        // for each inner bucket:
        size_t inner_beg, inner_end;
        std::tie(inner_beg, inner_end) = db.inner_seqs_range();
        if (inner_beg < inner_end) {
            // find position in `active` corresponding to first internal bucket
            size_t ai = 0;
            while (ai < active.size() && active[ai] < inner_beg-prefix)
                ++ai;
            assert(ai == 0 || ai == active.size() || local_B[active[ai]] != local_B[active[ai-1]]);

            // for each internal bucket
            while (ai < active.size() && active[ai] < inner_end-prefix) {
                // get global index of bucket begin
                size_t bucket_begin = active[ai]+prefix;
                assert(bucket_begin == local_B[active[ai]]-1);
                size_t bucket_offset = bucket_begin - prefix;
                size_t a_beg = ai;

                // find end of bucket
                size_t idx = active[ai];
                while (ai < active.size() && local_B[idx]-1 == bucket_begin) {
                    assert(idx == active[ai]);
                    ++ai; ++idx;
                }

                assert(ai > 0);
                size_t a_end = ai;
                size_t bucket_end = active[a_end-1] + prefix + 1;
                assert(bucket_end - bucket_begin == a_end - a_beg);
                size_t bucket_size = bucket_end - bucket_begin;

                // arg-sort bucket by b2
                std::vector<size_t> ais(bucket_size);
                std::iota(ais.begin(), ais.end(), a_beg);
                std::sort(ais.begin(), ais.end(), [&](index_t x, index_t y) {
                    return b2[x] < b2[y] || (b2[x] == 0 && b2[y] == 0 && local_SA[active[x]] < local_SA[active[y]]);
                });


                // local rebucket
                // save back into local_B, local_SA, etc
                index_t cur_b = bucket_begin + 1;
                size_t out_idx = bucket_offset;
                std::vector<index_t> bucket_SA(local_SA.begin() + bucket_offset, local_SA.begin() + bucket_offset + bucket_size);
                // assert previous bucket index is smaller
                assert(out_idx == 0 || local_B[out_idx-1] < cur_b);
                for (auto it = ais.begin(); it != ais.end(); ++it) {
                    if (it != ais.begin()) {
                        index_t pre_b2 = b2[*(it-1)];
                        index_t cur_b2 = b2[*it];
                        // if this is a new bucket, then update bucket number
                        if (pre_b2 != cur_b2 || cur_b2 == 0) {
                            cur_b = out_idx + prefix + 1;
                        }
                        if (_CONSTRUCT_LCP) {
                            if (pre_b2 == 0 || cur_b2 == 0) {
                                if (local_LCP[out_idx] == n)
                                    local_LCP[out_idx] = shift_by;
                            } else if (pre_b2 != cur_b2) {
                                // add as query item for LCP construction
                                index_t left_b  = std::min(pre_b2, cur_b2);
                                index_t right_b = std::max(pre_b2, cur_b2);
                                assert(0 < left_b && left_b < n);
                                assert(0 < right_b && right_b <= n);
                                minqueries.emplace_back(out_idx + prefix, left_b, right_b);
                            }
                        }
                    }
                    local_SA[out_idx] = bucket_SA[active[*it]-bucket_offset];
                    assert(local_B[out_idx] == bucket_begin + 1);
                    local_B[out_idx] = cur_b;
                    out_idx++;
                }
                // assert next bucket index is larger
                assert(out_idx == local_size || local_B[out_idx] == prefix+out_idx+1);
            }
        }

        /*
         * for each split bucket, exclusively in a subcommunicator
         */
        db.for_each_split_seq_2phase(comm, [&](size_t gbegin, size_t gend, const mxx::comm& subcomm) {
            if (active.size() == 0) {
                assert(gbegin == gend);
                return;
            }

            std::vector<TwoBSA<index_t> > border_bucket;
            size_t lbegin = std::max(gbegin, prefix) - prefix;
            size_t lend = std::min(gend, prefix+local_size) - prefix;
            // find `ai` for this bucket
            size_t ai = 0;
            while (active[ai] < lbegin)
                ++ai;
            assert(active[ai] == lbegin);
            for (size_t i = lbegin; i < lend; ++i) {
                assert(active[ai] == i);
                TwoBSA<index_t> tuple;
                //assert(msgit->second >= prefix && msgit->second < prefix+local_size);
                tuple.SA = local_SA[i];
                tuple.B1 = local_B[i];
                tuple.B2 = b2[ai];
                border_bucket.push_back(tuple);
                ++ai;
            }
            // sample sort the bucket with arbitrary distribution
            auto cmp = [](const TwoBSA<index_t>& x, const TwoBSA<index_t>&y) {
                return x.B2 < y.B2 || (x.B2 == 0 && y.B2 == 0 && x.SA < y.SA);
            };
            mxx::sort(border_bucket.begin(), border_bucket.end(), cmp, subcomm);

#ifndef NDEBUG
            index_t first_bucket = border_bucket[0].B1;
#endif
            size_t bucket_offset = lbegin;
            // rebucket with global offset of first -> in tuple form (also updates LCP)
            rebucket_bucket_gsa(border_bucket, subcomm, gbegin, minqueries, shift_by);
            // assert first bucket index remains the same
            assert(subcomm.rank() != 0 || first_bucket == border_bucket[0].B1);

            // save into full array (if this was left -> save to beginning)
            // (else, need offset of last)
            assert(bucket_offset == 0 || local_B[bucket_offset-1] < border_bucket[0].B1);
            assert(bucket_offset+border_bucket.size() <= local_size);
            for (size_t i = 0; i < border_bucket.size(); ++i) {
                local_SA[i+bucket_offset] = border_bucket[i].SA;
                local_B[i+bucket_offset] = border_bucket[i].B1;
            }
            assert(bucket_offset+border_bucket.size() == local_size || (local_B[bucket_offset+border_bucket.size()] > local_B[bucket_offset+border_bucket.size()-1]));
            assert(subcomm.rank() != 0 || local_B[bucket_offset] == bucket_offset+prefix+1);
        });

        /*
         * 4.1)   Update LCP
         */
        if (_CONSTRUCT_LCP) {
            // time LCP separately!
            SAC_TIMER_START();
            // get parallel-distributed RMQ for all queries, results are in `minqueries`
            bulk_rmq(n, local_LCP, minqueries, comm);

            // update the new LCP values:
            for (auto min_lcp : minqueries) {
                local_LCP[std::get<0>(min_lcp) - prefix] = shift_by + std::get<2>(min_lcp);
            }
            SAC_TIMER_END_SECTION("LCP update");
        }

        /*
         * 4.2)  Update ISA (TODO: formulate as bulk(sparse) write)
         */
        // message new bucket numbers to new SA[i] for all previously unfinished
        // buckets
        // since the message array is still available with the indices of unfinished
        // buckets -> reuse that information => no need to rescan the whole
        // local array
        std::vector<mypair<index_t> > msgs(active.size());
        for (size_t i = 0; i < active.size(); ++i) {
            size_t j = active[i];
            msgs[i].first = local_SA[j]; // SA[i]
            msgs[i].second = local_B[j]; // B[i]
        }

        // message exchange to processor which contains first index
        mxx::all2all_func(msgs, [&](const mypair<index_t>& x){return part.rank_of(x.first);}, comm);

        // update local ISA with new bucket numbers
        for (auto it = msgs.begin(); it != msgs.end(); ++it) {
            // TODO _mm_stream
            local_ISA[it->first-prefix] = it->second;
        }
        msgs = std::vector<mypair<index_t>>();

        // update remaining active elements
        active = get_active(local_B, active, comm, true);

        SAC_TIMER_END_SECTION("bucket-chaising iteration");
    }
}

void construct_msgs(std::vector<index_t>& local_B, std::vector<index_t>& local_ISA, int dist) {
    construct_msgs(local_B, local_ISA, dist,
            [&](const std::vector<index_t>& active, const std::vector<index_t>& B, const std::vector<index_t>& SA, size_t shift_by, const mxx::comm& comm) {
                return sparse_get_b2(active, B, SA, shift_by, comm);
            });
}

void construct_msgs_gsa(const dist_seqs& ds, std::vector<index_t>& local_B, std::vector<index_t>& local_ISA, int dist) {
    construct_msgs(local_B, local_ISA, dist,
            [&](const std::vector<index_t>& active, const std::vector<index_t>& B, const std::vector<index_t>& SA, size_t shift_by, const mxx::comm& comm) {
                return sparse_get_b2(ds, active, B, SA, shift_by, comm);
            });
}

/*********************************************************************
 *                         LCP construction                          *
 *********************************************************************/

// for a single bucket array (one k-mer)
void initial_kmer_lcp(unsigned int k, unsigned int bits_per_char) {
    //    get the LCP by getting position of first different bit and dividing by
    //    `bits_per_char`

    // resize to size `local_size` and set all items to max of n
    local_LCP.assign(local_size, n);

    // 1) getting next element to left
    index_t left_B = mxx::right_shift(local_B.back(), comm);

    // initialize first LCP
    if (comm.rank() == 0) {
        local_LCP[0] = 0;
    } else {
        if (left_B != local_B[0]) {
            local_LCP[0] = lcp_bitwise(left_B, local_B[0], k, bits_per_char);
        }
    }

    // intialize the LCP for all other elements for bucket boundaries
    for (std::size_t i = 1; i < local_size; ++i) {
        if (local_B[i-1] != local_B[i]) {
            local_LCP[i] = lcp_bitwise(local_B[i-1], local_B[i], k, bits_per_char);
        }
    }
}

template <typename T, typename Func>
void for_each_lpair_2vec(const std::vector<T>& v1, const std::vector<T>& v2, Func func, const mxx::comm& comm) {
    // assume both vectors are same size
    MXX_ASSERT(v1.size() == v2.size() && v1.size() > 0);

    // 1) getting next element to left
    std::pair<T, T> right_el(v1.back(), v2.back());
    std::pair<T, T> left_el = mxx::right_shift(right_el, comm);

    // call for first 2 pairs
    if (comm.rank() > 0) {
        func(left_el.first, left_el.second, v1.front(), v2.front(), 0);
    }

    for (size_t i = 1; i < v1.size(); ++i) {
        func(v1[i-1], v2[i-1], v1[i], v2[i], i);
    }
}

// for pairs of two buckets: pair[i] = (B1[i], B2[i])
void initial_kmer_lcp(unsigned int k, unsigned int bits_per_char,
                      const std::vector<index_t>& local_B2) {
    // for each bucket boundary (using both the B1 and the B2 buckets):
    //    get the LCP by getting position of first different bit and dividing by
    //    `bits_per_char`

    // resize to size `local_size` and set all items to max of n
    local_LCP.assign(local_size, n);
    if (comm.rank() == 0) {
        local_LCP[0] = 0;
    }

    for_each_lpair_2vec(local_B, local_B2, [k, bits_per_char, this](const index_t left1, const index_t left2, const index_t right1, const index_t right2, size_t i) {
        if (left1 != right1
            || (left1 == right1 && left2 != right2)) {
            unsigned int lcp = lcp_bitwise(left1, right1, k, bits_per_char);
            if (lcp == k)
                lcp += lcp_bitwise(left2, right2, k, bits_per_char);
            local_LCP[i] = lcp;
        }
    }, comm);
}


// for pairs of two buckets: pair[i] = (B1[i], B2[i])
void initial_kmer_lcp_gsa(unsigned int k, unsigned int bits_per_char,
                          const std::vector<index_t>& local_B2) {
    // for each bucket boundary (using both the B1 and the B2 buckets):
    //    get the LCP by getting position of first different bit and dividing by
    //    `bits_per_char`

    // resize to size `local_size` and set all items to max of n
    local_LCP.assign(local_size, n);
    if (comm.rank() == 0) {
        local_LCP[0] = 0;
    }

    for_each_lpair_2vec(local_B, local_B2, [k, bits_per_char, this](const index_t left1, const index_t left2, const index_t right1, const index_t right2, size_t i) {
        if (left1 != right1) {
            local_LCP[i] = lcp_bitwise(left1, right1, k, bits_per_char);
        } else {
            //unsigned int lcp = lcp_bitwise_no0(left1, right1, k, bits_per_char);
            assert(left1 != 0);
            unsigned int lcp = k - trailing_zeros(left1) / bits_per_char;
            if (lcp == k) {
                if(left2 != right2) {
                    lcp += lcp_bitwise(left2, right2, k, bits_per_char);
                    local_LCP[i] = lcp;
                } else {
                    if (left2 != 0) {
                        lcp += (k - trailing_zeros(left2) / bits_per_char);
                    }
                    if (lcp < 2*k) {
                        local_LCP[i] = lcp;
                    }
                }
            } else {
                // lcp is smaller than k => B1 has trailing 0
                local_LCP[i] = lcp;
            }
        }
    }, comm);
}


void resolve_next_lcp(int dist, const std::vector<index_t>& local_B2) {
    // 2.) find _new_ bucket boundaries (B1[i-1] == B1[i] && B2[i-1] != B2[i])
    // 3.) bulk-parallel-distributed RMQ for ranges (B2[i-1],B2[i]+1) to get min_lcp[i]
    // 4.) LCP[i] = dist + min_lcp[i]

    // find _new_ bucket boundaries and create associated parallel distributed
    // RMQ queries.
    std::vector<std::tuple<index_t, index_t, index_t> > minqueries;
    std::size_t prefix_size = part.eprefix_size();

    for_each_lpair_2vec(local_B, local_B2, [dist, prefix_size, &minqueries, this](const index_t left1, const index_t left2, const index_t right1, const index_t right2, size_t i) {
        if (left1 == right1) {
            if (left2 == 0 || right2 == 0) {
                if (local_LCP[i] == n)
                    local_LCP[i] = dist;
            } else if (left2 != right2) {
                index_t left_b  = std::min(left2, right2);
                index_t right_b = std::max(left2, right2);
                // we need the minumum LCP of all suffixes in buckets between
                // these two buckets. Since the first element in the left bucket
                // is the LCP of this bucket with its left bucket and we don't need
                // this LCP value, start one to the right:
                // (-1 each since buffer numbers are current index + 1)
                index_t range_left = (left_b-1) + 1;
                index_t range_right = (right_b-1) + 1; // +1 since exclusive index
                minqueries.emplace_back(i + prefix_size, range_left, range_right);
            }
        }
    }, comm);

#ifndef NDEBUG
    std::size_t nqueries = minqueries.size();
#endif

    // get parallel-distributed RMQ for all queries, results are in
    // `minqueries`
    // TODO: bulk updatable RMQs [such that we don't have to construct the
    //       RMQ for the local_LCP in each iteration]
    bulk_rmq(n, local_LCP, minqueries, comm);
    assert(minqueries.size() == nqueries);


    // update the new LCP values:
    for (auto min_lcp : minqueries) {
        local_LCP[std::get<0>(min_lcp) - prefix_size] = dist + std::get<2>(min_lcp);
    }
}

};


#endif // SUFFIX_ARRAY_HPP
