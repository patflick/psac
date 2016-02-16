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
#include "par_rmq.hpp"

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

template <typename T1, typename T2>
struct pair_sum {
    std::pair<T1,T2> operator()(const std::pair<T1,T2>& x, const std::pair<T1,T2>& y) {
        return std::pair<T1,T2>(x.first+y.first,x.second+y.second);
    }
};

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

template <typename InputIterator, typename index_t>
std::vector<index_t> alphabet_histogram(InputIterator begin, InputIterator end, const mxx::comm& comm) {
    static_assert(std::is_same<typename std::iterator_traits<InputIterator>::value_type, char>::value, "Iterator must be of value type `char`.");
    // get local histogram of alphabet characters
    std::vector<index_t> hist = get_histogram<index_t>(begin, end, 256);
    std::vector<index_t> out_hist = mxx::allreduce(hist, comm);
    return out_hist;
}

// distributed suffix array
template <typename InputIterator, typename index_t = std::size_t, bool _CONSTRUCT_LCP = false>
class suffix_array {
private:
public:
    suffix_array(InputIterator begin, InputIterator end, const mxx::comm& _comm)
        : comm(_comm.copy()), p(comm.size()),
        input_begin(begin), input_end(end)
    {
        // the local size of the input
        local_size = std::distance(begin, end);
        n = mxx::allreduce(local_size, this->comm);
        // get distribution
        part = mxx::partition::block_decomposition_buffered<index_t>(n, comm.size(), comm.rank());

        // assert a block decomposition
        if (part.local_size() != local_size)
            throw std::runtime_error("The input string must be equally block decomposed accross all MPI processes.");

        // get MPI type
        //mpi_index_t = index_mpi_dt.type();
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
    mxx::comm comm;

    /// number of processes = size of the communicator
    int p;
    /// The MPI datatype for the templated type `index_t`.
    //mxx::datatype<index_t> index_mpi_dt;
    //MPI_Datatype mpi_index_t;

    // The block decomposition for the suffix array
    mxx::partition::block_decomposition_buffered<index_t> part;

public:
    /// Iterators over the local input string
    InputIterator input_begin;
    /// End iterator for local input string
    InputIterator input_end;

public:
    /// mapping the ascii char to a "compressed" integer
    /// uses only ceillog(sigma+1) bits
    std::vector<uint16_t> alphabet_mapping;

    /// the number of unique characters in the input string
    /// (does not account for the special `0` character)
    unsigned int sigma;

    /// "compressed" integer to ascii


public: // TODO: make private again and provide some iterator and query access
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
void construct(bool fast_resolval = true, unsigned int k = 0) {
    SAC_TIMER_START();

    /***********************
     *  Initial bucketing  *
     ***********************/

    // create initial k-mers and use these as the initial bucket numbers
    // for each character position
    // `k` depends on the alphabet size and the word size of each suffix array
    // element. `k` is choosen to maximize the number of alphabet characters
    // that fit into one machine word
    unsigned int bits_per_char;
    std::tie(k, bits_per_char) = initial_bucketing(k);
    SAC_TIMER_END_SECTION("initial-bucketing");

    // init local_SA
    if (local_SA.size() != local_B.size()) {
        local_SA.resize(local_B.size());
    }

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
         *  Pairing buckets by shifting `shift_by` = 2^k  *
         **************************************************/
        // shift the B1 buckets by 2^k to the left => equals B2
        std::vector<index_t> local_B2;
        shift_buckets(shift_by, local_B2);
        SAC_TIMER_END_LOOP_SECTION(shift_by, "shift-buckets");

        /*************
         *  ISA->SA  *
         *************/
        // by using sample sort on tuples (B1,B2)
        isa_2b_to_sa(local_B2);
        SAC_TIMER_END_LOOP_SECTION(shift_by, "ISA-to-SA");

        /****************
         *  Update LCP  *
         ****************/
        // if this is the first iteration: create LCP, otherwise update
        if (_CONSTRUCT_LCP) {
            if (shift_by == k) {
                initial_kmer_lcp(k, bits_per_char, local_B2);
                SAC_TIMER_END_LOOP_SECTION(shift_by, "init-lcp");
            } else {
                resolve_next_lcp(shift_by, local_B2);
                SAC_TIMER_END_LOOP_SECTION(shift_by, "update-lcp");
            }
        }

        /*******************************
         *  Assign new bucket numbers  *
         *******************************/
        std::tie(unfinished_buckets, unfinished_elements) = rebucket(local_B2, true);
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
            reorder_sa_to_isa(cpy_SA);
            SAC_TIMER_END_LOOP_SECTION(shift_by, "SA-to-ISA");
            break;
        } else if ((shift_by << 1) >= n || unfinished_buckets == 0) {
            // if last iteration, use copy of local_SA for reorder and keep
            // original SA
            std::vector<index_t> cpy_SA(local_SA);
            reorder_sa_to_isa(cpy_SA);
            SAC_TIMER_END_LOOP_SECTION(shift_by, "SA-to-ISA");
        } else {
            reorder_sa_to_isa();
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

// generalized to more than "doubling" (e.g. prefix-trippling with L=3)
template <std::size_t L = 2>
void construct_arr(bool fast_resolval = true) {
    SAC_TIMER_START();

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
    SAC_TIMER_END_SECTION("initial-bucketing");

    // init local_SA
    if (local_SA.size() != local_B.size()) {
        local_SA.resize(local_B.size());
    }

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
        std::size_t offset = part.excl_prefix_size();
        for (std::size_t i = 0; i < local_size; ++i) {
            tuples[i][0] = i + offset;
            tuples[i][1] = local_B[i];
        }
        SAC_TIMER_END_LOOP_SECTION(shift_by, "arr-tupelize");

        /**************************************************
         *  Pairing buckets by shifting `shift_by` = 2^k  *
         **************************************************/
        // shift the B1 buckets by 2^k to the left => equals B2
        shift_buckets<L>(shift_by, tuples);
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
        std::tie(unfinished_buckets,unfinished_elements) = rebucket_arr<L>(tuples, true);
        if (comm.rank() == 0) {
            INFO("iteration " << shift_by << ": unfinished buckets = " << unfinished_buckets << ", unfinished elements = " << unfinished_elements);
        }
        SAC_TIMER_END_LOOP_SECTION(shift_by, "rebucket");


        /**************************************
         *  Reset local_SA array from tuples  *
         **************************************/
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
            reorder_sa_to_isa(cpy_SA);
            SAC_TIMER_END_LOOP_SECTION(shift_by, "SA-to-ISA");
            break;
        } else if ((shift_by * L) >= n || unfinished_buckets == 0) {
            // if last iteration, use copy of local_SA for reorder and keep
            // original SA
            std::vector<index_t> cpy_SA(local_SA);
            reorder_sa_to_isa(cpy_SA);
            SAC_TIMER_END_LOOP_SECTION(shift_by, "SA-to-ISA");
        } else {
            reorder_sa_to_isa();
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

void construct_fast() {
    SAC_TIMER_START();

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
    reorder_sa_to_isa(cpy_SA);
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



private:

/*********************************************************************
 *                         Initial Bucketing                         *
 *********************************************************************/
// TODO: externalize some code as "k-mer generation"
std::pair<unsigned int, unsigned int> initial_bucketing(unsigned int k = 0)
{
    std::size_t min_local_size = part.local_size(p-1);

    // get global alphabet histogram
    std::vector<index_t> alphabet_hist = alphabet_histogram<InputIterator, index_t>(input_begin, input_end, comm);
    // get mapping table and alphabet sizes
    alphabet_mapping = alphabet_mapping_tbl(alphabet_hist);
    sigma = alphabet_unique_chars(alphabet_hist);
    // bits per character: set l=ceil(log(sigma))
    unsigned int l = alphabet_bits_per_char(sigma);
    // number of characters per word => the `k` in `k-mer`
    unsigned int opt_k = alphabet_chars_per_word<index_t>(l);
    if (k == 0 || k > opt_k) {
        if (k > opt_k)
            std::cerr << "[WARNING] given `k` value of " << k << " is too large, setting k=" << opt_k << " instead." << std::endl;
        k = opt_k;
    }
    // if the input is too small for `k`, choose a smaller `k`
    if (k >= min_local_size) {
        k = min_local_size;
        if (comm.size() == 1 && k > 1)
            k--;
    }

    if (comm.rank() == 0)
        INFO("Detecting sigma=" << sigma << " => l=" << l << ", k=" << k);

    // get k-mer mask
    index_t kmer_mask = ((static_cast<index_t>(1) << (l*k)) - static_cast<index_t>(1));
    if (kmer_mask == 0)
        kmer_mask = ~static_cast<index_t>(0);

    // sliding window k-mer (for prototype only using ASCII alphabet)

    // fill first k-mer (until k-1 positions) and send to left processor
    // filling k-mer with first character = MSBs for lexicographical ordering
    InputIterator str_it = input_begin;
    index_t kmer = 0;
    for (unsigned int i = 0; i < k-1; ++i) {
        kmer <<= l;
        index_t s = (unsigned char)(*str_it);
        kmer |= alphabet_mapping[s];
        ++str_it;
    }

    // send first kmer to left processor
    // TODO: use async left shift!
    index_t last_kmer = mxx::left_shift(kmer, comm);

    // init output
    if (local_B.size() != local_size)
        local_B.resize(local_size);
    auto buk_it = local_B.begin();
    // continue to create all k-mers and add into histogram count
    while (str_it != input_end) {
        // get next kmer
        kmer <<= l;
        index_t s = (unsigned char)(*str_it);
        kmer |= alphabet_mapping[s];
        kmer &= kmer_mask;
        // add to bucket number array
        *buk_it = kmer;
        // increase iterators
        ++str_it;
        ++buk_it;
    }

    // finish the receive to get the last k-1 k-kmers with string data from the
    // processor to the right
    // if not last processor
    if (comm.rank() < comm.size()-1) {
        // TODO: use mxx::future to handle this async left shift
        // wait for the async receive to finish
        //MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
        //req.wait();
    } else {
        // in this case the last k-mers contains shifting `$` signs
        // we assume this to be the `\0` value
        last_kmer = 0;
    }


    // construct last (k-1) k-mers
    for (unsigned int i = 0; i < k-1; ++i) {
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


/*********************************************************************
 *               Shifting buckets (i -> i + 2^l) => B2               *
 *********************************************************************/
void shift_buckets(std::size_t dist, std::vector<index_t>& local_B2) {
    // get # elements to the left
    std::size_t prev_size = part.excl_prefix_size();
    assert(local_size == local_B.size());

    // init B2
    if (local_B2.size() != local_size){
        local_B2.clear();
        local_B2.resize(local_size, 0);
    }

    mxx::datatype mxxindex_t = mxx::get_datatype<index_t>();
    MPI_Datatype mpi_index_t = mxxindex_t.type();

    MPI_Request recv_reqs[2];
    int n_irecvs = 0;
    // receive elements from the right
    if (prev_size + dist < n) {
        std::size_t right_first_gl_idx = prev_size + dist;
        int p1 = part.target_processor(right_first_gl_idx);

        std::size_t p1_gl_end = part.prefix_size(p1);
        std::size_t p1_recv_cnt = p1_gl_end - right_first_gl_idx;

        if (p1 != comm.rank()) {
            // only receive if the source is not myself (i.e., `rank`)
            // [otherwise results are directly written instead of MPI_Sended]
            assert(p1_recv_cnt < std::numeric_limits<int>::max());
            int recv_cnt = p1_recv_cnt;
            MPI_Irecv(&local_B2[0],recv_cnt, mpi_index_t, p1,
                      PSAC_TAG_SHIFT, comm, &recv_reqs[n_irecvs++]);
        }

        if (p1_recv_cnt < local_size && p1 != p-1) {
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
    if (prev_size + local_size - 1 >= dist) {
        int p1 = -1;
        if (prev_size >= dist) {
            std::size_t first_gl_idx = prev_size - dist;
            p1 = part.target_processor(first_gl_idx);
        }
        std::size_t last_gl_idx = prev_size + local_size - 1 - dist;
        int p2 = part.target_processor(last_gl_idx);

        std::size_t local_split;
        if (p1 != p2) {
            // local start index of area for second processor
            if (p1 >= 0) {
                local_split = part.prefix_size(p1) + dist - prev_size;
                // send to first processor
                assert(p1 != comm.rank());
                MPI_Send(&local_B[0], local_split,
                         mpi_index_t, p1, PSAC_TAG_SHIFT, comm);
            } else {
                // p1 doesn't exist, then there is no prefix to add
                local_split = dist - prev_size;
            }
        } else {
            // only one target processor
            local_split = 0;
        }

        if (p2 != comm.rank()) {
            MPI_Send(&local_B[0] + local_split, local_size - local_split,
                     mpi_index_t, p2, PSAC_TAG_SHIFT, comm);
        } else {
            // in this case the split should be exactly at `dist`
            assert(local_split == dist);
            // locally reassign
            for (std::size_t i = local_split; i < local_size; ++i) {
                local_B2[i-local_split] = local_B[i];
            }
        }
    }

    // wait for successful receive:
    MPI_Waitall(n_irecvs, recv_reqs, MPI_STATUS_IGNORE);
}

// shifting with arrays (custom data types)
// shifts (L-1) times into the (L-1) additional Bucket entries
template <std::size_t L>
void shift_buckets(std::size_t k, std::vector<std::array<index_t, 1+L> >& tuples) {
    // get # elements to the left
    std::size_t prev_size = part.excl_prefix_size();

    mxx::datatype mxxindex_t = mxx::get_datatype<index_t>();
    MPI_Datatype mpi_index_t = mxxindex_t.type();
    // start receiving into second bucket and then continue with greater
    int bi = 2;
    for (std::size_t dist = k; dist < L*k; dist += k) {
        MPI_Request recv_reqs[2];
        int n_irecvs = 0;
        MPI_Datatype dts[4];
        int n_dts = 0;
        // receive elements from the right
        if (prev_size + dist < n) {
            std::size_t right_first_gl_idx = prev_size + dist;
            int p1 = part.target_processor(right_first_gl_idx);

            std::size_t p1_gl_end = part.prefix_size(p1);
            std::size_t p1_recv_cnt = p1_gl_end - right_first_gl_idx;

            if (p1 != comm.rank()) {
                // only receive if the source is not myself (i.e., `rank`)
                // [otherwise results are directly written instead of MPI_Sended]
                assert(p1_recv_cnt < std::numeric_limits<int>::max());
                int recv_cnt = p1_recv_cnt;
                // create custom datatype with stride (L+1)
                MPI_Type_vector(recv_cnt,1,L+1,mpi_index_t,&dts[n_dts]);
                MPI_Type_commit(&dts[n_dts]);
                MPI_Irecv(&tuples[0][bi],1, dts[n_dts], p1,
                          PSAC_TAG_SHIFT, comm, &recv_reqs[n_irecvs++]);
                n_dts++;
            }

            if (p1_recv_cnt < local_size && p1 != p-1) {
                // also receive from one more processor
                int p2 = p1+1;
                // since p2 has at least local_size - 1 elements and at least
                // one element came from p1, we can assume that the receive count
                // is our local size minus the already received elements
                std::size_t p2_recv_cnt = local_size - p1_recv_cnt;

                assert(p2_recv_cnt < std::numeric_limits<int>::max());
                int recv_cnt = p2_recv_cnt;
                // send to `p1` (which is necessarily different from `rank`)
                MPI_Type_vector(recv_cnt,1,L+1,mpi_index_t,&dts[n_dts]);
                MPI_Type_commit(&dts[n_dts]);
                MPI_Irecv(&tuples[p1_recv_cnt][bi],1, dts[n_dts], p2,
                          PSAC_TAG_SHIFT, comm, &recv_reqs[n_irecvs++]);
                n_dts++;
            }
        }

        // send elements to the left (split to at most 2 target processors)
        if (prev_size + local_size - 1 >= dist) {
            int p1 = -1;
            if (prev_size >= dist) {
                std::size_t first_gl_idx = prev_size - dist;
                p1 = part.target_processor(first_gl_idx);
            }
            std::size_t last_gl_idx = prev_size + local_size - 1 - dist;
            int p2 = part.target_processor(last_gl_idx);

            std::size_t local_split;
            if (p1 != p2) {
                // local start index of area for second processor
                if (p1 >= 0) {
                    local_split = part.prefix_size(p1) + dist - prev_size;
                    // send to first processor
                    assert(p1 != comm.rank());
                    MPI_Type_vector(local_split,1,L+1,mpi_index_t,&dts[n_dts]);
                    MPI_Type_commit(&dts[n_dts]);
                    MPI_Send(&tuples[0][1], 1,
                             dts[n_dts], p1, PSAC_TAG_SHIFT, comm);
                    n_dts++;
                } else {
                    // p1 doesn't exist, then there is no prefix to add
                    local_split = dist - prev_size;
                }
            } else {
                // only one target processor
                local_split = 0;
            }

            if (p2 != comm.rank()) {
                MPI_Type_vector(local_size - local_split,1,L+1,mpi_index_t,&dts[n_dts]);
                MPI_Type_commit(&dts[n_dts]);
                MPI_Send(&tuples[local_split][1], 1,
                         dts[n_dts], p2, PSAC_TAG_SHIFT, comm);
                n_dts++;
            } else {
                // in this case the split should be exactly at `dist`
                assert(local_split == dist);
                // locally reassign
                for (std::size_t i = local_split; i < local_size; ++i)
                {
                    tuples[i-local_split][bi] = tuples[i][1];
                }
            }
        }

        // wait for successful receive:
        MPI_Waitall(n_irecvs, recv_reqs, MPI_STATUS_IGNORE);

        // clean up data types from this round
        for (int i = 0; i < n_dts; ++i) {
            MPI_Type_free(&dts[i]);
        }

        // next target bucket
        bi++;
    }
}


/*********************************************************************
 *                     ISA -> SA (sort buckets)                      *
 *********************************************************************/
void isa_2b_to_sa(std::vector<index_t>& local_B2) {
    assert(local_B2.size() == local_size);
    SAC_TIMER_START();

    // convert the struct of arrays (local_SA, local_B, etc) into
    // array of structs (TwoBSA {.B1, .B2, .SA}) for sorting purposes

    // initialize tuple array
    std::vector<TwoBSA<index_t> > tuple_vec(local_size);

    // get global index offset
    std::size_t str_offset = part.excl_prefix_size();

    // fill tuple vector
    for (std::size_t i = 0; i < local_size; ++i) {
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
    mxx::sort(tuple_vec.begin(), tuple_vec.end(), comm);

    SAC_TIMER_END_SECTION("isa2sa_samplesort");

    // reallocate output
    local_B.resize(local_size);
    local_B2.resize(local_size);
    local_SA.resize(local_size);

    // back convert array of structs into struct of arrays

    // read back into input vectors
    for (std::size_t i = 0; i < local_size; ++i) {
        local_B[i] = tuple_vec[i].B1;
        local_B2[i] = tuple_vec[i].B2;
        local_SA[i] = tuple_vec[i].SA;
    }
    SAC_TIMER_END_SECTION("isa2sa_untupleize");
}


template <std::size_t L>
void sort_array_tuples(std::vector<std::array<index_t, L+1> >& tuples) {
    assert(tuples.size() == local_size);
    SAC_TIMER_START();

    // parallel, distributed sample-sorting of tuples (B1, B2, SA)
    mxx::sort(tuples.begin(), tuples.end(),
    [] (const std::array<index_t, L+1>& x,
        const std::array<index_t, L+1>& y) {
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
    std::size_t str_offset = part.excl_prefix_size();

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
    size_t prefix = part.excl_prefix_size();
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

// assumed sorted order (globally) by tuple (B1[i], B2[i])
// this reassigns new, unique bucket numbers in {1,...,n} globally
std::pair<size_t,size_t> rebucket(std::vector<index_t>& local_B2, bool count_unfinished) {
    /*
     * NOTE: buckets are indexed by the global index of the first element in
     *       the bucket with a ONE-BASED-INDEX (since bucket number `0` is
     *       reserved for out-of-bounds)
     */
    // assert inputs are of equal size
    assert(local_B.size() == local_B2.size() && local_B.size() > 0);

    // init result
    std::pair<size_t,size_t> result;

    // get my global starting index
    size_t prefix = part.excl_prefix_size();

    /*
     * assign local zero or one, depending on whether the bucket is the same
     * as the previous one
     */

    std::pair<index_t, index_t> last_element = std::make_pair(local_B.back(), local_B2.back());
    std::pair<index_t, index_t> prevRight = mxx::right_shift(last_element, comm);
    bool firstDiff = false;
    if (comm.rank() == 0) {
        firstDiff = true;
    } else if (prevRight.first != local_B[0] || prevRight.second != local_B2[0]) {
        firstDiff = true;
    }

    // set local_B1 to `1` if previous entry is different:
    // i.e., mark start of buckets
    bool nextDiff = firstDiff;
    for (std::size_t i = 0; i+1 < local_B.size(); ++i) {
        bool setOne = nextDiff;
        nextDiff = (local_B[i] != local_B[i+1] || local_B2[i] != local_B2[i+1]);
        local_B[i] = setOne ? prefix+i+1 : 0;
    }

    local_B.back() = nextDiff ? prefix+(local_size-1)+1 : 0;

    if (count_unfinished) {
        // count the number of unfinished elements and buckets
        index_t prev_right = mxx::right_shift(local_B.back(), comm);
        index_t local_unfinished_buckets = 0;
        index_t local_unfinished_els = 0;
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
    // 1.) find the max in the local sequence. since the max is the last index
    //     of a bucket, this should be somewhere at the end -> start scanning
    //     from the end
    auto rev_it = local_B.rbegin();
    size_t local_max = 0;
    while (rev_it != local_B.rend() && (local_max = *rev_it) == 0)
        ++rev_it;

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
// assumed sorted order (globally) by tuple (B1[i], B2[i])
// this reassigns new, unique bucket numbers in {1,...,n} globally
template <size_t L>
std::pair<size_t,size_t> rebucket_arr(std::vector<std::array<index_t, L+1> >& tuples, bool count_unfinished) {
    /*
     * NOTE: buckets are indexed by the global index of the first element in
     *       the bucket with a ONE-BASED-INDEX (since bucket number `0` is
     *       reserved for out-of-bounds)
     */

    // init result
    std::pair<size_t,size_t> result;
    // get my global starting index
    size_t prefix = part.excl_prefix_size();
    size_t local_max = 0;

    foreach_pair(tuples.begin(), tuples.end(), [&](const std::array<index_t, L+1>& prev, std::array<index_t, L+1>& cur, size_t i) {
        if (!std::equal(&prev[1], &prev[1]+L, &cur[1])) {
            local_B[i] = prefix + i + 1;
            local_max = prefix + i + 1;
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
        index_t prev_right = mxx::right_shift(local_B.back(), comm);
        index_t local_unfinished_buckets = 0;
        index_t local_unfinished_els = 0;
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

// same function as before, but this one assumes tuples instead of
// two arrays
// This is used in the bucket chaising construction. The MPI_Comm will most
// of the time be a subcommunicator (so do not use the member `comm`)
void rebucket_tuples(std::vector<TwoBSA<index_t> >& tuples, const mxx::comm& comm, std::size_t gl_offset, std::vector<std::tuple<index_t, index_t, index_t> >& minqueries)
{
    /*
     * NOTE: buckets are indexed by the global index of the first element in
     *       the bucket with a ONE-BASED-INDEX (since bucket number `0` is
     *       reserved for out-of-bounds)
     */
    // inputs can be of different size, since buckets can span bucket boundaries
    std::size_t local_size = tuples.size();
    std::size_t prefix = mxx::exscan(local_size, std::plus<size_t>(), comm);
    size_t local_max = 0;
    prefix += gl_offset;

    // iterate through all pairs of elements (spanning across processors)
    foreach_pair(tuples.begin(), tuples.end(), [&](const TwoBSA<index_t>& prev, TwoBSA<index_t>& cur, size_t i){
        // if this is a new bucket boundary: set current to prefix+index and update LCP
        if (prev.B1 != cur.B1 || prev.B2 != cur.B2){
            // set every bucket boundary to its global index and other elements to
            // 0, a following max-scan will then distribute this bucket index among
            // its elements
            cur.B1 = prefix+i+1;
            local_max = cur.B1;
            if (_CONSTRUCT_LCP) {
                index_t left_b  = std::min(prev.B2, cur.B2);
                index_t right_b = std::max(prev.B2, cur.B2);
                // we need the minumum LCP of all suffixes in buckets between
                // these two buckets. Since the first element in the left bucket
                // is the LCP of this bucket with its left bucket and we don't need
                // this LCP value, start one to the right:
                // (-1 each since buffer numbers are current index + 1)
                index_t range_left = (left_b-1) + 1;
                index_t range_right = (right_b-1) + 1; // +1 since exclusive index
                minqueries.emplace_back(i + prefix, range_left, range_right);
            }
        } else {
            cur.B1 = 0;
        }
    }, comm);

    // specially handle first element of first process
    if (comm.rank() == 0) {
        tuples[0].B1 = prefix + 1;
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
        if (tuples[i].B1 == 0)
            tuples[i].B1 = pre_max;
        else
            pre_max = tuples[i].B1;
        assert(tuples[i].B1 <= i+prefix+1);
        // first element of bucket has id of it's own global index:
        assert(i == 0 || (tuples[i-1].B1 ==  tuples[i].B1 || tuples[i].B1 == i+prefix+1));
    }
}

/*********************************************************************
 *                       SA->ISA (~bucketsort)                       *
 *********************************************************************/
void reorder_sa_to_isa(std::vector<index_t>& SA) {
    assert(SA.size() == local_B.size());

    SAC_TIMER_START();
    // 1.) local bucketing for each processor
    //
    // counting the number of elements for each processor
    std::vector<size_t> send_counts(p, 0);
    for (index_t sa : SA) {
        int target_p = part.target_processor(sa);
        assert(0 <= target_p && target_p < p);
        ++send_counts[target_p];
    }
    // get exclusive prefix sum
    std::vector<size_t> send_displs = mxx::local_exscan(send_counts);
    std::vector<size_t> upper_bound = mxx::local_scan(send_counts);

    // in-place bucketing
    int cur_p = 0;
    for (std::size_t i = 0; i < SA.size();) {
        // skip full buckets
        while (cur_p < p-1 && send_displs[cur_p] >= upper_bound[cur_p]) {
            // skip over full buckets
            i = send_displs[++cur_p];
        }
        // break if all buckets are done
        if (cur_p >= p-1)
            break;
        int target_p = part.target_processor(SA[i]);
        assert(target_p < p && target_p >= 0);
        if (target_p == cur_p) {
            // item correctly placed
            ++i;
        } else {
            // swap to correct bucket
            assert(target_p > cur_p);
            std::swap(SA[i], SA[send_displs[target_p]]);
            std::swap(local_B[i], local_B[send_displs[target_p]]);
        }
        send_displs[target_p]++;
    }

    SAC_TIMER_END_SECTION("sa2isa_bucketing");

    // get displacements again (since they were modified above)
    std::vector<index_t> recv_SA = mxx::all2allv(SA, send_counts, comm);
    std::vector<index_t> recv_B = mxx::all2allv(local_B, send_counts, comm);
    SAC_TIMER_END_SECTION("sa2isa_all2all");

    // rearrange locally
    for (std::size_t i = 0; i < SA.size(); ++i) {
        index_t out_idx = recv_SA[i] - part.excl_prefix_size();
        assert(0 <= out_idx && out_idx < recv_SA.size());
        local_B[out_idx] = recv_B[i];
    }

    // reassign the SA
    std::size_t global_offset = part.excl_prefix_size();
    for (std::size_t i = 0; i < SA.size(); ++i) {
        SA[i] = global_offset + i;
    }
    SAC_TIMER_END_SECTION("sa2isa_rearrange");
}

// SA->ISA defaulting with the local_SA array
void reorder_sa_to_isa()
{
    reorder_sa_to_isa(local_SA);
}


/*********************************************************************
 *          Faster construction for fewer remaining buckets          *
 *********************************************************************/

void construct_msgs(std::vector<index_t>& local_B, std::vector<index_t>& local_ISA, int dist)
{
    /*
     * Algorithm for few remaining buckets (more communication overhead per
     * element but sends only unfinished buckets -> less data in total if few
     * buckets remaining)
     *
     * INPUT:
     *  - SA in SA order
     *  - B in SA order
     *  - ISA in ISA order (<=> B in ISA order)
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


    /*
     * 0.) Preparation: need unfinished buckets (info accross proc. boundaries)
     */
    // get next element from right
    index_t right_B = mxx::left_shift(local_B[0], comm);

    // get global offset
    const std::size_t prefix = part.excl_prefix_size();

    // get active elements
    std::vector<index_t> active_elements;
    for (std::size_t j = 0; j < local_B.size(); ++j) {
        // get global index for each local index
        std::size_t i =  prefix + j;
        // check if this is a unresolved bucket
        // relying on the property that for resolved buckets:
        //   B[i] == i+1 and B[i+1] == i+2
        //   (where `i' is the global index)
        if (local_B[j] != i+1 || (local_B[j] == i+1
                    && ((j < local_size-1 && local_B[j+1] == i+1)
                        || (j == local_size-1 && comm.rank() < p-1 && right_B == i+1))))
        {
            // save local active indexes
            active_elements.push_back(j);
        }
    }
    SAC_TIMER_END_SECTION("get active elements");

    bool right_bucket_crosses_proc = (comm.rank() < p-1 && local_B.back() == right_B);

    for (index_t shift_by = dist; shift_by < n; shift_by <<= 1) {
        /*
         * 1.) on i: send tuple (`to:` Sa[i]+2^k, `from:` i)
         */
        //std::vector<std::pair<index_t, index_t> > msgs;
        //std::vector<std::pair<index_t, index_t> > out_of_bounds_msgs;
        std::vector<mypair<index_t> > msgs;
        std::vector<mypair<index_t> > out_of_bounds_msgs;
        // linear scan for bucket boundaries
        // and create tuples/pairs
        std::size_t unresolved_els = 0;
        std::size_t unfinished_b = 0;
        for (index_t j : active_elements) {
            // get global index for each local index
            std::size_t i =  prefix + j;
            // add tuple
            if (local_SA[j] + shift_by >= n)
                //out_of_bounds_msgs.push_back(std::make_pair<index_t,index_t>(0, static_cast<index_t>(i)));
                out_of_bounds_msgs.push_back({0, static_cast<index_t>(i)});
            else
                //msgs.push_back(std::make_pair<index_t,index_t>(local_SA[j]+shift_by, static_cast<index_t>(i)));
                msgs.push_back({local_SA[j]+shift_by, static_cast<index_t>(i)});
            unresolved_els++;
            if (local_B[j] == i+1) // if first element of unfinished bucket:
                unfinished_b++;
        }

        // check if all resolved
        std::size_t gl_unresolved = mxx::allreduce(unresolved_els, comm);
        std::size_t gl_unfinished = mxx::allreduce(unfinished_b, comm);
        if (comm.rank() == 0) {
            INFO("==== chaising iteration " << shift_by << " unresolved = " << gl_unresolved);
            INFO("==== chaising iteration " << shift_by << " unfinished = " << gl_unfinished);
        }
        if (gl_unresolved == 0)
            // finished!
            break;

        // message exchange to processor which contains first index
        //msgs_all2all(msgs, [&](const std::pair<index_t, index_t>& x){return part.target_processor(x.first);}, comm);
        mxx::all2all_func(msgs, [&](const mypair<index_t>& x){return part.target_processor(x.first);}, comm);

        // for each message, add the bucket no. into the `first` field
        for (auto it = msgs.begin(); it != msgs.end(); ++it) {
            it->first = local_ISA[it->first - prefix];
        }


        /*
         * 2.)
         */
        // send messages back to originator
        //msgs_all2all(msgs, [&](const std::pair<index_t, index_t>& x){return part.target_processor(x.second);}, comm);
        mxx::all2all_func(msgs, [&](const mypair<index_t>& x){return part.target_processor(x.second);}, comm);

        // append the previous out-of-bounds messages (since they all have B2 = 0)
        if (out_of_bounds_msgs.size() > 0)
            msgs.insert(msgs.end(), out_of_bounds_msgs.begin(), out_of_bounds_msgs.end());
        out_of_bounds_msgs.clear();

        assert(msgs.size() == unresolved_els);

        // sort received messages by the target index to enable consecutive
        // scanning of local buckets and messages
        std::sort(msgs.begin(), msgs.end(), [](const mypair<index_t>& x, const mypair<index_t>& y){ return x.second < y.second;});

        /*
         * 3.)
         */
        // building sequence of triplets for each unfinished bucket and sort
        // then rebucket, buckets which spread accross boundaries, sort via
        // MPI sub communicators and samplesort in two phases
        std::vector<TwoBSA<index_t> > bucket;
        std::vector<TwoBSA<index_t> > left_bucket;
        std::vector<TwoBSA<index_t> > right_bucket;

        // prepare LCP queries vector
        std::vector<std::tuple<index_t, index_t, index_t> > minqueries;

        // find bucket boundaries:
        auto msgit = msgs.begin();
        // overlap type:    0: no overlaps, 1: left overlap, 2:right overlap,
        //                  3: separate overlaps on left and right
        //                  4: contiguous overlap with both sides
        int overlap_type = 0; // init to no overlaps
        std::size_t bucket_begin = local_B[0]-1;
        std::size_t first_bucket_begin = bucket_begin;
        std::size_t right_bucket_offset = 0;
        while (msgit != msgs.end()) {
            bucket_begin = local_B[msgit->second - prefix]-1;
            assert(bucket_begin < prefix || bucket_begin == msgit->second);

            // find end of bucket
            while (msgit != msgs.end() && local_B[msgit->second - prefix]-1 == bucket_begin) {
                TwoBSA<index_t> tuple;
                assert(msgit->second >= prefix && msgit->second < prefix+local_size);
                tuple.SA = local_SA[msgit->second - prefix];
                tuple.B1 = local_B[msgit->second - prefix];
                tuple.B2 = msgit->first;
                bucket.push_back(tuple);
                msgit++;
            }

            // get bucket end (could be on other processor)
            if (msgit == msgs.end() && right_bucket_crosses_proc) {
                assert(comm.rank() < p-1 && local_B.back() == right_B);
                if (bucket_begin >= prefix)
                {
                    overlap_type += 2;
                    right_bucket.swap(bucket);
                    right_bucket_offset = bucket_begin - prefix;
                }
                else
                {
                    // bucket extends to left AND right
                    left_bucket.swap(bucket);
                    overlap_type = 4;
                }
            } else {
                if (bucket_begin >= prefix)
                {
                    // this is a local bucket => sort by B2, rebucket, and save
                    // TODO custom comparison that only sorts by B2, not by B1 as well
                    std::sort(bucket.begin(), bucket.end());
                    // local rebucket
                    // save back into local_B, local_SA, etc
                    index_t cur_b = bucket_begin + 1;
                    std::size_t out_idx = bucket_begin - prefix;
                    // assert previous bucket index is smaller
                    assert(out_idx == 0 || local_B[out_idx-1] < cur_b);
                    for (auto it = bucket.begin(); it != bucket.end(); ++it)
                    {
                        // if this is a new bucket, then update number
                        if (it != bucket.begin() && (it-1)->B2 != it->B2)
                        {
                            // update bucket index
                            cur_b = out_idx + prefix + 1;

                            if (_CONSTRUCT_LCP)
                            {
                                // add as query item for LCP construction
                                index_t left_b  = std::min((it-1)->B2, it->B2);
                                index_t right_b = std::max((it-1)->B2, it->B2);
                                // we need the minumum LCP of all suffixes in buckets between
                                // these two buckets. Since the first element in the left bucket
                                // is the LCP of this bucket with its left bucket and we don't need
                                // this LCP value, start one to the right:
                                // (-1 each since buffer numbers are current index + 1)
                                index_t range_left = (left_b-1) + 1;
                                index_t range_right = (right_b-1) + 1; // +1 since exclusive index
                                minqueries.emplace_back(out_idx + prefix, range_left, range_right);
                            }
                        }
                        local_SA[out_idx] = it->SA;
                        local_B[out_idx] = cur_b;
                        out_idx++;
                    }
                    // assert next bucket index is larger
                    assert(out_idx == local_size || local_B[out_idx] == prefix+out_idx+1);
                }
                else
                {
                    overlap_type += 1;
                    left_bucket.swap(bucket);
                }
            }
            bucket.clear();
        }

        // if we have left/right/both/or double buckets, do global comm in two phases
        int my_schedule = -1;
        if (comm.rank() == 0) {
            // gather all types to first processor
            std::vector<int> overlaps(p);
            MPI_Gather(&overlap_type, 1, MPI_INT, &overlaps[0], 1, MPI_INT, 0, comm);

            // create schedule using linear scan over the overlap types
            std::vector<int> schedule(p);
            int phase = 0; // start in first phase
            for (int i = 0; i < p; ++i) {
                switch (overlaps[i]) {
                    case 0:
                        schedule[i] = -1; // doesn't matter
                        break;
                    case 1:
                        // only left overlap -> participate in current phase
                        schedule[i] = phase;
                        break;
                    case 2:
                        // only right overlap, start with phase 0
                        phase = 0;
                        schedule[i] = phase;
                        break;
                    case 3:
                        // separate overlaps left and right -> switch phase
                        schedule[i] = phase; // left overlap starts with current phase
                        phase = 1 - phase;
                        break;
                    case 4:
                        // overlap with both: left and right => keep phase
                        schedule[i] = phase;
                        break;
                    default:
                        assert(false);
                        break;
                }
            }

            // scatter the schedule to the processors
            MPI_Scatter(&schedule[0], 1, MPI_INT, &my_schedule, 1, MPI_INT, 0, comm);
        } else {
            // send out my overlap type
            MPI_Gather(&overlap_type, 1, MPI_INT, NULL, 1, MPI_INT, 0, comm);

            // ... let master processor solve the schedule

            // receive schedule:
            MPI_Scatter(NULL, 1, MPI_INT, &my_schedule, 1, MPI_INT, 0, comm);
        }


        // two phase sorting across boundaries using sub communicators
        for (int phase = 0; phase <= 1; ++phase) {
            std::vector<TwoBSA<index_t> > border_bucket = left_bucket;
            // the leftmost processor of a group will be used as split
            int left_p = part.target_processor(first_bucket_begin);
            bool participate = (overlap_type != 0 && my_schedule == phase);
            std::size_t bucket_offset = 0; // left bucket starts from beginning
            std::size_t rebucket_offset = first_bucket_begin;
            if ((my_schedule != phase && overlap_type == 3) || (my_schedule == phase && overlap_type == 2)) {
                // starting a bucket at the end
                border_bucket = right_bucket;
                left_p = comm.rank();
                participate = true;
                bucket_offset = right_bucket_offset;
                rebucket_offset = prefix + bucket_offset;
            }

            comm.with_subset(participate,[&](const mxx::comm& sc) {
                // split communicator to `left_p`
                mxx::comm subcomm = sc.split(left_p);

                // sample sort the bucket with arbitrary distribution
                mxx::sort(border_bucket.begin(), border_bucket.end(), subcomm);

#ifndef NDEBUG
                index_t first_bucket = border_bucket[0].B1;
#endif
                // rebucket with global offset of first -> in tuple form (also updates LCP)
                rebucket_tuples(border_bucket, subcomm, rebucket_offset, minqueries);
                // assert first bucket index remains the same
                assert(subcomm.rank() != 0 || first_bucket == border_bucket[0].B1);

                // save into full array (if this was left -> save to beginning)
                // (else, need offset of last)
                assert(bucket_offset == 0 || local_B[bucket_offset-1] < border_bucket[0].B1);
                assert(bucket_offset+border_bucket.size() <= local_size);
                for (std::size_t i = 0; i < border_bucket.size(); ++i) {
                    local_SA[i+bucket_offset] = border_bucket[i].SA;
                    local_B[i+bucket_offset] = border_bucket[i].B1;
                }
                assert(bucket_offset+border_bucket.size() == local_size || (local_B[bucket_offset+border_bucket.size()] > local_B[bucket_offset+border_bucket.size()-1]));
                assert(subcomm.rank() != 0 || local_B[bucket_offset] == bucket_offset+prefix+1);

                /*
                 * LCP update
                 */
                // LCP is updated in the custom `rebucket_tuples` function
            });

            comm.barrier();
        }


        // get new bucket number to the right
        right_B = mxx::left_shift(local_B[0], comm);
        // check if right bucket still goes over boundary
        right_bucket_crosses_proc = (comm.rank() < p-1 && local_B.back() == right_B);

        // remember all the remaining active elements
        active_elements.clear();
        for (auto it = msgs.begin(); it != msgs.end(); ++it) {
            index_t j = it->second - prefix;
            index_t i = it->second;
            // check if this is a unresolved bucket
            // relying on the property that for resolved buckets:
            //   B[i] == i+1 and B[i+1] == i+2
            //   (where `i' is the global index)
            if (local_B[j] != i+1 || (local_B[j] == i+1
                        && ((j < local_size-1 && local_B[j+1] == i+1)
                            || (j == local_size-1 && comm.rank() < p-1 && right_B == i+1))))
            {
                // save local active indexes
                active_elements.push_back(j);
            }
        }

        /*
         * 4.1)   Update LCP
         */
        if (_CONSTRUCT_LCP) {
            // time LCP separately!
            SAC_TIMER_START();
            // get parallel-distributed RMQ for all queries, results are in
            // `minqueries`
            // TODO: bulk updatable RMQs [such that we don't have to construct the
            //       RMQ for the local_LCP in each iteration]
            bulk_rmq(n, local_LCP, minqueries, comm);

            // update the new LCP values:
            for (auto min_lcp : minqueries) {
                local_LCP[std::get<0>(min_lcp) - prefix] = shift_by + std::get<2>(min_lcp);
            }
            SAC_TIMER_END_SECTION("LCP update");
        }

        /*
         * 4.2)  Update ISA
         */
        // message new bucket numbers to new SA[i] for all previously unfinished
        // buckets
        // since the message array is still available with the indices of unfinished
        // buckets -> reuse that information => no need to rescan the whole
        // local array
        for (auto it = msgs.begin(); it != msgs.end(); ++it) {
            it->first = local_SA[it->second - prefix]; // SA[i]
            it->second = local_B[it->second - prefix]; // B[i]
        }

        // message exchange to processor which contains first index
        mxx::all2all_func(msgs, [&](const mypair<index_t>& x){return part.target_processor(x.first);}, comm);

        // update local ISA with new bucket numbers
        for (auto it = msgs.begin(); it != msgs.end(); ++it) {
            local_ISA[it->first-prefix] = it->second;
        }
        SAC_TIMER_END_SECTION("bucket-chaising iteration");
    }
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

// for pairs of two buckets: pair[i] = (B1[i], B2[i])
void initial_kmer_lcp(unsigned int k, unsigned int bits_per_char,
                      const std::vector<index_t>& local_B2) {
    // for each bucket boundary (using both the B1 and the B2 buckets):
    //    get the LCP by getting position of first different bit and dividing by
    //    `bits_per_char`

    // resize to size `local_size` and set all items to max of n
    local_LCP.assign(local_size, n);

    // find bucket boundaries (including across processors)

    // 1) getting next element to left
    std::pair<index_t, index_t> right_B(local_B.back(), local_B2.back());
    std::pair<index_t, index_t> left_B = mxx::right_shift(right_B, comm);

    // initialize first LCP
    if (comm.rank() == 0) {
        local_LCP[0] = 0;
    } else {
        if (left_B.first != local_B[0]
            || (left_B.first == local_B[0] && left_B.second != local_B2[0])) {
            unsigned int lcp = lcp_bitwise(left_B.first, local_B[0], k, bits_per_char);
            if (lcp == k)
                lcp += lcp_bitwise(left_B.second, local_B2[0], k, bits_per_char);
            local_LCP[0] = lcp;
        }
    }

    // intialize the LCP for all other elements for bucket boundaries
    for (std::size_t i = 1; i < local_size; ++i)
    {
        if (local_B[i-1] != local_B[i]
            || (local_B[i-1] == local_B[i] && local_B2[i-1] != local_B2[i])) {
            unsigned int lcp = lcp_bitwise(local_B[i-1], local_B[i], k, bits_per_char);
            if (lcp == k)
                lcp += lcp_bitwise(local_B2[i-1], local_B2[i], k, bits_per_char);
            local_LCP[i] = lcp;
        }
    }
}

void resolve_next_lcp(int dist, const std::vector<index_t>& local_B2) {
    // 2.) find _new_ bucket boundaries (B1[i-1] == B1[i] && B2[i-1] != B2[i])
    // 3.) bulk-parallel-distributed RMQ for ranges (B2[i-1],B2[i]+1) to get min_lcp[i]
    // 4.) LCP[i] = dist + min_lcp[i]

    std::size_t prefix_size = part.excl_prefix_size();

    // get right-most element of left processor and check if it is a new
    // bucket boundary
    std::pair<index_t, index_t> right_B(local_B.back(), local_B2.back());
    std::pair<index_t, index_t> left_B = mxx::right_shift(right_B, comm);

    // find _new_ bucket boundaries and create associated parallel distributed
    // RMQ queries.
    std::vector<std::tuple<index_t, index_t, index_t> > minqueries;

    // check if the first element is a new bucket boundary
    if (comm.rank() > 0 && left_B.first == local_B[0] && left_B.second != local_B2[0]) {
        index_t left_b  = std::min(left_B.second, local_B2[0]);
        index_t right_b = std::max(left_B.second, local_B2[0]);
        // we need the minumum LCP of all suffixes in buckets between
        // these two buckets. Since the first element in the left bucket
        // is the LCP of this bucket with its left bucket and we don't need
        // this LCP value, start one to the right:
        // (-1 each since buffer numbers are current index + 1)
        index_t range_left = (left_b-1) + 1;
        index_t range_right = (right_b-1) + 1; // +1 since exclusive index
        minqueries.emplace_back(0 + prefix_size, range_left, range_right);
    }

    // check for new bucket boundaries in all other elements
    for (std::size_t i = 1; i < local_size; ++i) {
        // if this is the first element of a new bucket
        if (local_B[i - 1] == local_B[i] && local_B2[i - 1] != local_B2[i]) {
            index_t left_b  = std::min(local_B2[i-1], local_B2[i]);
            index_t right_b = std::max(local_B2[i-1], local_B2[i]);
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

#ifndef NDEBUG
    std::size_t _nqueries = minqueries.size();
#endif

    // get parallel-distributed RMQ for all queries, results are in
    // `minqueries`
    // TODO: bulk updatable RMQs [such that we don't have to construct the
    //       RMQ for the local_LCP in each iteration]
    bulk_rmq(n, local_LCP, minqueries, comm);
    assert(minqueries.size() == _nqueries);


    // update the new LCP values:
    for (auto min_lcp : minqueries)
    {
        local_LCP[std::get<0>(min_lcp) - prefix_size] = dist + std::get<2>(min_lcp);
    }
}
};


#endif // SUFFIX_ARRAY_HPP
