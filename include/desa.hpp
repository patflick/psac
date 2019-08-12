/*
 * Copyright 2018 Georgia Institute of Technology
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

#ifndef DESA_HPP
#define DESA_HPP

#include "suffix_array.hpp"
#include "rmq.hpp"
#include "lookup_table.hpp"
#include "partition.hpp"
#include "dstrings.hpp"

#include <mxx/comm.hpp>
#include <mxx/reduction.hpp>
#include <mxx/bcast.hpp>
#include <mxx/file.hpp>

#define DESA_CONSTR_NAIVE_LC 0


/**
 * @brief Redistribute a distributed vector `v`, so that the calling processor
 *        contains global indexes [gbeg,gend).
 *
 * This is a collective call.
 *
 * @return The redistributed vector with `gend-gbeg` elements corresponding to
 *          global elements [gbeg, gend).
 */
template <typename T>
std::vector<T> redistr(const std::vector<T>& v, size_t gbeg, size_t gend, const mxx::comm& comm) {
    assert(gbeg <= gend);
    std::vector<T> res(gend - gbeg);
    // TODO: implememnt more efficient version for this (using send/recv only)
    mxx::redo_arbit_decomposition(v.begin(), v.end(), res.begin(), gend - gbeg, comm);
    return res;
}

/**
 * @brief Redistribute the distributed suffix array `sa`,
 *        such that the calling processor contains global indexes [gbeg,gend).
 *
 * This is a collective call.
 */
template <typename index_t, bool CONSTRUCT_LC>
void redistr_sa(suffix_array<char, index_t, true, CONSTRUCT_LC>& sa, size_t gbeg, size_t gend, const mxx::comm& comm) {
    sa.local_SA = redistr(sa.local_SA, gbeg, gend, comm);
    sa.local_LCP = redistr(sa.local_LCP, gbeg, gend, comm);
    if (!sa.local_B.empty()) {
        sa.local_B = redistr(sa.local_B, gbeg, gend, comm);
    }
    if (CONSTRUCT_LC) {
        sa.local_Lc = redistr(sa.local_Lc, gbeg, gend, comm);
    }
}




template <typename index_t>
struct tllt {
    using range_t = std::pair<index_t,index_t>;
    lookup_index<index_t> idx;
    size_t n;

    template <typename char_t>
    void construct(const suffix_array<char_t, index_t, true, true>& sa, const std::string& local_str, const mxx::comm& comm) {
        // choose a "good" `q`
        unsigned int l = sa.alpha.bits_per_char();
        unsigned int log_size = 24; // 2^24 = 16 M (*8 = 128 M)
        unsigned int q = log_size / l;

        // `q` might have to be choosen smaller if the input is very small
        if (sa.local_size < q) {
            q = sa.local_size;
            q = std::max(q,1u);
        }
        q = mxx::allreduce(q, mxx::min<unsigned int>(), comm);
        if (q < log_size / l && comm.rank() == 0) {
            std::cerr << "[WARNING] q for q-mer partitioning was choosen smaller (q=" << q << " vs opt_q=" << log_size/l << ") due to small local size." << std::endl;
        }

        // construct kmer lookup table
        std::vector<uint64_t> hist = kmer_hist<uint64_t>(local_str.begin(), local_str.end(), q, sa.alpha, comm);
        idx.construct_from_hist(hist, q, sa.alpha);
    }

    inline index_t minmatch() const {
        return idx.k;
    }

    // for partitioning
    const std::vector<index_t>& prefix() const {
        return idx.table;
    }
    std::vector<index_t>& prefix() {
        return idx.table;
    }

    template <typename String>
    inline range_t lookup(const String& P) const {
        return idx.lookup(P);
    }
};


    // partition information:
    // after partitioning subtrees, this processor now contains the following
    // indeces:

    // TODO: separate class for `arbitrary_distribution`:
    //       containing also the `lt_proc` and `part` tables from above and the info below
    //       and provide some useful functions

struct gen_dist {
    /// globally number of elements
    size_t global_size;
    /// global index of first element on this processor
    size_t my_begin;
    /// global index of first element on next processor
    size_t my_end;
    /// = my_end - my_begin: number of elements on this processor
    size_t local_size;

    /// partition: assigns each processor it's lookup table index (shared copy)

    /// inverse partition mapping
    // map each processor start index -> processor index
    std::map<size_t, int> lt_proc; // XXX: possibly optimize: use vector and binary search

    int target_processor(size_t gidx) {
        auto ub = lt_proc.upper_bound(gidx); // returns iterator to first element > gidx
        --ub; // then the previous element is the target processor
        return ub->second;
    }

    void check_correct(const mxx::comm& comm) const {
        // check that these are all exclusive?
        size_t next_beg = mxx::left_shift(my_begin, comm);
        size_t prev_end = mxx::right_shift(my_end, comm);

        assert(my_begin <= my_end);
        if (comm.rank() > 0) {
          assert(prev_end == my_begin);
        } else {
          assert(my_begin == 0);
        }

        if (comm.rank()+1 < comm.size()) {
          assert(next_beg == my_end);
        } else {
          assert(my_end == global_size);
        }
    }

    void print_imbalance_stats(const mxx::comm& comm) const {
        size_t min_size = mxx::allreduce(local_size, mxx::min<size_t>(), comm);
        size_t max_size = mxx::allreduce(local_size, mxx::max<size_t>(), comm);

        std::vector<size_t> local_sizes = mxx::gather(local_size, 0, comm);

        size_t ex_size = global_size / comm.size();

        double imbalance = (max_size - ex_size)*1. / ex_size;

        if (comm.rank() == 0) {
            std::cout << "Repartition load im-balance: " << imbalance*100. << "%,  range: [" << min_size << "," << max_size << "]" << std::endl;
            std::cout << "  sizes: " << local_sizes << std::endl;
        }
    }

    template <typename index_t=size_t>
    static gen_dist from_prefix_sizes(const std::vector<index_t>& table, const mxx::comm& comm) {
        mxx::section_timer t;
        gen_dist d;
        d.global_size = table.back();
        // partition the lookup index by processors
        std::vector<size_t> part = partition(table, comm.size());
        for (int i = 0; i < comm.size(); ++i) {
            size_t kmer_l = part[i];
            size_t proc_begin = (kmer_l == 0) ? 0 : table[kmer_l-1];
            d.lt_proc[proc_begin] = i;
        }

        // kmers from: [part[comm.rank()] .. part[comm.rank()+1])
        size_t my_kmer_l = part[comm.rank()];
        size_t my_kmer_r = (comm.rank()+1 == comm.size()) ? table.size() : part[comm.rank()+1];

        // kmer table is incl-prefix histogram, thus my_kmer_l is not the start global address
        // instead the start global address should be the previous kmer count (ie, tl[my_kmer_l-1])
        // the exclusive right global index of my global range should then be:
        // my_kmer_r-1
        d.my_begin = (my_kmer_l == 0) ? 0 : table[my_kmer_l-1];
        d.my_end = (my_kmer_r == 0) ? 0 : table[my_kmer_r-1];
        d.local_size = d.my_end - d.my_begin;

#ifndef NDEBUG
        d.check_correct(comm);
#endif
        t.end_section("repartition: 1D-partition");
        return d;
    }
};
/**
 * @brief Distributed Enhanced Suffix Array
 *
 * @tparam index_t  Index type used for SA, RMQ, TL-lookup-table.
 */
template <typename index_t, typename TLI = tllt<index_t>>
struct dist_desa {
    /// Distributed Suffix Array & LCP array, and L_c array
    suffix_array<char, index_t, true, true> sa;

    /// LCP iterator type for RMQ
    using it_t = typename std::vector<index_t>::const_iterator;

    /// RMQ of local LCP
    rmq<it_t, index_t> minq;

    /// Top-Level lookup table (shared copy)
    //lookup_index<index_t> lt; // shared kmer lookup table
    TLI tli;

    /// global size of input and arrays
    size_t n;

    // save the string
    std::string local_str;
    // block distribution of string
    mxx::blk_dist str_dist;

    // distribution of subtrees (general distribution of global range)
    gen_dist subtree_dist;

    /// type of query results: typeof [l,r)
    using range_t = std::pair<index_t,index_t>;

    /// creates an empty DESA
    dist_desa(const mxx::comm& c) : sa(c) {}

    /// naive implementation of L_c construciton
    // (used only for runtime comparison with the better algorithm)
#if DESA_CONSTR_NAIVE_LC
    std::vector<char> local_Lc;

    void naive_construct_Lc() {
        size_t local_size = sa.local_SA.size();
        sa.comm.with_subset(sa.local_SA.size() > 0, [&](const mxx::comm& comm) {
        // Note:
        //  LCP[i] = lcp(SA[i-1],SA[i])
        //  Lc[i]  = S[SA[i-1]+LCP[i]], i=1,...n-1
        index_t prev_SA = mxx::right_shift(sa.local_SA.back(), comm);

        mxx::blk_dist dist(sa.n, comm.size(), comm.rank());
        MXX_ASSERT(dist.local_size() == local_str.size());

        std::vector<size_t> counts(comm.size(), 0);
        if (comm.rank() > 0) {
            if (prev_SA + sa.local_LCP[0] < sa.n) {
                counts[dist.rank_of(prev_SA + sa.local_LCP[0])]++;
            }
        }
        for (size_t i = 0; i < local_size; ++i) {
            if (sa.local_SA[i-1] + sa.local_LCP[i] < sa.n) {
                counts[dist.rank_of(sa.local_SA[i-1] + sa.local_LCP[i])]++;
            }
        }
        std::vector<size_t> offsets = mxx::local_exscan(counts);
        size_t total_count = std::accumulate(counts.begin(), counts.end(), static_cast<size_t>(0));

        std::vector<size_t> charidx(total_count);
        // add bucketed requests
        if (comm.rank() > 0) {
            if (prev_SA + sa.local_LCP[0] < sa.n) {
                charidx[offsets[dist.rank_of(prev_SA + sa.local_LCP[0])]++] = prev_SA + sa.local_LCP[0];
            }
        }
        for (size_t i = 0; i < local_size; ++i) {
            if (sa.local_SA[i-1] + sa.local_LCP[i] < sa.n) {
                charidx[offsets[dist.rank_of(sa.local_SA[i-1] + sa.local_LCP[i])]++] = sa.local_SA[i-1] + sa.local_LCP[i];
            }
        }

        std::vector<char> resp_chars = bulk_rma(local_str.begin(), local_str.end(), charidx, counts, comm);
        charidx.clear();

        offsets = mxx::local_exscan(counts);

        local_Lc.resize(local_size);

        // add bucketed requests
        if (comm.rank() > 0) {
            if (prev_SA + sa.local_LCP[0] < sa.n) {
                local_Lc[0] = resp_chars[offsets[dist.rank_of(prev_SA + sa.local_LCP[0])]++];
            }
        }
        for (size_t i = 0; i < local_size; ++i) {
            if (sa.local_SA[i-1] + sa.local_LCP[i] < sa.n) {
                local_Lc[i] = resp_chars[offsets[dist.rank_of(sa.local_SA[i-1] + sa.local_LCP[i])]++];
            }
        }
        });
    }
#endif

    void repartition(const mxx::comm& comm) {
        mxx::section_timer t;
        redistr_sa(sa, subtree_dist.my_begin, subtree_dist.my_end, comm);
        t.end_section("repartition: redistribute");
    }

    /**
     * @brief   Constructs the distributed DESA given the block
     *          distributed string given by [begin,end).
     */
    template <typename Iterator>
    void construct(Iterator begin, Iterator end, const mxx::comm& comm) {
        mxx::section_timer t;
        /// we need a copy of the input string for aligning queries
        local_str = std::string(begin, end); // copy input string into data structure
        n = mxx::allreduce(local_str.size(), comm);
        str_dist = mxx::blk_dist(n, comm.size(), comm.rank());
        t.end_section("desa_construct: cpy str");

        // create SA/LCP
        sa.construct(local_str.begin(), local_str.end()); // move comm from constructor into .construct !?
        MXX_ASSERT(sa.local_SA.size() == str_dist.local_size());
        t.end_section("desa_construct: SA/LCP construct");

        tli.construct(sa, local_str, comm);
        t.end_section("desa_construct: TLI construct");

        subtree_dist = gen_dist::from_prefix_sizes(tli.prefix(), comm);
        t.end_section("desa_construct: 1-D partition");
        repartition(comm);
        subtree_dist.print_imbalance_stats(comm);
        t.end_section("desa_construct: repartition");

        // TODO: LCP might need some 1 element overlaps on boundaries (or different distribution??)
        // construct RMQ on local LCP
        if (subtree_dist.local_size > 0) {
            minq = rmq<it_t,index_t>(sa.local_LCP.begin(), sa.local_LCP.end());
        }
        t.end_section("desa_construct: RMQ construct");

#if DESA_CONSTR_NAIVE_LC
        naive_construct_Lc();
        t.end_section("desa_construct: Lc naive construct");
#endif
    }

    /// write distributed DESA to files using parallel IO
    void write(const std::string& basename, const mxx::comm& comm) {
        // writes only the SA, the rest gets re-created from the SA and input string
        sa.write(basename);
    }

    /// read and initialize the distributed DESA from file
    void read(const std::string& string_file, const std::string& basename, const mxx::comm& comm) {
        mxx::section_timer t;
        // read input string
        local_str = mxx::file_block_decompose(string_file.c_str(), comm);
        n = mxx::allreduce(local_str.size(), comm);
        str_dist = mxx::blk_dist(n, comm.size(), comm.rank());
        t.end_section("desa_read: read str");

        // read SA
        sa.read(basename);
        t.end_section("desa_read: read SA");

        tli.construct(sa, local_str, comm);
        t.end_section("desa_read: init q-mer hist");

        subtree_dist = gen_dist::from_prefix_sizes(tli.prefix(), comm);
        repartition(comm);
        subtree_dist.print_imbalance_stats(comm);
        t.end_section("desa_read: repartition");

        // construct RMQ(LCP)
        if (subtree_dist.local_size > 0) {
            minq = rmq<it_t,index_t>(sa.local_LCP.begin(), sa.local_LCP.end());
        }
        t.end_section("desa_read: RMQ construct");
    }

    // a ST node is virtually represented by it's interval [l,r] and it's first
    // child split point `i1`, where LCP[i1] = minLCP[l..r] is the string
    // depths `q` of the node. `c` is P[q], the (q+1)th char in P
    inline void find_child(size_t& l, size_t& i1, size_t& r, size_t& q, char c) {
        assert(l < r);
        assert(l <= i1);
        assert(i1 <= r);
        do {
            // `i` is the lcp(SA[i-1],SA[i])
            char lc = this->sa.local_Lc[i1]; // == S[SA[l]+lcpv] for first iter
            if (lc == c) {
                r = i1-1;
                break;
            }
            l = i1;
            if (l == r)
                break;

            i1 = this->minq(l+1, r);
        } while (l < r && sa.local_LCP[i1] == q);

        if (sa.local_LCP[i1] == q) {
            if (l+1 < r) {
                i1 = this->minq(l+1, r);
            } else {
                i1 = l;
            }
        }
        q = sa.local_LCP[i1];
    }

#if 0
    template <typename String>
    inline range_t local_locate_possible(const String& P, size_t l, size_t r) {
        assert(my_begin <= l && r < my_end);
        size_t m = P.size();

        // convert to local coords
        l -= my_begin;
        r -= my_begin;

        if (P.size() > lt.k && l <= r) {
            // further narrow down search space
            if (l < r) {
                size_t i = this->minq(l+1, r);
                size_t q = sa.local_LCP[i];
                assert(q >= lt.k);

                // FIXME: the check for l < i shouldn't be necessary, but
                // somehow it happens sometimes and leads to error in find_child
                while (q < m && l < r && l < i) {
                    this->find_child(l, i, r, q, P[q]);
                }
            }

        }
        return range_t(l+my_begin, r+1+my_begin);
    }
#else 

    // manually in-lined version
    template <typename String>
    inline range_t local_locate_possible(const String& P, size_t l, size_t r) {
        assert(subtree_dist.my_begin <= l && r < subtree_dist.my_end);
        size_t m = P.size();

        // convert to local coords
        l -= subtree_dist.my_begin;
        r -= subtree_dist.my_begin;
        if (P.size() > tli.minmatch() && l <= r) {
            // further narrow down search space
            if (l < r) {

                size_t i = this->minq(l+1, r);
                size_t q = sa.local_LCP[i];

                while (q < m && l < r && l < i) {

                    // NOTE: LCP[i] = lcp(SA[i-1],SA[i]), LCP[0] = 0
                    // using [l,r] as an inclusive SA range
                    // corresponding to LCP query range [l+1,r]

                    // check if we've reached the end of the pattern
                    if (q >= m) {
                        break;
                    }

                    char c = P[q];
                    do {
                        // `i` is the lcp(SA[i-1],SA[i])
                        char lc = sa.local_Lc[i]; // == S[SA[l]+lcpv] for first iter
                        if (lc == c) {
                            r = i-1;
                            break;
                        }
                        l = i;
                        if (l == r)
                            break;
                        i = this->minq(l+1, r);
                    } while (l < r && sa.local_LCP[i] == q);

                    if (sa.local_LCP[i] == q) {
                        if (l+1 < r) {
                            i = this->minq(l+1, r);
                        } else {
                            i = l;
                        }
                    }
                    q = sa.local_LCP[i];
                }
            }
        }
        return range_t(l+subtree_dist.my_begin, r+1+subtree_dist.my_begin);
    }

#endif

    template <typename String>
    range_t local_locate_possible(const String& P) {
        // only if `P` is on local processor
        index_t l, r;
        std::tie(l, r) = tli.lookup(P);
        if (l == r) {
            return range_t(l,l);
        }
        --r; // convert [l,r) to [l,r]

        return local_locate_possible(P, l, r);
    }

    /// execute in parallel (each processor checks the lookup table) and
    /// proceeds only if it's the owner of the pattern
    template <typename String>
    range_t locate_possible(const String& P) {
        // only if `P` is on local processor
        index_t l, r;
        std::tie(l, r) = tli.lookup(P);
        if (l == r) {
            return range_t(l,l);
        }
        --r; // convert [l,r) to [l,r]

        if (P.size() <= tli.minmatch()) {
            return range_t(l,r+1);
        }

        range_t result;
        int owner = -1;
        if (subtree_dist.my_begin <= l && r < subtree_dist.my_end) {
            result = local_locate_possible(P, l, r);
            owner = sa.comm.rank();
        }
        owner = mxx::allreduce(owner, mxx::max<int>(), sa.comm);
        mxx::bcast(result, owner, sa.comm);

        return result;
    }

    std::vector<range_t> bulk_locate(const strings& ss) {
        mxx::section_timer timer(std::cerr, sa.comm);
        size_t nstrings = ss.nstrings;
        std::vector<size_t> send_data_sizes(sa.comm.size()); // per processor sum of string sizes

        std::vector<int> sprocs(nstrings, -1);
        std::vector<range_t> solutions(nstrings);

        std::vector<size_t> send_counts(sa.comm.size());

        for (size_t i = 0; i < ss.nstrings; ++i) {
            mystring s;
            s.ptr = ss.str_begins[i];
            s.len = ss.str_lens[i];
            range_t res = tli.lookup(s);
            // get processor for this range
            int sproc = subtree_dist.target_processor(res.first);

            if (s.len > tli.minmatch() && res.second > res.first) {
                // this one needs further consideration
                if (sproc != sa.comm.rank()) {
                    assert(res.second <= subtree_dist.my_begin || res.first >= subtree_dist.my_end);
                    sprocs[i] = sproc;
                    ++send_counts[sproc];
                } else {
                    assert(subtree_dist.my_begin <= res.first && res.second <= subtree_dist.my_end);
                    // can be solved locally (do these while sending?)

                    if (res.first < res.second) {
                        // local query
                        res = local_locate_possible(s, res.first, res.second-1);
                    }
                    solutions[i] = res;
                }
            } else {
                // solution is known
                // save soluton, don't send this one
                solutions[i] = res;
            }
        }

        timer.end_section("Phase I: local queries");

        // communicate the patterns to where they can be answered
        strings recv_ss = all2all_strings(ss, sprocs, send_counts, sa.comm);
        timer.end_section("Phase I: all2all patterns");

        std::vector<range_t> recv_res(recv_ss.nstrings);
        // locally continue querying the received strings
        for (size_t i = 0; i < recv_ss.nstrings; ++i) {
            // need to lookup again, then query
            mystring P;
            P.ptr = recv_ss.str_begins[i];
            P.len = recv_ss.str_lens[i];

            range_t res = tli.lookup(P);
            assert(subtree_dist.my_begin <= res.first && res.second <= subtree_dist.my_end);
            assert(P.len > tli.minmatch());

            if (res.first < res.second) {
                // local query
                res = local_locate_possible(P, res.first, res.second-1);
            }
            recv_res[i] = res;
        }

        timer.end_section("Phase II: local_locate_possible");

        // rule out false positives
        // - take first or first few !?, sent to SA[first] for string alignment and checks
        // - string is still equally block distributed!
        //std::vector<size_t> sa_idx;
        // steps:
        // - bucket patterns and results by rank_of(SA[l])
        // - all2all of patterns, results, SA[l], and originating processor?
        // - local string compare (might cross boundaries!) [for now assume single boundary?]
        // - reverse all all2alls?
        std::vector<int> rank_sa(recv_ss.nstrings);
        std::vector<size_t> fp_send_counts(sa.comm.size());
        for (size_t i = 0; i < recv_ss.nstrings; ++i) {
            size_t stridx = sa.local_SA[recv_res[i].first - subtree_dist.my_begin];
            rank_sa[i] = str_dist.rank_of(stridx);
            fp_send_counts[rank_sa[i]]++;
        }
        std::vector<size_t> stridxs(recv_ss.nstrings);
        std::vector<size_t> fp_offset = mxx::local_exscan(fp_send_counts);
        for (size_t i = 0; i < recv_ss.nstrings; ++i) {
            stridxs[fp_offset[rank_sa[i]]++] = sa.local_SA[recv_res[i].first - subtree_dist.my_begin];
        }

        std::vector<size_t> fp_recv_counts = mxx::all2all(fp_send_counts, sa.comm);
        stridxs = mxx::all2allv(stridxs, fp_send_counts, sa.comm);
        strings fp_ss = all2all_strings(recv_ss, rank_sa, fp_send_counts, sa.comm);

        timer.end_section("all2all patterns for cmp");


        std::vector<size_t> overlap_sidx;
        std::vector<size_t> overlap_strs;
        // strcmp the patterns to the stridxs in the underlying string data
        //
        std::vector<int> fp_cmp(stridxs.size());
        for (size_t i = 0; i < stridxs.size(); ++i) {
            bool match = true;
            mystring P;
            P.ptr = fp_ss.str_begins[i];
            P.len = fp_ss.str_lens[i];
            // compare P with `stridxs`
            size_t cmp_len = std::min(str_dist.iprefix_size() - stridxs[i], P.len);
            for (size_t j = 0; j < cmp_len; ++j) {
                match = match && (local_str[stridxs[i] - str_dist.eprefix_size()+j] == P.ptr[j]);
            }
            if (match && cmp_len < P.len) {
                overlap_sidx.emplace_back(stridxs[i]);
                overlap_strs.emplace_back(i);
            }
            fp_cmp[i] = match;
        }

        timer.end_section("Phase III: local strcmp");


        // TODO: right shift those in overlap_xxx and cmp on next processor

        // return results all the way back to originating processor
        fp_cmp = mxx::all2allv(fp_cmp, fp_recv_counts, fp_send_counts, sa.comm);
        fp_offset = mxx::local_exscan(fp_send_counts);
        for (size_t i = 0; i < recv_ss.nstrings; ++i) {
            if (!fp_cmp[fp_offset[rank_sa[i]]++]) {
                recv_res[i] = range_t(recv_res[i].first, recv_res[i].first);
            }
        }

        timer.end_section("return P3 results");


        /// Phase IV: send shit back to its origin

        // TODO: re-use this from within the all2all_strings
        std::vector<size_t> recv_counts = mxx::all2all(send_counts, sa.comm);

        // send back results
        std::vector<range_t> ret_res = mxx::all2allv(recv_res, recv_counts, sa.comm);

        // iterate through the original strings and add the results in correct place
        std::vector<size_t> offset = mxx::local_exscan(send_counts);
        for (size_t i = 0; i < ss.nstrings; ++i) {
            if (sprocs[i] >= 0) {
                // save received solutions
                solutions[i] = ret_res[offset[sprocs[i]]++];
            }
        }

        timer.end_section("return P2 results");

        return solutions;
    }
};

#endif // DESA_HPP
