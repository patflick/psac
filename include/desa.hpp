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


template <typename T>
std::vector<T> redistr(const std::vector<T>& v, size_t gbeg, size_t gend, const mxx::comm& comm) {
    assert(gbeg <= gend);
    std::vector<T> res(gend - gbeg);
    // TODO: implememnt more efficient version for this (using send/recv only)
    mxx::redo_arbit_decomposition(v.begin(), v.end(), res.begin(), gend - gbeg, comm);
    return res;
}

template <typename index_t, bool CONSTRUCT_LC>
void redistr_sa(suffix_array<char, index_t, true, CONSTRUCT_LC>& sa, size_t gbeg, size_t gend, const mxx::comm& comm) {
    sa.local_SA = redistr(sa.local_SA, gbeg, gend, comm);
    sa.local_LCP = redistr(sa.local_LCP, gbeg, gend, comm);
    sa.local_B = redistr(sa.local_B, gbeg, gend, comm);
    if (CONSTRUCT_LC) {
        sa.local_Lc = redistr(sa.local_Lc, gbeg, gend, comm);
    }
}

template <typename index_t>
struct dist_desa {
    suffix_array<char, index_t, true, true> sa;
    using it_t = typename std::vector<index_t>::const_iterator;
    rmq<it_t, index_t> minq;
    lookup_index<index_t> lt; // shared kmer lookup table
    std::vector<char> local_Lc;
    std::vector<size_t> part; // [partition] processor -> lt index
    std::map<size_t, int> lt_proc; // map each processor start index -> processor index

    size_t n;

    // TODO: save the string
    std::string local_str;
    size_t str_local_size;
    mxx::blk_dist str_dist;


    size_t my_begin;
    size_t my_end;
    size_t local_size;

    using range_t = std::pair<index_t,index_t>;

    dist_desa(const mxx::comm& c) : sa(c) {}


    void naive_construct_Lc() {
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

        // TODO: sometimes the very first element of the redistributed Lc array
        //       doesn't match with the naive version. This doesn't make a difference
        //       in querying, bc we never need the first character in the local array
        //       (it's getting skipped by the top level table)
        // TODO: eventually figure out why!? not really necessary tho
        /*
        if (sa.local_Lc != local_Lc) {
            //std::cerr << "rank " << comm.rank() << ": local Lc arrays are different" << std::endl;
            fprintf(stderr, "rank %i: local Lc arrays are different!\n", comm.rank()); fflush(stderr);
            if (sa.local_Lc.size() == local_Lc.size()) {
                size_t tot = 0;
                for (size_t i = 0; i < sa.local_Lc.size(); ++i) {
                    if (sa.local_Lc[i] != local_Lc[i]) {
                        if (tot < 5) {
                            fprintf(stderr, "rank %i: sa.lc[%lu]=%c != lc[%lu]=%c\n", comm.rank(), i, sa.local_Lc[i], i, local_Lc[i]); fflush(stdout);
                            //std::cerr << "rank " << comm.rank() <<  ": sa.lc[" << i << "=" << sa.local_Lc[i] << " != " << local_Lc[i] << "=lc[" << i << "]" << std::endl;
                        }
                        ++tot;
                    }
                }
                fprintf(stderr, "rank %i: total diff %lu\n", comm.rank(), tot); fflush(stderr);
                //std::cerr << "rank " << comm.rank() << ": total diff " << tot << std::endl;
            } else {
            std::cerr << "SIZES DON'T MATCH" << std::endl;
            }
        } else {
            std::cerr << "YAYAYAYAYAYAY the Lc array works!!!!!" << std::endl;
        }
        */
    }

    template <typename Iterator>
    void construct(Iterator begin, Iterator end, const mxx::comm& comm) {
        mxx::section_timer t;
        local_str = std::string(begin, end); // copy input string into data structure
        n = mxx::allreduce(local_str.size(), comm);
        str_dist = mxx::blk_dist(n, comm.size(), comm.rank());

        t.end_section("desa_construct: cpy str");

        // create SA/LCP
        sa.construct(local_str.begin(), local_str.end()); // move comm from constructor into .construct !?
        MXX_ASSERT(sa.local_SA.size() == str_dist.local_size());
        t.end_section("desa_construct: SA/LCP construct");

        // choose a "good" `q`
        unsigned int l = sa.alpha.bits_per_char();
        unsigned int log_size = 24; // 2^24 = 16 M (*8 = 128 M)
        unsigned int q = log_size / l;

        // `q` might have to be choosen smaller if the input is very small
        if (std::distance(begin,end) < q) {
            q = std::distance(begin,end);
            q = std::max(q,1u);
        }
        q = mxx::allreduce(q, mxx::min<unsigned int>(), comm);
        if (q < log_size / l && comm.rank() == 0) {
            std::cerr << "[WARNING] q for q-mer partitioning was choosen smaller (q=" << q << " vs opt_q=" << log_size/l << ") due to small local size." << std::endl;
        }

        // construct kmer lookup table
        std::vector<uint64_t> hist = kmer_hist<uint64_t>(begin, end, q, sa.alpha, comm);
        lt.construct_from_hist(hist, q, sa.alpha);
        t.end_section("desa_construct: q-mer hist");

        // partition the lookup index by processors
        part = partition(lt.table, comm.size());
        for (int i = 0; i < comm.size(); ++i) {
            size_t kmer_l = part[i];
            size_t proc_begin = (kmer_l == 0) ? 0 : lt.table[kmer_l-1];
            lt_proc[proc_begin] = i;
        }

        // kmers from: [part[comm.rank()] .. part[comm.rank()+1])
        //

        size_t my_kmer_l = part[comm.rank()];
        size_t my_kmer_r = (comm.rank()+1 == comm.size()) ? lt.table.size() : part[comm.rank()+1];

        // kmer table is incl-prefix histogram, thus my_kmer_l is not the start global address
        // instead the start global address should be the previous kmer count (ie, tl[my_kmer_l-1])
        // the exclusive right global index of my global range should then be:
        // my_kmer_r-1
        my_begin = (my_kmer_l == 0) ? 0 : lt.table[my_kmer_l-1];
        my_end = (my_kmer_r == 0) ? 0 : lt.table[my_kmer_r-1];


#ifndef NDEBUG
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
          assert(my_end == sa.n);
        }
#endif
        t.end_section("desa_construct: partition");


        redistr_sa(sa, my_begin, my_end, comm);
        local_size = my_end - my_begin;
        t.end_section("desa_construct: redistr");

        mxx::sync_cout(comm) << "[Rank " << comm.rank() << "] local_size (after part): " << local_size << std::endl;

        // TODO: LCP might need some 1 element overlaps on boundaries (or different distribution??)
        // construct RMQ on local LCP
        if (local_size > 0) {
            minq = rmq<it_t,index_t>(sa.local_LCP.begin(), sa.local_LCP.end());
        }
        t.end_section("desa_construct: RMQ construct");

        /*
        naive_construct_Lc();
        t.end_section("desa_construct: Lc naive construct");
        */
        local_Lc.swap(sa.local_Lc);
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
            char lc = this->local_Lc[i1]; // == S[SA[l]+lcpv] for first iter
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


    template <typename String>
    range_t local_locate_possible(const String& P, size_t l, size_t r) {
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

    template <typename String>
    range_t local_locate_possible(const String& P) {
        // only if `P` is on local processor
        index_t l, r;
        std::tie(l, r) = lt.lookup(P);
        if (l == r) {
            return range_t(l,l);
        }
        --r; // convert [l,r) to [l,r]

        return local_locate_possible(P, l, r);
    }

        /* TODO:
            // check if pattern matches
            if (l <= r) {
                int cmp = strncmp(&this->strbegin[this->SA[l]], &P[0], P.size());
                if (cmp == 0) {
                    return std::pair<index_t, index_t>(l, r+1);
                } else {
                    // no match
                    return std::pair<index_t,index_t>(l, l);
                }
            }
        */

    /// execute in parallel (each processor checks the lookup table) and
    /// proceeds only if it's the owner of the pattern
    template <typename String>
    range_t locate_possible(const String& P) {
        // only if `P` is on local processor
        index_t l, r;
        std::tie(l, r) = lt.lookup(P);
        if (l == r) {
            return range_t(l,l);
        }
        --r; // convert [l,r) to [l,r]

        if (P.size() <= lt.k) {
            return range_t(l,r+1);
        }

        range_t result;
        int owner = -1;
        if (my_begin <= l && r < my_end) {
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
        //std::vector<index_t> kmer_offset(nstrings);
        std::vector<size_t> send_data_sizes(sa.comm.size()); // per processor sum of string sizes

        std::vector<int> sprocs(nstrings, -1);
        std::vector<range_t> solutions(nstrings);

        std::vector<size_t> send_counts(sa.comm.size());

        for (size_t i = 0; i < ss.nstrings; ++i) {
            mystring s;
            s.ptr = ss.str_begins[i];
            s.len = ss.str_lens[i];
            range_t res = lt.lookup(s);
            // get processor for this range
            auto ub = lt_proc.upper_bound(res.first);
            --ub;
            int sproc = ub->second;

            if (s.len > lt.k && res.second > res.first) {
                // this one needs further consideration
                if (sproc != sa.comm.rank()) {
                    assert(res.second <= my_begin || res.first >= my_end);
                    sprocs[i] = sproc;
                    ++send_counts[sproc];
                } else {
                    assert(my_begin <= res.first && res.second <= my_end);
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

            range_t res = lt.lookup(P);
            assert(my_begin <= res.first && res.second <= my_end);
            assert(P.len > lt.k);

            if (res.first < res.second) {
                // local query
                res = local_locate_possible(P, res.first, res.second-1);
            }
            recv_res[i] = res;
        }

        timer.end_section("Phase II: local_locate_possible");

        // TODO: rule out false positives
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
            size_t stridx = sa.local_SA[recv_res[i].first - my_begin];
            rank_sa[i] = str_dist.rank_of(stridx);
            fp_send_counts[rank_sa[i]]++;
        }
        std::vector<size_t> stridxs(recv_ss.nstrings);
        std::vector<size_t> fp_offset = mxx::local_exscan(fp_send_counts);
        for (size_t i = 0; i < recv_ss.nstrings; ++i) {
            stridxs[fp_offset[rank_sa[i]]++] = sa.local_SA[recv_res[i].first - my_begin];
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
