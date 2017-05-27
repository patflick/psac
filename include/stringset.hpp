/*
 * Copyright 2016 Georgia Institute of Technology
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

#ifndef STRINGSET_HPP
#define STRINGSET_HPP

#include <string>
#include <vector>
#include <algorithm>

#include <cxx-prettyprint/prettyprint.hpp>

#include <mxx/comm.hpp>
#include "shifting.hpp"
#include "bulk_rma.hpp"

// distributed stringset with strings split across boundaries
// and each string not necessarily starting in memory right after the previous
// each string is represented as ptr + size
class simple_dstringset {
public:
    bool first_split, last_split;
    size_t left_size, right_size;
    //const char* data_begin, data_end;
    std::vector<const char*> str_begins;
    std::vector<size_t> sizes;
    size_t sum_sizes;

    template <typename Iterator>
    void parse(Iterator begin, Iterator end, const mxx::comm& comm) {
        size_t local_size = std::distance(begin, end);

        assert(local_size > 0);
        char left_char = mxx::right_shift(*(end-1), comm);
        char right_char = mxx::left_shift(*begin, comm);
        sum_sizes = 0;

        // parse
        Iterator it = begin;
        // skip over separator characters
        while(it != end && *it == '$')
            ++it;
        while (it != end) {
            // find end of string
            Iterator e = it;
            while (e != end && *e != '$')
                ++e;

            // found valid substring [it, e)
            sizes.emplace_back(std::distance(it, e));
            sum_sizes += sizes.back();
            str_begins.emplace_back(&(*it));

            // skip over non string chars
            it = e;
            while(it != end && *it == '$')
                ++it;
            // `it` now marks the beginning of the next string (if not ==end).
        }

        if (comm.rank() > 0 && *begin != '$' && left_char != '$') {
            first_split = true;
        }

        if (comm.rank() != comm.size()-1 && *(end-1) != '$' && right_char != '$') {
            last_split = true;
        }
    }

    void get_split_sizes(const mxx::comm& comm) {
        bool any_splits = first_split || last_split;
        any_splits = mxx::any_of(any_splits, comm);

        // get left and right sizes
        std::pair<int, size_t> last_size;
        if (sizes.size() == 1 && last_split && first_split) {
            // left and right extend are same string -> just add mine and don't reset
            last_size.first = -1;
            last_size.second = sizes.back();
        } else {
            // just send size of last one
            last_size.first = comm.rank();
            last_size.second = sizes.back();
        }
        std::pair<int, size_t> left_sums = mxx::exscan(last_size, [](const std::pair<int, size_t>& x, const std::pair<int, size_t>& y){
            std::pair<int, size_t> result;
            if (y.first < x.first) {
                result.first = x.first;
                result.second = x.second + y.second;
            } else {
                result.first = y.first;
                result.second = y.second;
            }
            return result;
        }, comm);

        std::pair<int, size_t> first_size;
        if (sizes.size() == 1 && last_split && first_split) {
            // left and right extend are same string -> just add mine and don't reset
            first_size.first = comm.size();
            first_size.second = sizes.front();
        } else {
            // just send size of last one
            first_size.first = comm.rank();
            first_size.second = sizes.front();
        }

        std::pair<int, size_t> right_sums = mxx::exscan(first_size, [](const std::pair<int, size_t>& x, const std::pair<int, size_t>& y){
            std::pair<int, size_t> result;
            if (y.first > x.first) {
                result.first = x.first;
                result.second = x.second + y.second;
            } else {
                result.first = y.first;
                result.second = y.second;
            }
            return result;
        }, comm.reverse());

        left_size = 0;
        right_size = 0;
        if (first_split) {
            left_size = left_sums.second;
        }
        if (last_split) {
            right_size = right_sums.second;
        }
    }

    // parse!
    template <typename Iterator>
    simple_dstringset(Iterator begin, Iterator end, const mxx::comm& comm)
        : first_split(false), last_split(false), str_begins(), sizes() {
            parse(begin, end, comm);
            get_split_sizes(comm);
    }
};


// overlap type:    0: no overlaps, 1: left overlap, 2:right overlap,
//                  3: separate overlaps on left and right
//                  4: contiguous overlap with both sides
int get_schedule(int overlap_type, const mxx::comm& comm) {
    // if we have left/right/both/or double buckets, do global comm in two phases
    int my_schedule = -1;

    // gather all types to first processor
    std::vector<int> overlaps = mxx::gather(overlap_type, 0, comm);
    if (comm.rank() == 0) {
        // create schedule using linear scan over the overlap types
        std::vector<int> schedule(comm.size());
        int phase = 0; // start in first phase
        for (int i = 0; i < comm.size(); ++i) {
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
        //MPI_Scatter(&schedule[0], 1, MPI_INT, &my_schedule, 1, MPI_INT, 0, comm);

        my_schedule = mxx::scatter_one(schedule, 0, comm);
    } else {
        my_schedule = mxx::scatter_one_recv<int>(0, comm);
    }
    return my_schedule;
}




class dist_seqs_base {

public:
    // inner range [first_sep, last_sep)
    /// whether there are sequence separators on this processor
    bool has_local_seps;
    size_t first_sep;
    size_t last_sep;

    /// possibly remote sequence separators for subsequences which have
    /// elements on this processor but also on other processors
    bool is_init_splits;
    size_t left_sep;
    int left_sep_rank;
    size_t right_sep;
    int right_sep_rank;

    /// !collective
    /// Given that the first and last separators are set, this initializes the
    template <typename Dist>
    void init_split_sequences(Dist dist, const mxx::comm& comm) {
        if (!has_local_seps) {
            first_sep = std::numeric_limits<size_t>::max();
            last_sep = 0;
            if (comm.rank() == comm.size() - 1) {
                first_sep = dist.global_size();
            }
        }

        auto maxpair = [](const std::pair<size_t, int>& x, const std::pair<size_t, int>& y) {
            return (x.first < y.first) ? y : x;
        };
        std::tie(left_sep,left_sep_rank) = mxx::exscan(std::make_pair(last_sep, comm.rank()), maxpair, comm);
        auto minpair = [](const std::pair<size_t, int>& x, const std::pair<size_t, int>& y) {
            return (x.first > y.first) ? y : x;
        };
        std::tie(right_sep,right_sep_rank) = mxx::exscan(std::make_pair(first_sep, comm.rank()), minpair, comm.reverse());
        if (comm.rank() == comm.size() - 1) {
            //right_sep = dist.iprefix();
            right_sep = dist.iprefix_size();
            right_sep_rank = comm.rank();
        }
        if (right_sep == dist.iprefix_size()) {
            last_sep = right_sep;
        }
        if (comm.rank() == 0) {
            first_sep = 0;
        }
        if (first_sep == dist.eprefix_size()) {
            left_sep = first_sep;
            left_sep_rank = comm.rank();
        }
        is_init_splits = true;
    }

    bool is_left_split() const {
        return left_sep < first_sep;
    }

    bool is_right_split() const {
        return last_sep < right_sep;
    }

    /// returns whether any subsequence is split across processor boundaries
    /// either to the left, the right, or both
    bool has_split_seqs() const {
        if (has_local_seps) {
            return left_sep < first_sep || last_sep < right_sep;
        } else {
            return true;
        }
    }

    /// returns whether this processor has any subsequences that lie
    /// exclusively on this processor
    bool has_inner_seqs() const {
        if (has_local_seps) {
            return first_sep < last_sep;
        } else {
            return false;
        }
    }

    /// returns all those subsequences which are split across processor
    /// boundaries (not fully contained on this processor)
    /// Each of those subsequences is represented by their half-open
    /// global-index range [gidx_begin, gidx_end) returned in the form of a
    /// std::pair
    std::vector<std::pair<size_t, size_t>> split_seqs() const {
        std::vector<std::pair<size_t, size_t>> result;
        if (has_local_seps) {
            if (left_sep < first_sep) {
                result.emplace_back(left_sep, first_sep);
            }
            if (last_sep < right_sep) {
                result.emplace_back(last_sep, right_sep);
            }
        } else {
            result.emplace_back(left_sep, right_sep);
        }
        return result;
    }


    std::pair<size_t, size_t> inner_seqs_range() const {
        if (has_local_seps) {
            return std::pair<size_t, size_t>(first_sep, last_sep);
        } else {
            return std::pair<size_t, size_t>(0, 0);
        }
    }

    template <typename Func>
    void for_each_split_seq_2phase(const mxx::comm& comm, Func func) const {

        // overlap type:    0: no overlaps, 1: left overlap, 2:right overlap,
        //                  3: separate overlaps on left and right
        //                  4: contiguous overlap with both sides
        int overlap_type = 0;
        if (has_local_seps) {
            if (left_sep < first_sep)
                overlap_type += 1;
            if (last_sep < right_sep)
                overlap_type += 2;
        } else {
            overlap_type = 4;
        }

        // if there are no overlaps at all, skip!
        if (mxx::all_of(overlap_type == 0, comm))
            return;

        // create schedule of two phases to process all split sequences
        int my_schedule = get_schedule(overlap_type, comm);

        // execute two phases separately, synchronized by a barrier
        for (int phase = 0; phase <= 1; ++phase) {
            // the leftmost processor of a group will be used as split
            bool participate = overlap_type == 3 || (overlap_type != 0 && my_schedule == phase);

            int left_p;
            size_t begin, end;
            if ((my_schedule != phase && overlap_type == 3) || (my_schedule == phase && overlap_type == 2)) {
                // right bucket
                begin = last_sep;
                end = right_sep;
                left_p = comm.rank();
            } else if (my_schedule == phase && overlap_type == 4) {
                begin = left_sep;
                end = right_sep;
                left_p = left_sep_rank;
            } else if (my_schedule == phase && (overlap_type == 1 || overlap_type == 3)) {
                begin = left_sep;
                end = first_sep;
                left_p = left_sep_rank;
            }

            comm.with_subset(participate,[&](const mxx::comm& sc) {
                // split communicator to `left_p`
                mxx::comm subcomm = sc.split(left_p);
                func(begin, end, subcomm);
            });

            comm.barrier();
        }
    }
};


// equally distributed prefix sizes
// with shadow elements for left and right processor boundaries
struct dist_seqs : public dist_seqs_base {
    mxx::blk_dist part;
    size_t global_size;
    std::vector<size_t> prefix_sizes;
    //bool shadow_initialized;

    void init_from_dss(const simple_dstringset& dss, const mxx::comm& comm) {
        // input distributed stringset might not be (equally) block distributed
        // with regards to character count. Thus we redistribute prefix_size
        // seqeuences so that they are
        size_t ss_local_size = std::accumulate(dss.sizes.begin(), dss.sizes.end(), static_cast<size_t>(0));
        size_t ss_global_size = mxx::allreduce(ss_local_size, comm);
        size_t ss_prefix = mxx::exscan(ss_local_size, comm);

        part = mxx::blk_dist(ss_global_size, comm.size(), comm.rank());
        global_size = ss_global_size;

        std::vector<size_t> send_counts(comm.size(), 0);
        // for all sequence starts: ps[i] = gidx_sum[i-1] + size[i]
        std::vector<size_t> gidx;
        gidx.reserve(dss.sizes.size());
        size_t size_sum = ss_prefix;
        int pi;
        if (!dss.first_split) {
            pi = part.rank_of(ss_prefix);
            ++send_counts[pi];
            gidx.emplace_back(size_sum);
        } else {
            pi = part.rank_of(ss_prefix+dss.sizes[0]);
        }
        size_t pi_end = part.iprefix_size(pi);

        // create prefix sums and keep track the processor id for their target
        for (size_t i = 0; i < dss.sizes.size()-1; ++i) {
            size_sum += dss.sizes[i];
            while (size_sum >= pi_end) {
                ++pi;
                pi_end = part.iprefix_size(pi);
            }
            gidx.emplace_back(size_sum);
            ++send_counts[pi];
        }

        // XXX: possibly optimize this communication (expected very low volume,
        //      and mostly with direct neighbors)
        prefix_sizes = mxx::all2allv(gidx, send_counts, comm);
    }

    static dist_seqs from_dss(const simple_dstringset& dss, const mxx::comm& comm) {
        dist_seqs res;
        res.init_from_dss(dss, comm);
        if (!res.prefix_sizes.empty()) {
            res.first_sep = res.prefix_sizes.front();
            res.last_sep = res.prefix_sizes.back();
            res.has_local_seps = true;
        } else {
            res.has_local_seps = false;
        }
        res.init_split_sequences(res.part, comm);
        return res;
    }

    // calls the given function for each sequence on this processor, by passing
    // the global start and end indexes as the two parameters for all sequences
    // which have at least on element on this processor
    template <typename Func>
    void for_each_seq(Func f) const {
        if (has_local_seps) {
            if (left_sep < first_sep) {
                f(left_sep, first_sep);
            }
            for (size_t i = 1; i < prefix_sizes.size(); ++i) {
                f(prefix_sizes[i-1], prefix_sizes[i]);
            }
            if (prefix_sizes.back() < last_sep) {
                f(prefix_sizes.back(), last_sep);
            }
            if (last_sep < right_sep) {
                f(last_sep, right_sep);
            }
        } else {
            f(left_sep, right_sep);
        }
    }

    // returns the size of the subsequences starting (owned by) this processor
    std::vector<size_t> sizes() const {
        std::vector<size_t> result(prefix_sizes.size());
        for (size_t i = 0; i < prefix_sizes.size(); ++i) {
            if (i+1 < prefix_sizes.size()) {
                result[i] = prefix_sizes[i+1] - prefix_sizes[i];
            } else {
                result[i] = right_sep - prefix_sizes[i];
            }
        }
        return result;
    }

};

struct dist_seqs_buckets : public dist_seqs_base {
    mxx::blk_dist part;
    size_t global_size;
    bool has_local_els;

    template <typename T, typename Func = std::equal_to<T>>
    static dist_seqs_buckets from_func(const std::vector<T>& seq, const mxx::comm& comm, Func f = std::equal_to<T>()) {
        assert(seq.size() > 0);
        // init size and distribution
        dist_seqs_buckets d;
        d.has_local_els = seq.size() > 0;
        d.global_size = mxx::allreduce(seq.size(), comm);
        d.part = mxx::blk_dist(d.global_size, comm.size(), comm.rank());

        // set these three:
        T prev = mxx::right_shift(seq.back(), comm);
        d.has_local_seps = !f(seq.front(), seq.back()) || (!comm.is_first() && !f(prev, seq.front()));
        if (d.has_local_seps) {
            // find first
            if (comm.is_first() || !f(prev, seq.front())) {
                d.first_sep = d.part.eprefix_size();
            } else {
                size_t i = 0;
                while (i+1 < seq.size() && f(seq[i], seq[i+1]))
                    ++i;
                d.first_sep = i+1 + d.part.eprefix_size();
            }

            // find first entry of sequence equal to last element
            size_t i = seq.size()-1;
            while (i > 0 && f(seq[i-1],seq[i]))
                --i;
            d.last_sep = i + d.part.eprefix_size();
        }
        d.init_split_sequences(d.part, comm);

        return d;
    }


};

std::ostream& operator<<(std::ostream& os, const dist_seqs& ds) {
    return os << "(" << ds.left_sep << "), " << ds.prefix_sizes << ", (" << ds.right_sep << ")";
}

std::ostream& operator<<(std::ostream& os, const dist_seqs_buckets& ds) {
    if (ds.has_local_seps)
        return os << "(" << ds.left_sep << "@" << ds.left_sep_rank << "), [" << ds.first_sep << ",...," << ds.last_sep << "), (" << ds.right_sep << "@" << ds.right_sep_rank << ")";
    else
        return os << "(" << ds.left_sep << "@" << ds.left_sep_rank << "), [], (" << ds.right_sep << "@" << ds.right_sep_rank << ")";
}


/*
 * bulk queries on dist_seqs
 */

template <typename Func>
std::vector<typename std::result_of<Func(size_t,size_t,size_t)>::type>
local_ds_rma(const dist_seqs& ds, const std::vector<size_t>& local_queries, Func query) {
    //
    using T = typename std::result_of<Func(size_t,size_t,size_t)>::type;
    // argsort the local_queries
    std::vector<size_t> argsort(local_queries.size());
    std::iota(argsort.begin(), argsort.end(), 0);
    std::sort(argsort.begin(), argsort.end(), [&local_queries](size_t i, size_t j) { return local_queries[i] < local_queries[j]; });

    std::vector<T> results(local_queries.size());
    size_t i = 0;
    // linear in number of local strings + number of queries
    ds.for_each_seq([&](size_t str_beg, size_t str_end) {
        while (i < local_queries.size() && local_queries[argsort[i]] < str_end) {
            size_t qi = argsort[i];
            results[qi] = query(local_queries[qi], str_beg, str_end);
            ++i;
        }
    });
    return results;
}

template <typename T, typename Func>
std::vector<T> bulk_query_ds(const dist_seqs& ds, const std::vector<T>& vec, const std::vector<size_t>& rma_reqs, const mxx::comm& comm, Func query) {
    size_t local_size = vec.size();
    size_t global_size = mxx::allreduce(local_size, comm);
    mxx::blk_dist dist(global_size, comm.size(), comm.rank());

    std::vector<size_t> original_pos;
    std::vector<size_t> bucketed_rma;
    std::vector<size_t> send_counts = idxbucketing(rma_reqs, [&dist](size_t gidx) { return dist.rank_of(gidx); }, comm.size(), bucketed_rma, original_pos);

    std::vector<size_t> recv_counts = mxx::all2all(send_counts, comm);

    // send all queries via all2all
    std::vector<size_t> local_queries = mxx::all2allv(bucketed_rma, send_counts, recv_counts, comm);

    std::vector<T> results = local_ds_rma(ds, local_queries, query);
    results = mxx::all2allv(results, recv_counts, send_counts, comm);

    std::vector<T> rma_b2 = permute(results, original_pos);
    return rma_b2;
}



std::string flatten_strings(const std::vector<std::string>& v, const char sep = '$') {
    std::string result;
    size_t outsize = 0;
    for (auto s : v) {
        outsize += s.size() + 1;
    }
    result.resize(outsize);
    auto outit = result.begin();
    for (auto s : v) {
        outit = std::copy(s.begin(), s.end(), outit);
        *outit = sep;
        ++outit;
    }
    return result;
}

template <typename T>
std::vector<std::vector<T>> gather_dist_seq(const dist_seqs& ds, const std::vector<T>& vec, const mxx::comm& comm) {
    // gather sizes of subsequences
    std::vector<size_t> allsizes = mxx::gatherv(ds.sizes(), 0, comm);

    // gather whole sequence to rank 0
    std::vector<T> allvec = mxx::gatherv(vec, 0, comm);

    // create the vectors per sequence
    std::vector<std::vector<T>> result(allsizes.size());
    auto it = allvec.begin();
    for (size_t i = 0; i < allsizes.size(); ++i) {
        result[i] = std::vector<T>(it, it + allsizes[i]);
        it += allsizes[i];
    }

    return result;
}


// example class for string sets
// required operations
class vstringset {
public:
    using iterator = std::vector<std::string>::iterator;
    using const_iterator = std::vector<std::string>::const_iterator;

private:
    std::vector<std::string> data;
    size_t total_chars;

    void init_sizes() {
        total_chars = 0;
        for (size_t i = 0; i < data.size(); ++i) {
            total_chars += data[i].size();
        }
    }

public:

    vstringset(const std::vector<std::string>& strings) : data(strings) {
        init_sizes();
    }

    vstringset(std::vector<std::string>&& strings) : data(std::move(strings)) {
        init_sizes();
    }

    vstringset() = default;
    vstringset(const vstringset&) = default;
    vstringset(vstringset&&) = default;

    // iterators through the strings/sequences, each of which just requires .size(), .begin(), .end() and has a value of some char type
    iterator begin() {
        return data.begin();
    }
    const_iterator begin() const {
        return data.begin();
    }
    iterator end() {
        return data.end();
    }
    const_iterator end() const {
        return data.end();
    }

    // number of strings/sequences
    size_t size() {
        return data.size();
    }

    // whether strings are split accross processor boundaries in distributed stringset
    inline bool is_split() const {
        return false;
    }

    /// number of characters in the stringset on this processor
    size_t total_lengths() const {
        return total_chars;
    }

    // convert between (i,j) <-> gidx
};


#endif // STRINGSET_HPP
