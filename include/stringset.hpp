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

// generate a distributed string set with splits
std::string random_dstringset(size_t lsize, const mxx::comm& c) {
    std::string local_str;
    local_str.resize(lsize);
    char alpha[] = {'a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', '$'};
    srand(c.rank()*13 + 23);
    std::generate(local_str.begin(), local_str.end(), [&alpha]() {
        return alpha[rand() % sizeof(alpha)];
    });
    return local_str;
}

// distributed stringset with strings split across boundaries
// and each string not necessarily starting in memory right after the previous
// each string is represented as ptr/gidx + size
// where size can be represented as size[i] = prefix_size[i]-prefix_size[i-1]
//
// alternatives for representing split strings:
// bool first_split, etc...
// just last_left etc doesn't work for non-contiguous sequences
// there is some dependency on the parser, eg. it somehow needs to parse across
// processor boundaries to figure out how sequences are split/if theyare split

// create explicit representation from implicit parser, or via specific separating character?
// (corresponding to fasta/fastq?)
//


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

// equal distributed (re-distributed) prefix_size, from there things become simple?
// mxx::stable_distribute_inplace()...
//
//
//

class dist_seqs_base {

public:
    // inner range [first_sep, last_sep)
    /// whether there are sequence separators on this processor
    bool has_local_seps;
    size_t first_sep;
    size_t last_sep;

    /// possibly remove sequence separators for subsequences which have
    /// elements on this processor but also on other processors
    bool is_init_splits;
    size_t left_sep;
    size_t right_sep;

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

        left_sep = mxx::exscan(last_sep, mxx::max<size_t>(), comm);
        right_sep = mxx::exscan(first_sep, mxx::min<size_t>(), comm.reverse());
        if (comm.rank() == comm.size() - 1) {
            //right_sep = dist.iprefix();
            right_sep = dist.prefix_size();
        }
        if (right_sep == dist.prefix_size()) {
            last_sep = right_sep;
        }
        if (comm.rank() == 0) {
            first_sep = 0;
        }
        if (first_sep == dist.excl_prefix_size()) {
            left_sep = first_sep;
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
};


// equally distributed prefix sizes
// with shadow elements for left and right processor boundaries
struct dist_seqs : public dist_seqs_base {
    mxx::partition::block_decomposition_buffered<size_t> part;
    size_t global_size;
    std::vector<size_t> prefix_sizes;
    //bool shadow_initialized;

    void init_from_dss(simple_dstringset& dss, const mxx::comm& comm) {
        // input distributed stringset might not be (equally) block distributed
        // with regards to character count. Thus we redistribute prefix_size
        // seqeuences so that they are
        size_t ss_local_size = std::accumulate(dss.sizes.begin(), dss.sizes.end(), static_cast<size_t>(0));
        size_t ss_global_size = mxx::allreduce(ss_local_size, comm);
        size_t ss_prefix = mxx::exscan(ss_local_size, comm);

        part = mxx::partition::block_decomposition_buffered<size_t>(ss_global_size, comm.size(), comm.rank());
        global_size = ss_global_size;

        std::vector<size_t> send_counts(comm.size(), 0);
        // for all sequence starts: ps[i] = gidx_sum[i-1] + size[i]
        // TODO: do across procs boundaries/split sequences
        std::vector<size_t> gidx;
        gidx.reserve(dss.sizes.size());
        size_t size_sum = ss_prefix;
        int pi;
        if (!dss.first_split) {
            pi = part.target_processor(ss_prefix);
            ++send_counts[pi];
            gidx.emplace_back(size_sum);
        } else {
            pi = part.target_processor(ss_prefix+dss.sizes[0]);
        }
        size_t pi_end = part.prefix_size(pi);

        // create prefix sums and keep track the processor id for their target
        for (size_t i = 0; i < dss.sizes.size()-1; ++i) {
            size_sum += dss.sizes[i];
            while (size_sum >= pi_end) {
                ++pi;
                pi_end = part.prefix_size(pi);
            }
            gidx.emplace_back(size_sum);
            ++send_counts[pi];
        }

        // XXX: possibly optimize this communication (expected very low volume, and only neighbor comm)
        prefix_sizes = mxx::all2allv(gidx, send_counts, comm);
    }

    static dist_seqs from_dss(simple_dstringset& dss, const mxx::comm& comm) {
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

std::ostream& operator<<(std::ostream& os, const dist_seqs& ds) {
    return os << "(" << ds.left_sep << "), " << ds.prefix_sizes << ", (" << ds.right_sep << ")";
}

// for use with mxx::sync_cout
std::ostream& operator<<(std::ostream& os, const simple_dstringset& ss) {
    for (size_t i = 0; i < ss.sizes.size(); ++i) {
        if (!(ss.first_split && i == 0)) {
            os << "\"";
        }
        std::string s(ss.str_begins[i], ss.str_begins[i]+ss.sizes[i]);
        os << s;
        if (!(ss.last_split && i == ss.sizes.size()-1)) {
            os << "\", ";
        }
    }
    os.flush();
    return os;
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
