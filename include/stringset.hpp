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

    template <typename Iterator>
    void parse(Iterator begin, Iterator end, const mxx::comm& comm) {
        size_t local_size = std::distance(begin, end);

        assert(local_size > 0);
        char left_char = mxx::right_shift(*(end-1), comm);
        char right_char = mxx::left_shift(*begin, comm);

        // parse
        Iterator it = begin;
        while (it != end) {
            // find end of string
            Iterator e = it;
            while (e != end && *e != '$')
                ++e;

            // found valid substring [it, e)
            sizes.emplace_back(std::distance(it, e));
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

// TODO: create this representation from the stringset
//       but equally distributed base sequence (buckets)
struct dist_seqs_prefix_sizes {
    mxx::partition::block_decomposition_buffered<size_t> part;
    size_t global_size;
    std::vector<size_t> prefix_sizes;
    bool shadow_initialized;


    void init_from_dss(simple_dstringset& dss, const mxx::comm& comm) {
        size_t ss_local_size = std::accumulate(dss.sizes.begin(), dss.sizes().end(), static_cast<size_t>(0));
        size_t ss_global_size = mxx::allreduce(ss_local_size, comm);

        part = mxx::partition::block_decomposition_buffered(ss_global_size, comm.size(), comm.rank());
        
    }

    dist_seqs_prefix_sizes(simple_dstringset& dss) {
    }
};


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
    return os;
}



/*
class ds_stringset : public dist_seqs_base<blk_dist> {

public:
    std::string local_str;
    std::vector<size_t> seq_seps;

    template <typename Iterator>
    static std::vector<size_t> parse_seps(Iterator begin, Iterator end, char sep, const mxx::comm& comm) {
        std::vector<size_t> seps;

        size_t local_size = std::distance(begin, end);
        size_t prefix = mxx::exscan(local_size, comm);
        // parse string, and insert seprator for each string separator character
        size_t gidx = prefix;
        for (Iterator it = begin; it != end; ++it, ++gidx) {
            if (*it == sep) {
                seps.push_back(gidx+1);
            }
        }
        return seps;
    }

    ds_stringset(const std::string& lstr, std::vector<size_t>&& seps, const mxx::comm& c) : dist_seqs_base(blk_dist(c, lstr.size()), seps), local_str(lstr), seq_seps(seps) {
    }

    ds_stringset(const std::string& lstr, const mxx::comm& c) : ds_stringset(lstr, parse_seps(lstr.begin(), lstr.end(), '$', c), c) {}
};
*/


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
