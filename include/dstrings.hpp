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

#ifndef DSTRINGS_HPP
#define DSTRINGS_HPP

/**
 * This file implements a simple distributed stringset, used mainly for
 * loading query-patterns.
 * In this representation, strings are split between processors, such
 * that each stirng is fully contained within a processor.
 */


#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <memory>

#include <mxx/comm.hpp>
#include <mxx/file.hpp>
#include <mxx/stream.hpp>

/**
 * @brief A simple string representation that does not own its own storage.
 *
 * Not necessarily '\0' terminated. Just a pointer and length.
 *
 * Implements some simple string operations, so that it can be used
 * interchangably with std::string in some functions. Namely implements:
 *
 * - size(), length()
 * - operator[]
 * - data()
 */
struct mystring {
    /// pointer to string start
    char* ptr;
    /// string length
    size_t len;

    /// returns the size (length) of the string
    size_t size() const {
        return len;
    }

    /// returns the size (length) of the string
    size_t length() const {
        return len;
    }

    /// returns a reference to the `i`th character
    char& operator[](size_t i) const {
        return ptr[i];
    }

    /// returns a reference to the `i`th character
    char& operator[](size_t i) {
        return ptr[i];
    }

    /// returns a pointer to the first character
    char* data() {
        return ptr;
    }

    /// returns a pointer to the first character
    const char* data() const {
        return ptr;
    }
};


/**
 * @brief A simple stringset with shared storage.
 *
 * The data for all stirngs is stored in `data`.
 * Each string `i` is represented by a pointer to its start
 * `str_begins[i]` and its length `str_lens[i]`.
 */
struct strings {
    /// raw data
    std::vector<char> data;
    /// number of strings
    size_t nstrings;
    /// pointer to first char for each string
    std::vector<char*> str_begins;
    /// length of each string
    std::vector<size_t> str_lens;
    static constexpr char sep = '\n';

    // TODO: constructors??
    // TODO: iterators through strings?

    static strings from_vec(const std::vector<std::string>& v) {
        strings ss;
        ss.nstrings = v.size();
        size_t total_size = 0;
        for (size_t i = 0; i < v.size(); ++i) {
            total_size += v[i].size() + 1;
        }
        ss.data.resize(total_size);

        ss.str_begins.resize(ss.nstrings);
        ss.str_lens.resize(ss.nstrings);

        char* out = ss.data.data();
        for (size_t i = 0; i < v.size(); ++i) {
            ss.str_begins[i] = out;
            ss.str_lens[i] = v[i].size();
            memcpy(out, &v[i][0], v[i].size());
            out += v[i].size();
            if (i+1 < v.size()) {
                *(out++) = '\n';
            } else {
                *(out++) = '\0';
            }
        }

        // replace very last sep with 0

        return ss;
    }

    void parse() {
        // parse the strings by seperator into str_begins, and str_lens
        // given that the data vector contains the string data
        // now parse in mem data
        nstrings = 0;
        str_begins.clear();
        str_lens.clear();

        size_t pos = 0;
        while (pos < data.size()) {
            // skip seperators
            while (pos < data.size() && data[pos] == '\n') {
                ++pos;
            }
            size_t str_beg = pos;
            while (pos < data.size() && data[pos] != '\n') {
                ++pos;
            }
            if (pos > str_beg && pos < data.size()) {
                // save string info
                nstrings++;
                str_begins.emplace_back(data.data() + str_beg);
                str_lens.emplace_back(pos - str_beg);
            }
        }
    }

    static strings from_string(const std::string& s) {
        strings ss;

        // copy into ss
        ss.data.resize(s.size()+1);
        memcpy(ss.data.data(), &s[0], s.size()+1);
        ss.parse();

        return ss;
    }

    static strings from_dfile(const std::string& filename, const mxx::comm& comm) {
        size_t size = mxx::get_filesize(filename.c_str());
        mxx::blk_dist dist(size, comm.size(), comm.rank());

        // open file and seek to my pos
        std::ifstream t(filename);
        t.seekg(dist.eprefix_size());
        // scan for first newline (sep?)
        size_t my_offset = 0;
        if (comm.rank() == 0) {
            my_offset = 0;
        } else {
            my_offset = dist.eprefix_size();
            // find first '\n'
            char c;
            while (t.get(c) && c != '\n') {
                ++my_offset;
            }
            if (my_offset < size) {
                ++my_offset; // advance one further
            }
        }

        size_t my_end = mxx::left_shift(my_offset, comm);
        if (comm.rank() + 1 == comm.size()) {
            my_end = size;
        }

        // create rangebuf
        mxx::rangebuf rb(my_offset, my_end-my_offset, t.rdbuf());
        // read file (range) buffer into string stream
        std::stringstream sstr;
        sstr << &rb;

        std::string local_str(sstr.str());

        return strings::from_string(local_str);
    }
};

std::ostream& operator<<(std::ostream& os, const strings& ss) {
    os << "{nstrings=" << ss.nstrings << ", nbytes=" << ss.data.size() << ", [";
    for (size_t i = 0; i < ss.nstrings; ++i) {
        std::string s(ss.str_begins[i], ss.str_begins[i] + ss.str_lens[i]);
        os << "\"" << s << "\"";
        if (i+1 < ss.nstrings)
            os << ", ";
    }
    os << "]}";
    return os;
}

strings all2all_strings(const strings& ss, std::vector<int>& target, std::vector<size_t>& send_counts, const mxx::comm& comm) {
    std::vector<size_t> offset = mxx::local_exscan(send_counts);
    size_t send_num = offset.back() + send_counts.back();

    // string lengths to send
    std::vector<size_t> send_lens(send_num); // str-lens in bucketed order of strings
    std::vector<size_t> send_data_sizes(comm.size()); // total data size per target proc
    for (size_t i = 0; i < ss.nstrings; ++i) {
        if (target[i] >= 0 && target[i] != comm.rank()) {
            send_lens[offset[target[i]]++] = ss.str_lens[i];
            send_data_sizes[target[i]] += ss.str_lens[i];
        }
    }

    std::vector<size_t> data_offsets = mxx::local_exscan(send_data_sizes);
    size_t send_data_size = data_offsets.back() + send_data_sizes.back();
    std::vector<char> send_data(send_data_size); // buffer for sending
    // create a "sorted" send buffer
    for (size_t i = 0; i < ss.nstrings; ++i) {
        if (target[i] >= 0 && target[i] != comm.rank()) {
            memcpy(&send_data[data_offsets[target[i]]], ss.str_begins[i], ss.str_lens[i]);
            data_offsets[target[i]] += ss.str_lens[i];
            // TODO: just copy this extra char from `ss`?
            //send_data[data_offsets[sprocs[i]]] = '\n';
            //data_offsets[sprocs[i]] += 1;
            // keep string lengths
            //send_lens[lens_offset[target[i]]++] = ss.str_lens[i];
        }
    }

    // send/recv the sequence lengths
    std::vector<size_t> recv_counts = mxx::all2all(send_counts, comm);
    std::vector<size_t> recv_lens = mxx::all2allv(send_lens, send_counts, recv_counts, comm);

    // send/recv the string data of the patterns
    std::vector<size_t> recv_data_sizes = mxx::all2all(send_data_sizes, comm);
    std::vector<char> recv_data = mxx::all2allv(send_data, send_data_sizes, recv_data_sizes, comm);

    // create `strings` data structure from received data
    strings recv_ss;
    recv_ss.data.swap(recv_data); // FIXME allocate memory for data
    recv_ss.nstrings = std::accumulate(recv_counts.begin(), recv_counts.end(), static_cast<size_t>(0));
    recv_ss.str_lens = recv_lens;
    recv_ss.str_begins.resize(recv_ss.nstrings);

    // init str_begins
    size_t beg = 0;
    for (size_t i = 0; i < recv_ss.nstrings; ++i) {
        recv_ss.str_begins[i] = recv_ss.data.data() + beg;
        beg += recv_ss.str_lens[i];
    }

    return recv_ss;
}

#endif // DSTRINGS_HPP
