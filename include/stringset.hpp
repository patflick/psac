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

// generate a distributed string set with splits
void random_dstringset() {
    // TODO
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
