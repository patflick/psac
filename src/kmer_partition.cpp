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

#include <mpi.h>

#include <iostream>
#include <vector>
#include <stack>

// using TCLAP for command line parsing
#include <tclap/CmdLine.h>

// mxx dependency
#include <mxx/env.hpp>

// suffix array construction
#include <divsufsort_wrapper.hpp>
#include <alphabet.hpp> // for random DNA
#include <kmer.hpp>
#include <partition.hpp>
#include <seq_query.hpp>
#include <rmq.hpp>
#include <tldt.hpp>

std::string file2string(const std::string& filename) {
    // read input file into in-memory string
    std::string input_str;
    std::ifstream t(filename.c_str());
    std::stringstream buffer;
    buffer << t.rdbuf();
    input_str = buffer.str();
    return input_str;
}


template <typename index_t>
struct tl_desa {
    index_t total_size;
    std::vector<char> lc;
    std::vector<index_t> lcp;
    std::vector<index_t> off; // offsets

    using range_t = std::pair<index_t, index_t>;

    using it_t = typename std::vector<index_t>::const_iterator;
    rmq<it_t, index_t> minq;

    void construct(const std::vector<index_t>& LCP, const std::vector<char> Lc, index_t maxsize) {
        // local ansv !?
        off = sample_lcp(LCP, maxsize);

        // create sampled DESA
        std::cout << "creating TL DT with " << off.size() << " els" << std::endl;
        lcp.resize(off.size());
        lc.resize(off.size());

        for (size_t i = 0; i < off.size(); ++i) {
            lcp[i] = LCP[off[i]];
            lc[i] = Lc[off[i]];
        }

        // constrct RMQ over new sampled LCP
        minq = rmq<it_t,index_t>(lcp.begin(), lcp.end());
    }


    range_t locate(const std::string& P) {

        size_t n = lcp.size();
        size_t m = P.size();
        size_t l = 0;
        size_t r = n-1;

        // get first child interval and depth
        size_t i = this->minq(l+1, r);
        index_t q = this->lcp[i];

        // blind search
        while (q < m && l < r) {

            // NOTE: LCP[i] = lcp(SA[i-1],SA[i]), LCP[0] = 0
            // using [l,r] as an inclusive SA range
            // corresponding to LCP query range [l+1,r]

            // check if we've reached the end of the pattern
            if (q >= m) {
                break;
            }

            do {
                // `i` is the lcp(SA[i-1],SA[i])
                char lc = this->lc[i]; // == S[SA[l]+lcpv] for first iter
                if (lc == P[q]) {
                    r = i-1;
                    break;
                }
                l = i;
                if (l == r)
                    break;

                i = this->minq(l+1, r);
            } while (l < r && this->lcp[i] == q);

            if (this->lcp[i] == q) {
                if (l+1 < r) {
                    i = this->minq(l+1, r);
                } else {
                    i = l;
                }
            }
            q = this->lcp[i];
        }

        // return the range using offsets
        if (r == n-1) {
            return range_t(off[l], total_size);
        } else {
            return range_t(off[l], off[r+1]);
        }
    }
};

// table: inclusive prefix sum?
void evaluate_partition(const std::vector<size_t>& table) {

    for (int p = 2; p <= 1024; p <<= 1) {
        // partitining by bisection
        std::vector<size_t> part = partition(table, p);

        // partition pointers to partition size
        int start = 0;
        std::vector<size_t> part_size(p, 0);
        for (int i = 0; i < p; ++i) {
            int end = (i+1 < p) ? part[i+1] : table.size();
            //table[start] // inclusive !?
            // table[i] = 
            // hist[i] = table[i] - table[i-1]
            // sum [start,end) ->
            //     table[end-1] - table[start-1]
            part_size[i] = table[end-1];
            if (start > 0)
                part_size[i] -= table[start-1];

            start = end;
        }

        size_t max_size = *std::max_element(part_size.begin(), part_size.end());
        size_t min_size = *std::min_element(part_size.begin(), part_size.end());
        size_t avg = table.back() / p;

        std::cout << "for p=" << p << " size range [" << max_size << "," << min_size << "], load imbalance: " << (max_size - avg)*100./avg << " %" << std::endl;
    }
}

int main(int argc, char *argv[])
{
    // set up mxx / MPI
    mxx::env e(argc, argv);

    try {
        // define commandline usage
        TCLAP::CmdLine cmd("Load Imbalance study for partitioning of kmers");
        TCLAP::ValueArg<std::string> fileArg("f", "file", "Input filename.", true, "", "filename");
        cmd.add(fileArg);
        TCLAP::SwitchArg trieArg("t", "trie", "use dynamic TL trie", false);
        cmd.add(trieArg);
        cmd.parse(argc, argv);

        if (!trieArg.getValue()) {
            int bits = 25;

            std::string str = file2string(fileArg.getValue());
            alphabet<char> a = alphabet<char>::from_sequence(str.begin(), str.end());
            unsigned int l = a.bits_per_char();
            std::cout << "alphabet: " << a << std::endl;
            int k = bits / l;
            std::cout << "using k = " << k << std::endl;

            // scan through string to create kmer histogram
            std::vector<size_t> hist = kmer_hist<size_t>(str.begin(), str.end(), k, a);
            std::vector<size_t> table(hist.size());
            std::cout << "kmer table size: " << table.size()*sizeof(size_t) / 1024 << " kiB" << std::endl;

            std::partial_sum(hist.begin(), hist.end(), table.begin());
            evaluate_partition(table);

        } else {
            // use top level trie
            std::cout << "loading string" << std::endl;
            std::string str = file2string(fileArg.getValue());
            std::cout << "construting DESA" << std::endl;
            desa_index<size_t> idx;
            idx.construct(str.begin(), str.end());


            std::cout << "constructing TL DESA" << std::endl;
            tl_desa<size_t> tl;
            size_t maxsize = str.size() / (1024*128);
            tl.construct(idx.LCP, idx.Lc, maxsize);
            std::cout << "constructed TL with " << tl.lcp.size() << " elements" << std::endl;
            std::cout << "TL total size: " << tl.off.size()*(2*sizeof(size_t)+1) / 1024 << " kiB" << std::endl;

            // use TL offsets and create inclusive prefix sum
            std::vector<size_t> table(tl.off.size());

            for (size_t i = 0; i+1 < table.size(); ++i) {
                table[i] = tl.off[i+1];
            }
            table.back() = str.size();

            evaluate_partition(table);
        }

    // catch any TCLAP exception
    } catch (TCLAP::ArgException& e) {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    }

    return 0;
}
