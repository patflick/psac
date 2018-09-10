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

/**
 * @file    gsac.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Generalized suffix array construction.
 */

// include MPI
#include <mpi.h>

// C++ includes
#include <fstream>
#include <iostream>
#include <string>

// using TCLAP for command line parsing
#include <tclap/CmdLine.h>

// distributed suffix array construction
#include <suffix_array.hpp>
#include <check_suffix_array.hpp>
#include <alphabet.hpp>

// suffix tree construction
#include <suffix_tree.hpp>
#include <check_suffix_tree.hpp>

#include <divsufsort_wrapper.hpp>

// parallel file block decompose
#include <mxx/env.hpp>
#include <mxx/comm.hpp>
#include <mxx/file.hpp>
#include <mxx/utils.hpp>
// Timer
#include <mxx/timer.hpp>

// TODO differentiate between index types (input param or automatic based on
// size!)
typedef uint64_t index_t;

// SA to GSA !? (sequential)
template <typename Iterator>
std::vector<index_t> dss_gsa(Iterator begin, Iterator end, char sep='\n') {
    std::vector<index_t> dss_sa;
    dss::construct(begin, end, dss_sa);

    // count separating characters
    size_t m = std::count(begin, end, sep);

    // get the number of separating characters appearing in front of any character
    std::vector<index_t> ps(dss_sa.size(), 0);
    for (size_t i = 0; i < dss_sa.size(); ++i) {
        if (*(begin+i) == sep) {
            ps[i] = 1;
        }
    }
    mxx::local_scan_inplace(ps);

    for (size_t i = m; i < dss_sa.size(); ++i) {
        dss_sa[i] -= ps[dss_sa[i]];
    }

    // "remove" the first `m` entries in SA
    // (assuming the separating character is smallest in alphabet)
    std::vector<index_t> gsa(dss_sa.begin() + m, dss_sa.end());

    return gsa;
}

// gathers everything to rank 0 and checks the GSA against SA created by divsufsort
bool gl_check_gsa(const std::vector<index_t>& local_sa, const std::string& local_str, const mxx::comm& c) {
    std::vector<char> gstr = mxx::gatherv(&local_str[0], local_str.size(), 0, c);
    std::vector<index_t> gsa = mxx::gatherv(local_sa, 0, c);

    bool success = true;
    if (c.rank() == 0) {
        std::vector<index_t> dgsa = dss_gsa(gstr.begin(), gstr.end());


        if (gsa.size() != dgsa.size()) {
            std::cerr << "[ERROR] GSAs are not equal sized" << std::endl;
            return false;
        }

        // TODO: this should be supported by the (distributed) stringset class
        //       e.g. via succinct rank/select for gidx -> string_idx + offset
        //       using the prefix sizes L[]
        std::vector<index_t> gidx_to_s(gsa.size(),0);
        for (size_t i = 0, j=0; i < gstr.size(); ++i) {
            if (gstr[i] != '\n') {
                gidx_to_s[j++] = i;
            }
        }

        for (size_t i = 0; i < dgsa.size(); ++i) {
            if (gsa[i] != dgsa[i]) {
                // possibly non-unique suffix of different strings
                // check that both suffixes are equal until separating character
                index_t s1 = gidx_to_s[gsa[i]];
                index_t s2 = gidx_to_s[dgsa[i]];
                // check that suffixes s1 and s2 are identical
                while (s1 < gstr.size() && s2 < gstr.size() && gstr[s1] == gstr[s2] && gstr[s1] != '\n') {
                    ++s1; ++s2;
                }

                // advanced till end-of-string (eos):
                bool s1_eos = (s1 == gstr.size() || gstr[s1] == '\n');
                bool s2_eos = (s2 == gstr.size() || gstr[s2] == '\n');
                // suffixes are equal only if both s1&s2 are at eos
                if (!(s1_eos && s2_eos)) {
                    std::cerr << "[ERROR] gsa[" << i << "] != dgsa[" << i << "]" << std::endl;
                    success = false;
                }
            }
        }
    }
    if (success) {
        std::cout << "[SUCCESS] GSA correct" << std::endl;
    }
    return success;
}

int main(int argc, char *argv[]) {
    // set up MPI
    mxx::env e(argc, argv);
    mxx::env::set_exception_on_error();
    mxx::comm comm = mxx::comm();
    mxx::print_node_distribution(comm);

    try {
    // define commandline usage
    TCLAP::CmdLine cmd("Parallel distributed generalized suffix array and LCP construction.");
    TCLAP::ValueArg<std::string> fileArg("f", "file", "Input filename.", true, "", "filename");
    cmd.add(fileArg);
    /*
    TCLAP::ValueArg<std::size_t> randArg("r", "random", "Random input size", true, 0, "size");
    cmd.xorAdd(fileArg, randArg);
    TCLAP::ValueArg<int> seedArg("s", "seed", "Sets the seed for the ranom input generation", false, 0, "int");
    cmd.add(seedArg);
    */
    TCLAP::SwitchArg  lcpArg("l", "lcp", "Construct the LCP alongside the SA.", false);
    cmd.add(lcpArg);
    /*
    TCLAP::SwitchArg  stArg("t", "tree", "Construct the Suffix Tree structute.", false);
    cmd.add(stArg);
    */
    TCLAP::SwitchArg  checkArg("c", "check", "Check correctness of SA (and LCP).", false);
    cmd.add(checkArg);
    cmd.parse(argc, argv);

    // block decompose input file and create distributed stringset
    std::string local_str = mxx::file_block_decompose(fileArg.getValue().c_str(), MPI_COMM_WORLD);
    simple_dstringset ss(local_str.begin(), local_str.end(), comm, '\n');

    // read through input to create alphabet histogram
    alphabet<char> alpha = alphabet<char>::from_stringset(ss, comm);

    // construct GSA
    mxx::timer t;
    double start = t.elapsed();
    //if (lcpArg.getValue()) {
        // construct SA+LCP
        //suffix_array<char, index_t, true> sa(comm);
        // TODO:
        // - get alphabet
        // - pass ss and alphabet
        //sa.construct_ss(ss, alpha);
        //double end = t.elapsed() - start;
        //if (comm.rank() == 0)
        //    std::cerr << "PSAC time: " << end << " ms" << std::endl;
    //} else {
        // construct just SA
        suffix_array<char, index_t, false> sa(comm);
        sa.construct_ss(ss, alpha);
        //sa.local_SA
        double end = t.elapsed() - start;
        if (comm.rank() == 0)
            std::cerr << "PSAC time: " << end << " ms" << std::endl;
    //}

        if (checkArg.getValue()) {
            gl_check_gsa(sa.local_SA, local_str, comm);
        }


    // catch any TCLAP exception
    } catch (TCLAP::ArgException& e) {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
        exit(EXIT_FAILURE);
    }

    return 0;
}
