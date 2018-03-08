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

/**
 * @file    ldss.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Executes and times the suffix array construction using
 *          libdivsufsort.
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

int main(int argc, char *argv[]) {
    // set up MPI
    mxx::env e(argc, argv);
    mxx::env::set_exception_on_error();
    mxx::comm comm = mxx::comm();
    mxx::print_node_distribution(comm);

    try {
    // define commandline usage
    TCLAP::CmdLine cmd("Parallel distributed suffix array and LCP construction.");
    TCLAP::ValueArg<std::string> fileArg("f", "file", "Input filename.", true, "", "filename");
    TCLAP::ValueArg<std::size_t> randArg("r", "random", "Random input size", true, 0, "size");
    cmd.xorAdd(fileArg, randArg);
    TCLAP::ValueArg<int> seedArg("s", "seed", "Sets the seed for the ranom input generation", false, 0, "int");
    cmd.add(seedArg);
    TCLAP::SwitchArg  lcpArg("l", "lcp", "Construct the LCP alongside the SA.", false);
    cmd.add(lcpArg);
    TCLAP::SwitchArg  stArg("t", "tree", "Construct the Suffix Tree structute.", false);
    cmd.add(stArg);
    TCLAP::SwitchArg  checkArg("c", "check", "Check correctness of SA (and LCP).", false);
    cmd.add(checkArg);
    cmd.parse(argc, argv);

    // read input file or generate input on master processor
    // block decompose input file
    std::string local_str;
    if (fileArg.getValue() != "") {
        local_str = mxx::file_block_decompose(fileArg.getValue().c_str(), MPI_COMM_WORLD);
    } else {
        // TODO proper distributed random!
        local_str = rand_dna(randArg.getValue()/comm.size(), seedArg.getValue() * comm.rank());
    }

    // TODO differentiate between index types

    // run our distributed suffix array construction
    mxx::timer t;
    double start = t.elapsed();
    if (stArg.getValue()) {
        // construct SA+LCP+ST
        suffix_array<char, size_t, true> sa(comm);
        sa.construct(local_str.begin(), local_str.end());
        double sa_time = t.elapsed() - start;
        // build ST
        std::vector<size_t> local_st_nodes = construct_suffix_tree(sa, local_str.begin(), local_str.end(), comm);
        double st_time = t.elapsed() - sa_time;
        if (comm.rank() == 0) {
            std::cerr << "SA time: " << sa_time << " ms" << std::endl;
            std::cerr << "ST time: " << st_time << " ms" << std::endl;
            std::cerr << "Total  : " << sa_time+st_time << " ms" << std::endl;
        }
        if (checkArg.getValue())  {
            gl_check_suffix_tree(local_str, sa, local_st_nodes, comm);
        }

    } else if (lcpArg.getValue()) {
        // construct SA+LCP
        suffix_array<char, index_t, true> sa(comm);
        sa.construct(local_str.begin(), local_str.end(), true);
        double end = t.elapsed() - start;
        if (comm.rank() == 0)
            std::cerr << "PSAC time: " << end << " ms" << std::endl;
        if (checkArg.getValue()) {
            gl_check_correct(sa, local_str.begin(), local_str.end(), comm);
        }
    } else {
        // construct SA
        suffix_array<char, index_t, false> sa(comm);
        sa.construct(local_str.begin(), local_str.end(), true);
        double end = t.elapsed() - start;
        if (comm.rank() == 0)
            std::cerr << "PSAC time: " << end << " ms" << std::endl;
        if (checkArg.getValue()) {
            d_check_sa(sa, local_str.begin(), local_str.end(), comm);
        }
    }

    // catch any TCLAP exception
    } catch (TCLAP::ArgException& e) {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
        exit(EXIT_FAILURE);
    }

    // finalize MPI
    //MPI_Finalize();

    return 0;
}
