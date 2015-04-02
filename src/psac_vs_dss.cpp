/**
 * @file    ldss.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Executes and times the suffix array construction using
 *          libdivsufsort.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 *
 * TODO add Licence
 */

// include MPI
#include <mpi.h>

// C++ includes
#include <fstream>
#include <iostream>
#include <string>

// using TCLAP for command line parsing
#include <tclap/CmdLine.h>

// libdivsufsort wrapper
#include <divsufsort_wrapper.hpp>
// distributed suffix array construction
#include <suffix_array.hpp>
// random input generation:
#include <alphabet.hpp>

// gather/scatter
#include <mxx/collective.hpp>

// Timer
#include <timer.hpp>

// TODO differentiate between index types (input param or automatic based on
// size!)
typedef uint64_t index_t;

int main(int argc, char *argv[])
{
    // set up MPI
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    try {
    // define commandline usage
    TCLAP::CmdLine cmd("Compare our parallel implementation with divsufsort.");
    TCLAP::ValueArg<std::string> fileArg("f", "file", "Input filename.", true, "", "filename");
    TCLAP::ValueArg<std::size_t> randArg("r", "random", "Random input size", true, 0, "size");
    TCLAP::ValueArg<int> seedArg("s", "seed", "Sets the seed for the ranom input generation", false, 0, "int");
    cmd.add(seedArg);
    cmd.xorAdd(fileArg, randArg);
    TCLAP::SwitchArg  checkArg("c", "check", "Check correctness of SA for PSAC using divsufsort `sufcheck()`.", false);
    cmd.add(checkArg);
    cmd.parse(argc, argv);

    // read input file or generate input on master processor
    std::string input_str;
    if (rank == 0) {
        if (fileArg.getValue() != "") {
            std::ifstream t(fileArg.getValue().c_str());
            std::stringstream buffer;
            buffer << t.rdbuf();
            input_str = buffer.str();
        } else {
            input_str = rand_dna(randArg.getValue(), seedArg.getValue());
        }
    }

    // TODO differentiate between index types

    // run our distributed suffix array construction
    std::string local_str = mxx::scatter_string_block_decomp(input_str);
    timer t;
    double start = t.get_ms();
    suffix_array<std::string::iterator, index_t, false> sa(local_str.begin(), local_str.end(), MPI_COMM_WORLD);
    // TODO: choose construction method!
    sa.construct_arr<2>(true);
    //sa.construct_arr<2>();
    double end = t.get_ms() - start;
    if (rank == 0)
        std::cerr << "PSAC time: " << end << " ms" << std::endl;
    std::vector<index_t> glSA = mxx::gather_vectors(sa.local_SA);


    // run construction with divsufsort locally on rank 0
    if (rank == 0) {
        timer t;
        std::vector<index_t> SA;
        double start = t.get_ms();
        dss::construct(input_str.begin(), input_str.end(), SA);
        double end = t.get_ms() - start;
        std::cerr << "divsufsort time: " << end << " ms" << std::endl;

        // check correctness if we should do so!
        if (checkArg.getValue()) {
            std::cerr << "Checking for correctness..." << std::endl;
            if(!dss::check(input_str.begin(), input_str.end(), glSA)) {
                std::cerr << "ERROR: wrong suffix array from PSAC" << std::endl;
                return false;
            }
            if(!dss::check(input_str.begin(), input_str.end(), SA)) {
                std::cerr << "ERROR: wrong suffix array from divsufsort" << std::endl;
            }
        }
    }

    // catch any TCLAP exception
    } catch (TCLAP::ArgException& e) {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
        exit(EXIT_FAILURE);
    }

    // finalize MPI
    MPI_Finalize();

    return 0;
}
