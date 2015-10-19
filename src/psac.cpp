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

// distributed suffix array construction
#include <suffix_array.hpp>
#include <check_suffix_array.hpp>
#include <alphabet.hpp>

// parallel file block decompose
#include <mxx/comm.hpp>
#include <mxx/file.hpp>
// Timer
#include <mxx/timer.hpp>

// TODO differentiate between index types (input param or automatic based on
// size!)
typedef uint64_t index_t;

int main(int argc, char *argv[])
{
    // set up MPI
    MPI_Init(&argc, &argv);
    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    try {
    // define commandline usage
    TCLAP::CmdLine cmd("Parallel distirbuted suffix array and LCP construction.");
    TCLAP::ValueArg<std::string> fileArg("f", "file", "Input filename.", true, "", "filename");
    TCLAP::ValueArg<std::size_t> randArg("r", "random", "Random input size", true, 0, "size");
    cmd.xorAdd(fileArg, randArg);
    TCLAP::ValueArg<int> seedArg("s", "seed", "Sets the seed for the ranom input generation", false, 0, "int");
    cmd.add(seedArg);
    TCLAP::SwitchArg  lcpArg("l", "lcp", "Construct the LCP alongside the SA.", false);
    cmd.add(lcpArg);
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
        local_str = rand_dna(randArg.getValue()/p, seedArg.getValue() * rank);
    }

    // TODO differentiate between index types

    // run our distributed suffix array construction
    mxx::timer t;
    double start = t.elapsed();
    if (lcpArg.getValue()) {
        // construct with LCP
        suffix_array<std::string::iterator, index_t, true> sa(local_str.begin(), local_str.end(), MPI_COMM_WORLD);
        // TODO choose construction method
        sa.construct(true);
        double end = t.elapsed() - start;
        if (rank == 0)
            std::cerr << "PSAC time: " << end << " ms" << std::endl;
        if (checkArg.getValue()) {
            gl_check_correct(sa, local_str.begin(), local_str.end(), MPI_COMM_WORLD);
        }
    } else {
        // construct without LCP
        suffix_array<std::string::iterator, index_t, false> sa(local_str.begin(), local_str.end(), MPI_COMM_WORLD);
        // TODO choose construction method
        sa.construct_arr<2>(true);
        double end = t.elapsed() - start;
        if (rank == 0)
            std::cerr << "PSAC time: " << end << " ms" << std::endl;
        if (checkArg.getValue()) {
            gl_check_correct(sa, local_str.begin(), local_str.end(), MPI_COMM_WORLD);
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
