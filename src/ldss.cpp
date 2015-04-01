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

// C++ includes
#include <fstream>
#include <iostream>
#include <string>

// using TCLAP for command line parsing
#include <tclap/CmdLine.h>

// libdivsufsort wrapper
#include <divsufsort_wrapper.hpp>
// random input generation:
#include <alphabet.hpp>

// Timer
#include <timer.hpp>

int main(int argc, char *argv[])
{
    try {
    // define commandline usage
    TCLAP::CmdLine cmd("Run libdivsufsort suffix array construction and time its execution.");
    TCLAP::ValueArg<std::string> fileArg("f", "file", "Input filename.", true, "", "filename");
    TCLAP::ValueArg<std::size_t> randArg("r", "random", "Random input size", true, 0, "size");
    TCLAP::ValueArg<int> seedArg("s", "seed", "Sets the seed for the ranom input generation", false, 0, "int");
    cmd.xorAdd(fileArg, randArg);
    TCLAP::ValueArg<int> iterArg("i", "iterations", "Number of iterations to run", false, 1, "num");
    cmd.add(iterArg);
    cmd.parse(argc, argv);

    std::string input_str;
    if (fileArg.getValue() != "")
    {
        std::ifstream t(fileArg.getValue().c_str());
        std::stringstream buffer;
        buffer << t.rdbuf();
        input_str = buffer.str();
    }
    else
    {
        input_str = rand_dna(randArg.getValue(), seedArg.getValue());
    }

    // TODO differentiate between index types
    timer t;
    for (int i = 0; i < iterArg.getValue(); ++i)
    {
        std::vector<uint32_t> SA;
        double start = t.get_ms();
        dss::construct(input_str.begin(), input_str.end(), SA);
        double end = t.get_ms() - start;
        std::cerr << end << " ms" << std::endl;
    }

    // catch any TCLAP exception
    } catch (TCLAP::ArgException& e) {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
        exit(EXIT_FAILURE);
    }
    return 0;
}
