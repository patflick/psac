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
 * @file    dss.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Executes and times the suffix array construction using
 *          libdivsufsort.
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
#include <mxx/timer.hpp>

int main(int argc, char *argv[])
{
    try {
    // define commandline usage
    TCLAP::CmdLine cmd("Run libdivsufsort suffix array construction and time its execution.");
    TCLAP::ValueArg<std::string> fileArg("f", "file", "Input filename.", true, "", "filename");
    TCLAP::ValueArg<std::size_t> randArg("r", "random", "Random input size", true, 0, "size");
    cmd.xorAdd(fileArg, randArg);
    TCLAP::ValueArg<int> seedArg("s", "seed", "Sets the seed for the ranom input generation", false, 0, "int");
    cmd.add(seedArg);
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
    mxx::timer t;
    for (int i = 0; i < iterArg.getValue(); ++i) {
        std::vector<uint64_t> SA;
        double start = t.elapsed();
        dss::construct(input_str.begin(), input_str.end(), SA);
        double end = t.elapsed() - start;
        std::cerr << end << " ms" << std::endl;
    }

    // catch any TCLAP exception
    } catch (TCLAP::ArgException& e) {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
        exit(EXIT_FAILURE);
    }
    return 0;
}
