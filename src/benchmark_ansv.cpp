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

// using TCLAP for command line parsing
#include <tclap/CmdLine.h>

// parallel block decomposition of a file
#include <mxx/env.hpp>
#include <mxx/comm.hpp>
#include <mxx/distribution.hpp>
#include <mxx/file.hpp>
#include <mxx/timer.hpp>

// ansv methods
#include <ansv.hpp>


void benchmark_all(const std::vector<size_t>& local_input, const mxx::comm& comm) {
    mxx::timer t;

    {
    std::vector<size_t> left_nsv;
    std::vector<size_t> right_nsv;
    std::vector<std::pair<size_t, size_t>> lr_mins;
    size_t nonsv = std::numeric_limits<size_t>::max();
    double start = t.elapsed();
    my_ansv_minpair_lbub<size_t, nearest_sm, nearest_sm, local_indexing>(local_input, left_nsv, right_nsv, lr_mins, comm, nonsv);
    double time = t.elapsed() - start;
    std::string method_name = "minpair_lbub";
    if (comm.rank() == 0)
        std::cout << comm.size() << ";" << method_name << ";" << time << std::endl;
    }
    {
    std::vector<size_t> left_nsv(local_input.size());
    std::vector<size_t> right_nsv;
    std::vector<std::pair<size_t, size_t>> lr_mins;
    size_t nonsv = std::numeric_limits<size_t>::max();
    double start = t.elapsed();
    ansv_local_finish_furthest_eq<size_t, dir_left, dir_left, local_indexing>(local_input, lr_mins.begin(), lr_mins.begin(), 0, 0, nonsv, left_nsv);
    double time = t.elapsed() - start;
    std::string method_name = "finish_furthest_eq";
    if (comm.rank() == 0)
        std::cout << comm.size() << ";" << method_name << ";" << time << std::endl;
    }
}

std::vector<size_t> generate_input(size_t n, const mxx::comm& c) {
    std::vector<size_t> result;
    if (c.rank() == 0) {
        result.resize(n);
        std::generate(result.begin(), result.end(), [&n]() { return std::rand() % n; });
    }
    mxx::stable_distribute_inplace(result, c);
    return result;
}

int main(int argc, char *argv[])
{
    mxx::env e(argc, argv);
    mxx::comm comm = mxx::comm();

    try {
    // define commandline usage
    TCLAP::CmdLine cmd("Benchmark different ANSV variants.");
    // TODO: benchmark for actual LCP numbers (-> read via file)
    // TCLAP::ValueArg<std::string> fileArg("f", "file", "Input filename.", true, "", "filename");
    TCLAP::ValueArg<std::size_t> randArg("r", "random", "Random input size", true, 0, "size");
    cmd.add(randArg);
    //cmd.xorAdd(fileArg, randArg);
    TCLAP::ValueArg<int> iterArg("i", "iterations", "Number of iterations to run", false, 1, "num");
    cmd.add(iterArg);
    cmd.parse(argc, argv);

    std::vector<size_t> local_input = generate_input(100000000, comm);
    /*
    if (fileArg.getValue() != "")
    {
        local_str = mxx::file_block_decompose(fileArg.getValue().c_str());
    } else {
    }
    */

    // run all benchmarks
    for (int i = 0; i < iterArg.getValue(); ++i)
        benchmark_all(local_input, comm);

    // catch any TCLAP exception
    } catch (TCLAP::ArgException& e) {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    }

    return 0;
}
