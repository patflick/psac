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

//#define MXX_DISABLE_TIMER 1

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

    /*
    {
    std::vector<size_t> left_nsv(local_input.size());
    std::vector<size_t> right_nsv;
    std::vector<std::pair<size_t, size_t>> lr_mins;
    size_t nonsv = std::numeric_limits<size_t>::max();
    comm.barrier();
    double start = t.elapsed();
    ansv<size_t, nearest_sm, nearest_sm, local_indexing>(local_input, left_nsv, right_nsv, lr_mins, comm, nonsv);
    double time = t.elapsed() - start;
    std::string method_name = "ansv-allpair";
    if (comm.rank() == 0)
        std::cout << comm.size() << ";" << method_name << ";" << time << std::endl;
    }


    {
    std::vector<size_t> left_nsv(local_input.size());
    std::vector<size_t> right_nsv;
    std::vector<std::pair<size_t, size_t>> lr_mins;
    size_t nonsv = std::numeric_limits<size_t>::max();
    comm.barrier();
    double start = t.elapsed();
    hh_ansv<size_t>(local_input, left_nsv, right_nsv, lr_mins, comm, nonsv);
    double time = t.elapsed() - start;
    std::string method_name = "hh-ansv-incorrect";
    if (comm.rank() == 0)
        std::cout << comm.size() << ";" << method_name << ";" << time << std::endl;
    }

    {
    std::vector<size_t> left_nsv(local_input.size());
    std::vector<size_t> right_nsv;
    std::vector<std::pair<size_t, size_t>> lr_mins;
    size_t nonsv = std::numeric_limits<size_t>::max();
    comm.barrier();
    double start = t.elapsed();
    my_ansv_minpair<size_t>(local_input, left_nsv, right_nsv, lr_mins, comm, nonsv);
    double time = t.elapsed() - start;
    std::string method_name = "my-ansv-minpair";
    if (comm.rank() == 0)
        std::cout << comm.size() << ";" << method_name << ";" << time << std::endl;
    }
    */
    {
    std::vector<size_t> left_nsv;
    std::vector<size_t> right_nsv;
    std::vector<std::pair<size_t, size_t>> lr_mins;
    size_t nonsv = std::numeric_limits<size_t>::max();
    comm.barrier();
    double start = t.elapsed();
    old_gansv<size_t, nearest_sm, nearest_sm, local_indexing>(local_input, left_nsv, right_nsv, lr_mins, comm, nonsv);
    double time = t.elapsed() - start;
    std::string method_name = "gansv-old";
    if (comm.rank() == 0)
        std::cout << comm.size() << ";" << method_name << ";" << time << std::endl;
    }

    {
    std::vector<size_t> left_nsv;
    std::vector<size_t> right_nsv;
    std::vector<std::pair<size_t, size_t>> lr_mins;
    size_t nonsv = std::numeric_limits<size_t>::max();
    comm.barrier();
    double start = t.elapsed();
    gansv_impl<size_t, nearest_sm, nearest_sm, local_indexing, allpair>(local_input, left_nsv, right_nsv, lr_mins, comm, nonsv);
    double time = t.elapsed() - start;
    std::string method_name = "gansv-allpair";
    if (comm.rank() == 0)
        std::cout << comm.size() << ";" << method_name << ";" << time << std::endl;
    }

    {
    std::vector<size_t> left_nsv;
    std::vector<size_t> right_nsv;
    std::vector<std::pair<size_t, size_t>> lr_mins;
    size_t nonsv = std::numeric_limits<size_t>::max();
    comm.barrier();
    double start = t.elapsed();
    gansv_impl<size_t, nearest_sm, nearest_sm, local_indexing, minpair>(local_input, left_nsv, right_nsv, lr_mins, comm, nonsv);
    double time = t.elapsed() - start;
    std::string method_name = "gansv-minpair";
    if (comm.rank() == 0)
        std::cout << comm.size() << ";" << method_name << ";" << time << std::endl;
    }

    {
    std::vector<size_t> left_nsv;
    std::vector<size_t> right_nsv;
    std::vector<std::pair<size_t, size_t>> lr_mins;
    size_t nonsv = std::numeric_limits<size_t>::max();
    comm.barrier();
    double start = t.elapsed();
    gansv_impl<size_t, nearest_sm, nearest_sm, local_indexing, left>(local_input, left_nsv, right_nsv, lr_mins, comm, nonsv);
    double time = t.elapsed() - start;
    std::string method_name = "gansv-left";
    if (comm.rank() == 0)
        std::cout << comm.size() << ";" << method_name << ";" << time << std::endl;
    }

    {
    std::vector<size_t> left_nsv;
    std::vector<size_t> right_nsv;
    std::vector<std::pair<size_t, size_t>> lr_mins;
    size_t nonsv = std::numeric_limits<size_t>::max();
    comm.barrier();
    double start = t.elapsed();
    gansv_impl<size_t, nearest_sm, nearest_sm, local_indexing, minpair_duplex>(local_input, left_nsv, right_nsv, lr_mins, comm, nonsv);
    double time = t.elapsed() - start;
    std::string method_name = "gansv-minpair-duplex";
    if (comm.rank() == 0)
        std::cout << comm.size() << ";" << method_name << ";" << time << std::endl;
    }
    {
    std::vector<size_t> left_nsv;
    std::vector<size_t> right_nsv;
    std::vector<std::pair<size_t, size_t>> lr_mins;
    size_t nonsv = std::numeric_limits<size_t>::max();
    comm.barrier();
    double start = t.elapsed();
    gansv_impl<size_t, nearest_sm, nearest_sm, local_indexing, berkman>(local_input, left_nsv, right_nsv, lr_mins, comm, nonsv);
    double time = t.elapsed() - start;
    std::string method_name = "gansv-berkman";
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

std::vector<size_t> generate_input_procpeaks(size_t n, const mxx::comm& c) {
    size_t np = n/c.size();
    std::vector<size_t> local_els(np);

    // 1.) pick random number on processor
    size_t proc_min = std::rand() % n;

    std::srand(0);
    std::srand(std::rand()*c.rank());
    // generate linear peak from `n-1` to proc_min
    size_t n2 = np/2;
    local_els[n2] = proc_min;

    for (size_t i = 0; i < n2; ++i) {
        local_els[i] = n - ((n - proc_min)*i / n2);
    }
    for (size_t i = n2+1; i < np; ++i) {
        local_els[i] = (n-2*proc_min) + ((n - proc_min)*i / n2);
    }

    return local_els;
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

    //std::vector<size_t> local_input = generate_input(100000000, comm);
    std::vector<size_t> local_input = generate_input_procpeaks(80000000, comm);
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
