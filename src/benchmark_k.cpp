#include <mpi.h>

#include <iostream>
#include <vector>

// using TCLAP for command line parsing
#include <tclap/CmdLine.h>

// parallel block decomposition of a file
#include <mxx/file.hpp>

// suffix array construction
#include <suffix_array.hpp>
#include <alphabet.hpp> // for random DNA

#include <timer.hpp>


void benchmark_k(const std::string& local_str, int k, MPI_Comm comm)
{
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);
    timer t;

    typedef suffix_array<std::string::const_iterator, std::size_t, false> sa_t;

    {
        // without LCP and fast
        std::string method_name = "reg-fast-nolcp";
        double start = t.get_ms();
        sa_t sa(local_str.begin(), local_str.end(), comm);
        sa.construct(true, k);
        double time = t.get_ms() - start;
        if (rank == 0)
            std::cout << p << ";" << method_name << ";" << time << std::endl;
    }
    {
        // without LCP and slow
        std::string method_name = "reg-nolcp";
        double start = t.get_ms();
        sa_t sa(local_str.begin(), local_str.end(), comm);
        sa.construct(false, k);
        double time = t.get_ms() - start;
        if (rank == 0)
            std::cout << p << ";" << method_name << ";" << time << std::endl;
    }
    // TODO: array construction with multiple
}

int main(int argc, char *argv[])
{
    // set up MPI
    MPI_Init(&argc, &argv);
    int p, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    try {
    // define commandline usage
    TCLAP::CmdLine cmd("Benchmark different suffix array construction variants.");
    TCLAP::ValueArg<std::string> fileArg("f", "file", "Input filename.", true, "", "filename");
    TCLAP::ValueArg<std::size_t> randArg("r", "random", "Random input size", true, 0, "size");
    cmd.xorAdd(fileArg, randArg);
    TCLAP::ValueArg<int> iterArg("i", "iterations", "Number of iterations to run", false, 1, "num");
    cmd.add(iterArg);
    TCLAP::ValueArg<int> kArg("k", "kmer-size", "The size of the `k` in the intial sorting.", false, 0, "size");
    cmd.add(kArg);
    cmd.parse(argc, argv);

    std::string local_str;
    if (fileArg.getValue() != "")
    {
        local_str = mxx::file_block_decompose(fileArg.getValue().c_str());
    }
    else
    {
        // TODO: proper parallel random generation!!
        local_str = rand_dna(randArg.getValue(), rank);
    }

    // run all benchmarks
    for (int i = 0; i < iterArg.getValue(); ++i)
        benchmark_k(local_str, kArg.getValue(), MPI_COMM_WORLD);

    // catch any TCLAP exception
    } catch (TCLAP::ArgException& e) {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    }

    // finalize MPI
    MPI_Finalize();
    return 0;
}
