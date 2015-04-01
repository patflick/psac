#include <mpi.h>

#include <iostream>
#include <string>

#include "test_sac_libdss.hpp"
#include "mpi_sa_test.hpp"


void my_mpi_errorhandler(MPI_Comm* comm, int* errorcode, ...)
{
    // throw exception, enables gdb stack trace analysis
    throw std::runtime_error("Shit: mpi fuckup");
}

int main(int argc, char *argv[])
{
    // set up MPI
    MPI_Init(&argc, &argv);

    // get communicator size and my rank
    MPI_Comm comm = MPI_COMM_WORLD;
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    // set custom error handler (for debugging with working stack-trace on gdb)
    MPI_Errhandler errhandler;
    MPI_Errhandler_create(&my_mpi_errorhandler, &errhandler);
    //MPI_Errhandler_set(comm, errhandler);

    // test PSAC against libdivsufsort
    //std::string str = "mississippi";
    /*
    for (int i = 2*p; i <= 1031; ++i)
    {
        std::string str = rand_dna(i, 0);
        if (!test_compare_divsufsort_psac(str,comm))
        {
            std::cerr << "Failed with i = " << i << std::endl;
            exit(1);
        }
    }
    */
    std::string str;
    //if (false)
    if (argc >= 2)
    {
        std::ifstream t(argv[1]);
        std::stringstream buffer;
        buffer << t.rdbuf();
        str = buffer.str();
    }
    else
    {
        std::cerr << "Warning: no input file provided, testing with random DNA string." << std::endl;
    //    long n = atol(argv[1]);
        long n = 100;
        str = rand_dna(n,0);
    }
    //if (!test_compare_divsufsort_psac(str,comm))
    {
        std::cerr << "Failed with p = " << p << std::endl;
        exit(1);
    }

    // finalize MPI
    MPI_Finalize();
    return 0;
}

