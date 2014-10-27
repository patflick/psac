#include <mpi.h>

#include <iostream>
#include <string>

#include "test_sac_libdss.hpp"
#include "mpi_sa_test.hpp"

int main(int argc, char *argv[])
{
    // set up MPI
    MPI_Init(&argc, &argv);

    // get communicator size and my rank
    MPI_Comm comm = MPI_COMM_WORLD;
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    // test PSAC against libdivsufsort
    //std::string str = "mississippi";
    /*
    for (auto size : {20, 50, 100, 133, 137, 4000, 13791})
    {
        std::string str = rand_dna(size, 0);
        test_compare_divsufsort_psac(str, comm);
    }
    */
    for (int i = 10; i <= 1031; ++i)
    {
        std::string str = rand_dna(i, 0);
        if (!test_compare_divsufsort_psac(str,comm))
        {
            std::cerr << "Failed with i = " << i << std::endl;
            exit(1);
        }
    }

    // finalize MPI
    MPI_Finalize();
    return 0;
}

