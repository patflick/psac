#include <mpi.h>

#include <iostream>
#include <vector>

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
    //MPI_Errhandler errhandler;
    //MPI_Errhandler_create(&my_mpi_errorhandler, &errhandler);
    //MPI_Errhandler_set(comm, errhandler);

    // attach to process 0
    //wait_gdb_attach(0, comm);

    // run the suffix array construction
    test_sa(comm, 30000000);
    //test_sa(comm, 1379, true);

    // finalize MPI
    MPI_Finalize();
    return 0;
}
