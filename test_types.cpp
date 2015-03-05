#include <mpi.h>
#include <iostream>
#include <tuple>


#include "mpi_types.hpp"

int main(int argc, char *argv[])
{
    // set up MPI
    MPI_Init(&argc, &argv);

    // get communicator size and my rank
    MPI_Comm comm = MPI_COMM_WORLD;
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);


    //typedef std::tuple<double, int, char> t;
    typedef std::tuple<int, int, int, int, int> t;



    // FIXME: somehow it is not working for the first two values 
    if (rank == 0)
    {
        mpi::datatype<t> mpi_type;
        MPI_Datatype mpi_dt = mpi_type.type();
        t x = std::make_tuple(0, 1, -2, 3, -4);
        MPI_Bcast(&x, 1, mpi_dt, 0, comm);
    }
    else
    {
        mpi::datatype<t> mpi_type;
        MPI_Datatype mpi_dt = mpi_type.type();
        t x;
        MPI_Bcast(&x, 1, mpi_dt, 0, comm);
        std::cout << "On processor with rank = " << rank << std::endl;
        std::cout << "Received tuple (" << std::get<0>(x) << " " << std::get<1>(x) << " " << std::get<2>(x) << ")" << std::endl;
    }

    // finalize MPI
    MPI_Finalize();
    return 0;
}
