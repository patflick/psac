#include <mpi.h>
#include <iostream>
#include <tuple>

#include "prettyprint.hpp"


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
    typedef std::tuple<char, int, int, char, double, int, char, int, char> t;


    struct X
    {
        int x;
        char y;
        int x2;
        char z;
    };

    if (rank == 0)
    {
        mxx::datatype<t> mpi_type;
        MPI_Datatype mpi_dt = mpi_type.type();
        t x[2];
        x[0]  = std::make_tuple('x',0,  -2, 'y', -3.333, -42, 'P', 1337, '+');
        x[1]  = std::make_tuple('h',13, -13, 'i', -3.141, -41, 'Q', 1444, '-');
        MPI_Bcast(x, 2, mpi_dt, 0, comm);
        std::cout << "addresses: " << (void*) &std::get<0>(x[0]) << ", "
                                   << (void*) &std::get<1>(x[0]) << ", "
                                   << (void*) &std::get<2>(x[0]) << ", "
                                   << (void*) &std::get<3>(x[0]) << ", "
                                   << (void*) &std::get<4>(x[0]) << ", "
                                   << (void*) &std::get<5>(x[0]) << ", "
                                   << (void*) &std::get<6>(x[0]) << ", "
                                   << (void*) &std::get<7>(x[0]) << ", "
                                   << (void*) &std::get<8>(x[0]) << std::endl;
        X x2;
        std::cout << "addresses: " << (void*) &x2.x << ", "
                                   << (void*) &x2.y << ", "
                                   << (void*) &x2.x2 << ", "
                                   << (void*) &x2.z << std::endl;
    }
    else
    {
        mxx::datatype<t> mpi_type;
        MPI_Datatype mpi_dt = mpi_type.type();
        t x[2];
        MPI_Bcast(x, 2, mpi_dt, 0, comm);
        std::cout << "On processor with rank = " << rank << std::endl;
        std::cout << "Received tuple " << x[0] << std::endl;
        std::cout << "Received tuple " << x[1] << std::endl;
    }

    // finalize MPI
    MPI_Finalize();
    return 0;
}
