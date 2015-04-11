
#include <iostream>

#include <mxx/reduction.hpp>

int main(int argc, char *argv[])
{
    // set up MPI
    MPI_Init(&argc, &argv);

    // get communicator size and my rank
    MPI_Comm comm = MPI_COMM_WORLD;
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    /* code */
    int x = 2*rank+1;
    //int y = mxx::allreduce(x, [](int x, int y){return x+y;}, comm);
    int y = mxx::allreduce(x, std::max<int>, comm);
    std::cout << "sum = " << y << std::endl;

    // finalize MPI
    MPI_Finalize();
    return 0;
}
