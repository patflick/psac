/**
 * @file    mpi_sa_test.hpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Tests for the distibuted parallel suffix array construction.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 *
 * TODO add Licence
 */

#ifndef MPI_SA_TEST_HPP
#define MPI_SA_TEST_HPP

#include <mpi.h>

#include <vector>
#include <iostream>
#include <string>

#include <mxx/utils.hpp>
#include <mxx/file.hpp>

#include "lcp.hpp"
#include "suffix_array.hpp"

#include "timer.hpp"

#define SAC_TEST_ENABLE_TIMER 0
#if SAC_TEST_ENABLE_TIMER
#define SAC_TEST_TIMER_START() TIMER_START()
#define SAC_TEST_TIMER_END_SECTION(str) TIMER_END_SECTION(str)
#else
#define SAC_TEST_TIMER_START()
#define SAC_TEST_TIMER_END_SECTION(str)
#endif


/**************************************
 *  test correctness of suffix array  *
 **************************************/
/*
void sa_test_random_dna(MPI_Comm comm, std::size_t input_size, bool test_correct = false)
{
    // get comm parameters
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    //std::string local_str = "missisippi";
    std::string local_str = rand_dna(input_size, rank);

    // construct local SA for input string
    suffix_array<std::string::iterator, std::size_t, true> sa(local_str.begin(), local_str.end(), comm);
    sa.construct();

    // final SA and ISA
    if (test_correct)
    {
        // gather SA and ISA to local
        gl_check_correct(sa, local_str.begin(), local_str.end(), comm);
    }
}
*/

/*
void sa_test_file(const char* filename, MPI_Comm comm, std::size_t max_local_size=0, bool test_correct = false)
{
    // get comm parameters
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    // print out node distribution
    mxx::print_node_distribution(comm);

    SAC_TIMER_START();

    // block decompose input file
    std::string local_str = mxx::file_block_decompose(filename, comm, max_local_size);

    SAC_TEST_TIMER_END_SECTION("load-input");

    // construct local SA for input string
    //sa_construction(local_str, local_SA, local_ISA, local_LCP, comm);
    suffix_array<std::string::iterator, unsigned int, true> sa(local_str.begin(), local_str.end(), comm);
    //sa.construct_arr();
    //sa.construct_arr<5>();
    sa.construct(true);

    SAC_TEST_TIMER_END_SECTION("sac");

    if (test_correct)
    {
        // test the correctness of the SA and the LCP
        gl_check_correct(sa, local_str.begin(), local_str.end(), comm);
    }
}
*/
#endif // MPI_SA_TEST_HPP
