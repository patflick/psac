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

#include "mpi_utils.hpp"
#include "mpi_sa_constr.hpp"
#include "mpi_file.hpp"
#include "lcp.hpp"

#include "timer.hpp"

#define SAC_TEST_ENABLE_TIMER 0
#if SAC_TEST_ENABLE_TIMER
#define SAC_TEST_TIMER_START() TIMER_START()
#define SAC_TEST_TIMER_END_SECTION(str) TIMER_END_SECTION(str)
#else
#define SAC_TEST_TIMER_START()
#define SAC_TEST_TIMER_END_SECTION(str)
#endif

/*****************************
 *  create random DNA input  *
 *****************************/

// TODO: put these functions elsewhere
inline char rand_dna_char()
{
    char DNA[4] = {'A', 'C', 'G', 'T'};
    return DNA[rand() % 4];
}

std::string rand_dna(std::size_t size, int seed)
{
    srand(1337*seed);
    std::string str;
    str.resize(size, ' ');
    for (std::size_t i = 0; i < size; ++i)
    {
        str[i] = rand_dna_char();
    }
    return str;
}

template <typename index_t>
bool check_lcp(const std::string& str, const std::vector<index_t>& SA, const std::vector<index_t>& ISA, const std::vector<index_t>& LCP)
{
    // construct reference LCP (sequentially)
    std::vector<index_t> ref_LCP;
    lcp_from_sa(str, SA, ISA, ref_LCP);

    // check if reference is equal to this LCP
    if (LCP.size() != ref_LCP.size())
    {
        std::cerr << "[ERROR] LCP size is wrong: " << LCP.size() << "!=" << ref_LCP.size() << std::endl;
        return false;
    }
    // check that all LCP values are equal
    bool all_correct = true;
    for (std::size_t i = 0; i < LCP.size(); ++i)
    {
        if (LCP[i] != ref_LCP[i])
        {
            std::cerr << "[ERROR] LCP[" << i << "]=" << LCP[i] << " != " << ref_LCP[i] << "=ref_LCP[" << i << "]" << std::endl;
            all_correct = false;
     //       throw std::runtime_error("");
        }
    }
    return all_correct;
}

/**************************************
 *  test correctness of suffix array  *
 **************************************/

void sa_test_random_dna(MPI_Comm comm, std::size_t input_size, bool test_correct = false)
{
    // get comm parameters
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    //std::string local_str = "missisippi";
    std::string local_str = rand_dna(input_size, rank);

    std::vector<std::size_t> local_SA;
    std::vector<std::size_t> local_ISA;
    std::vector<std::size_t> local_LCP;

    // construct local SA for input string
    sa_construction(local_str, local_SA, local_ISA, local_LCP, comm);

    // final SA and ISA
    if (test_correct)
    {
        // gather SA and ISA to local
        std::vector<std::size_t> global_SA = gather_vectors(local_SA, comm);
        std::vector<std::size_t> global_ISA = gather_vectors(local_ISA, comm);
        std::vector<std::size_t> global_LCP = gather_vectors(local_LCP, comm);
        std::vector<char> global_str_vec = gather_range(local_str.begin(), local_str.end(), comm);
        std::string global_str(global_str_vec.begin(), global_str_vec.end());
        if (rank == 0)
        {
#if 0
            std::cerr << "##################################################" << std::endl;
            std::cerr << "#               Final SA and ISA                 #" << std::endl;
            std::cerr << "##################################################" << std::endl;
            std::cerr << "STR: " << global_str << std::endl;
            std::cerr << "SA : "; print_vec(global_SA);
            std::cerr << "ISA: "; print_vec(global_ISA);
            std::cerr << "LCP: "; print_vec(global_LCP);
#endif

            // check if correct
            if (!gl_check_correct_SA(global_SA, global_ISA, global_str))
            {
                std::cerr << "[ERROR] Test unsuccessful" << std::endl;
                exit(1);
            }
            else
            {
                std::cerr << "[SUCCESS] Suffix Array is correct" << std::endl;
            }

            if (!check_lcp(global_str, global_SA, global_ISA, global_LCP))
            {
                std::cerr << "[ERROR] Test unsuccessful" << std::endl;
                exit(1);
            }
            else
            {
                std::cerr << "[SUCCESS] LCP Array is correct" << std::endl;
            }
        }
    }
}


void sa_test_file(const char* filename, MPI_Comm comm, std::size_t max_local_size=0, bool test_correct = false)
{
    // get comm parameters
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    // print out node distribution
    print_node_distribution(comm);

    SAC_TIMER_START();

    // block decompose input file
    std::string local_str = file_block_decompose(filename, comm, max_local_size);

    SAC_TEST_TIMER_END_SECTION("load-input");

    std::vector<std::size_t> local_SA;
    std::vector<std::size_t> local_ISA;
    std::vector<std::size_t> local_LCP;

    // construct local SA for input string
    sa_construction(local_str, local_SA, local_ISA, local_LCP, comm);

    SAC_TEST_TIMER_END_SECTION("sac");

    // final SA and ISA
    if (test_correct)
    {
        // gather SA and ISA to local
        std::vector<std::size_t> global_SA = gather_vectors(local_SA, comm);
        std::vector<std::size_t> global_ISA = gather_vectors(local_ISA, comm);
        std::vector<char> global_str_vec = gather_range(local_str.begin(), local_str.end(), comm);
        std::string global_str(global_str_vec.begin(), global_str_vec.end());
        if (rank == 0)
        {
#if 0
            std::cerr << "##################################################" << std::endl;
            std::cerr << "#               Final SA and ISA                 #" << std::endl;
            std::cerr << "##################################################" << std::endl;
            std::cerr << "STR: " << global_str << std::endl;
            std::cerr << "SA : "; print_vec(global_SA);
            std::cerr << "ISA: "; print_vec(global_ISA);
#endif

            // check if correct
            if (!gl_check_correct_SA(global_SA, global_ISA, global_str))
            {
                std::cerr << "[ERROR] Test unsuccessful" << std::endl;
                exit(1);
            }
            else
            {
                std::cerr << "[SUCCESS]" << std::endl;
            }
        }
    }
}
#endif // MPI_SA_TEST_HPP
