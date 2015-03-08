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
bool gl_check_correct_SA(const std::vector<index_t>& SA, const std::vector<index_t>& ISA, const std::string& str)
{
    std::size_t n = SA.size();
    bool success = true;

    for (std::size_t i = 0; i < n; ++i)
    {
        // check valid range
        if (SA[i] >= n || SA[i] < 0)
        {
            std::cerr << "[ERROR] SA[" << i << "] = " << SA[i] << " out of range 0 <= sa < " << n << std::endl;
            success = false;
        }

        // check SA conditions
        if (i >= 1 && SA[i-1] < n-1)
        {
            if (!(str[SA[i]] >= str[SA[i-1]]))
            {
                std::cerr << "[ERROR] wrong SA order: str[SA[i]] >= str[SA[i-1]]" << std::endl;
                success = false;
            }

            // if strings are equal, the ISA of these positions have to be
            // ordered
            if (str[SA[i]] == str[SA[i-1]])
            {
                if (!(ISA[SA[i-1]+1] < ISA[SA[i]+1]))
                {
                    std::cerr << "[ERROR] invalid SA order: ISA[SA[" << i-1 << "]+1] < ISA[SA[" << i << "]+1]" << std::endl;
                    std::cerr << "[ERROR] where SA[i-1]=" << SA[i-1] << ", SA[i]=" << SA[i] << ", ISA[SA[i-1]+1]=" << ISA[SA[i-1]+1] << ", ISA[SA[i]+1]=" << ISA[SA[i]+1] << std::endl;
                    success = false;
                }
            }
        }
    }

    return success;
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


template <typename InputIterator, typename index_t, bool test_lcp>
void gl_check_correct(const suffix_array<InputIterator, index_t, test_lcp>& sa, InputIterator str_begin, InputIterator str_end,  MPI_Comm comm)
{
    // gather all the data to rank 0
    std::vector<index_t> global_SA = mxx::gather_vectors(sa.local_SA, comm);
    std::vector<index_t> global_ISA = mxx::gather_vectors(sa.local_B, comm);
    std::vector<index_t> global_LCP;
    if (test_lcp)
        global_LCP = mxx::gather_vectors(sa.local_LCP, comm);
    // gather string
    std::vector<char> global_str_vec = mxx::gather_range(str_begin, str_end, comm);
    std::string global_str(global_str_vec.begin(), global_str_vec.end());

    int rank;
    MPI_Comm_rank(comm, &rank);

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
        if (!gl_check_correct_SA(global_SA, global_ISA, global_str))
        {
            std::cerr << "[ERROR] Test unsuccessful" << std::endl;
            exit(1);
        }
        else
        {
            std::cerr << "[SUCCESS] Suffix Array is correct" << std::endl;
        }

        if (test_lcp)
        {
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
    suffix_array<std::string::iterator, unsigned int, false> sa(local_str.begin(), local_str.end(), comm);
    sa.construct_arr();

    SAC_TEST_TIMER_END_SECTION("sac");

    if (test_correct)
    {
        // test the correctness of the SA and the LCP
        gl_check_correct(sa, local_str.begin(), local_str.end(), comm);
    }
}
#endif // MPI_SA_TEST_HPP
