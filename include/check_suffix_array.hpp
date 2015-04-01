/**
 * @file    check_suffix_array.hpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Correctness tests for Suffix array and LCP array.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 *
 * TODO add Licence
 */
#ifndef CHECK_SUFFIX_ARRAY_HPP
#define CHECK_SUFFIX_ARRAY_HPP

#include <vector>
#include <string>
#include <iostream>

#include <suffix_array.hpp>

/**
 * @brief   Checks whether a given suffix array is correct. This function is
 *          sequential and needs all components to be available in-memory.
 *
 * @tparam index_t  The index type.
 * @param SA        The full suffix array (not only the local part).
 * @param ISA       The full ISA (not only the local part).
 * @param str       The full input string.
 *
 * @return  Whether the given suffix array is correct given the string.
 */
template <typename index_t>
bool check_SA(const std::vector<index_t>& SA, const std::vector<index_t>& ISA, const std::string& str)
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
            if (!(str[SA[i-1]] <= str[SA[i]]))
            {
                std::cerr << "[ERROR] wrong SA order: str[SA[i]] >= str[SA[i-1]]" << std::endl;
                success = false;
            }

            // if strings are equal, the ISA of these positions have to be
            // ordered
            if (str[SA[i-1]] == str[SA[i]])
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


/**
 * @brief   Checks whether the given LCP array is correct.
 *
 * This is a sequential check, which requires that all arrays are fully
 * available in local memory.
 *
 * @tparam index_t  The index type (e.g. uint32_t, uint64_t).
 * @param str       The input string.
 * @param SA        The suffix array.
 * @param ISA       The inverse suffix array.
 * @param LCP       The LCP array to be checked (given that the SA and ISA are correct).
 *
 * @return  Whether the LCP is correct given the string, SA, and ISA.
 */
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
        }
    }
    return all_correct;
}


/**
 * @brief   Checks the correctness of the distributed suffix and LCP array.
 *
 * This method gathers all arrays to processor 0 and then uses sequential
 * correctness checkers. Thus this method only works for small inputs, where
 * everything fits onto the memory of a single processor.
 *
 * The template parameters will be deduced from the given distributed suffix
 * array instance.
 *
 * @tparam InputIterator    The type of the char/string input iterator.
 * @tparam index_t          The type of the index (e.g. uint32_t, uint64_t).
 * @tparam test_lcp         Whether the LCP was constructed and should be tested.
 *
 * @param sa            The distributed suffix array instance.
 * @param str_begin     Iterator to the string for which the suffix array was
 *                      constructed.
 * @param str_end       End Iterator to the string for which the suffix array
 *                      was constructed.
 * @param comm          The communictor.
 */
template <typename InputIterator, typename index_t, bool test_lcp>
void gl_check_correct(const suffix_array<InputIterator, index_t, test_lcp>& sa,
                      InputIterator str_begin, InputIterator str_end,
                      // TODO: get the communicator from the distirbuted suffix
                      // array?
                      MPI_Comm comm = MPI_COMM_WORLD)
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


#endif // CHECK_SUFFIX_ARRAY_HPP
