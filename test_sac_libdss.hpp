/**
 * @file    test_sac_libdss.hpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   descr
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 *
 * TODO add Licence
 */
#ifndef TEST_SAC_LIBDSS_HPP
#define TEST_SAC_LIBDSS_HPP

#include <mpi.h>

#include <divsufsort.h>
#include "mpi_sa_constr.hpp"
#include "timer.hpp"

// TODO: put this elsewhere
/*
template<typename _Iterator>
void print_range(_Iterator begin, _Iterator end)
{
    while (begin != end)
        std::cerr << *(begin++) << " ";
    std::cerr << std::endl;
}
*/

// C++ interface for libdivsufsort
void divsufsort_sa_construction(const std::string& str, std::vector<saidx_t>& SA)
{
    saidx_t n = str.size();
    std::basic_string<sauchar_t> T(reinterpret_cast<const sauchar_t*>(str.data()), n);
    SA.resize(n);
    divsufsort(T.data(), &SA[0], n);
}

bool divsufsort_sa_check(const std::string& str, const std::vector<saidx_t>& SA)
{
    saidx_t n = str.size();
    const sauchar_t* T = reinterpret_cast<const sauchar_t*>(str.data());
    return sufcheck(T, &SA[0], n, 1) == 0;
}

bool test_compare_divsufsort_psac(std::string& str, MPI_Comm comm)
{
    // run PSAC (same index type)
    std::vector<saidx_t> psac_SA;
    std::vector<saidx_t> psac_ISA;
    timer t;
    double start = t.get_ms();
    sa_construction_gl<saidx_t>(str, psac_SA, psac_ISA, comm);
    double end = t.get_ms() - start;
    std::cerr << "PSAC Time: " << end << " ms" << std::endl;

    // get rank
    int rank;
    MPI_Comm_rank(comm, &rank);

    if (rank == 0)
    {
        // run libdivsufsort
        std::vector<saidx_t> dss_SA;
        double start = t.get_ms();
        divsufsort_sa_construction(str, dss_SA);
        double end = t.get_ms() - start;
        std::cerr << "Libdivsufsort Time: " << end << " ms" << std::endl;

        // print out both resultsS
        //std::cerr << "STR: " << str << std::endl;
        //std::cerr << "DSS: "; print_range(dss_SA.begin(), dss_SA.end());
        //std::cerr << "PSAC:"; print_range(psac_SA.begin(), psac_SA.end());

        if(!divsufsort_sa_check(str, psac_SA))
        {
            std::cerr << "ERROR: wrong suffix array from PSAC" << std::endl;
            return false;
        }
        if(!divsufsort_sa_check(str, dss_SA))
        {
            std::cerr << "ERROR: wrong suffix array from libdivsufsort" << std::endl;
        }
    }
    return true;
}

#endif // TEST_SAC_LIBDSS_HPP
