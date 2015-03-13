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

#include <mxx/collective.hpp>

#include <divsufsort64.h>

#include "suffix_array.hpp"
#include "timer.hpp"


// C++ interface for libdivsufsort
void divsufsort_sa_construction(const std::string& str, std::vector<saidx64_t>& SA)
{
    saidx64_t n = str.size();
    std::basic_string<sauchar_t> T(reinterpret_cast<const sauchar_t*>(str.data()), n);
    SA.resize(n);
    divsufsort64(T.data(), &SA[0], n);
}

bool divsufsort_sa_check(const std::string& str, const std::vector<saidx64_t>& SA)
{
    saidx64_t n = str.size();
    const sauchar_t* T = reinterpret_cast<const sauchar_t*>(str.data());
    return sufcheck64(T, &SA[0], n, 1) == 0;
}

bool test_compare_divsufsort_psac(std::string& str, MPI_Comm comm)
{
    // run PSAC (same index type)
    //std::vector<saidx_t> psac_SA;
    //std::vector<saidx_t> psac_ISA;
    // distribute input
    //std::string local_str = mxx::scatter_string_block_decomp(str, comm);
    timer t;
    double start = t.get_ms();
    //sa_construction_gl<saidx_t>(str, psac_SA, psac_ISA, comm);
    //suffix_array<std::string::iterator, saidx_t, false> sa(local_str.begin(), local_str.end(), comm);
    //sa.construct_arr<2>(true);
    //sa.construct_arr<2>();
    double end = t.get_ms() - start;

    // get rank
    int rank;
    MPI_Comm_rank(comm, &rank);

    //std::vector<saidx_t> glSA = mxx::gather_vectors(sa.local_SA, comm);

    if (rank == 0)
    {
        std::cerr << "PSAC Time: " << end << " ms" << std::endl;
        // run libdivsufsort
        std::vector<saidx64_t> dss_SA;
        double start = t.get_ms();
        divsufsort_sa_construction(str, dss_SA);
        double end = t.get_ms() - start;
        std::cerr << "Libdivsufsort Time: " << end << " ms" << std::endl;

        // print out both resultsS
        //std::cerr << "STR: " << str << std::endl;
        //std::cerr << "DSS: "; print_range(dss_SA.begin(), dss_SA.end());
        //std::cerr << "PSAC:"; print_range(psac_SA.begin(), psac_SA.end());

        //if(!divsufsort_sa_check(str, glSA))
        //{
        //    std::cerr << "ERROR: wrong suffix array from PSAC" << std::endl;
        //    return false;
        //}
        if(!divsufsort_sa_check(str, dss_SA))
        {
            std::cerr << "ERROR: wrong suffix array from libdivsufsort" << std::endl;
        }
    }
    return true;
}

#endif // TEST_SAC_LIBDSS_HPP
