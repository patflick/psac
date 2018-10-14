
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <utility>

#include <mxx/env.hpp>
#include <mxx/comm.hpp>

#include <suffix_array.hpp>
#include <lcp.hpp>
#include <divsufsort_wrapper.hpp>

#include "seq_query.hpp"


#include <sdsl/suffix_arrays.hpp>

// TODO:
// - bunch of automated test cases for these types of (sequential) queries
// - parallel algo for computing Lc (& Rc)
// - optimizations
// - compare runtime to different sais-lite indeces
//      - benchmark suite comparing both on large datasets
// - distribution of ESA (by kmer/ vs by dynamic length suffixes !?)
// - top level lookup table (duplicated)
// - large scale query experiments


int main(int argc, char *argv[]) {
    mxx::env e;
    mxx::comm c;
    // given a file, compute suffix array and lcp array, how to do count/locate query?
    // via sa_range()
    if (argc < 2) {
        std::cerr << "Usage: ./xx <filename>" << std::endl;
    }

    // read input file into in-memory string
    std::string filename(argv[1]);
    std::string input_str;
    std::ifstream t(filename.c_str());
    std::stringstream buffer;
    buffer << t.rdbuf();
    input_str = buffer.str();

    // construct SA and LCP
    using index_t = uint64_t;
    esa_index<index_t> idx;
    idx.construct(input_str.begin(), input_str.end());

    std::string P = "i"; // pattern
    if (argc >= 3) {
        P = argv[2];
    }

    std::pair<index_t, index_t> r = idx.locate(P);
    std::cout << "Found " << r.second - r.first << " matches: " << "[" << r.first << "," << r.second << ")" << std::endl;
    // print out all matches !?
    std::string S = input_str;
    for (size_t i = r.first; i < r.second; ++i) {
        std::cout << "Match at S[" << idx.SA[i] << "..]: \"" << std::string(&S[idx.SA[i]]) << "\"" << std::endl;
    }


    //using csa_t = csa_wt<wt_huff<bit_vector,rank_support_v5<>,select_support_scan<>,select_support_scan<> >,S_SA,S_ISA,text_order_sa_sampling<sd_vector<> > >;
    /*
    constexpr int sa_sample = 32;
    constexpr int isa_sample = 32;
    using csa_t = csa_wt<wt_huff<>, sa_sample, isa_sample, text_order_sa_sampling<sd_vector<> >>;
    */
    using csa_t = sdsl::csa_wt<>;
    csa_t csa;
    sdsl::construct(csa, filename, 1);
    auto occ = sdsl::locate(csa, P.begin(), P.end());
    for (auto o : occ) {
        std::cout << "Match: " << o << std::endl;
    }

    return 0;
}
