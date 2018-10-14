
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <utility>

#include <mxx/env.hpp>
#include <mxx/comm.hpp>
#include <mxx/file.hpp>
#include <mxx/utils.hpp>

#include <suffix_array.hpp>
#include <lcp.hpp>
#include <divsufsort_wrapper.hpp>

#include "seq_query.hpp"
#include "desa.hpp"


int main(int argc, char *argv[]) {
    mxx::env e(argc, argv);
    mxx::env::set_exception_on_error();
    mxx::comm c;
    mxx::print_node_distribution(c);

    // given a file, compute suffix array and lcp array, how to do count/locate query?
    // via sa_range()
    if (argc < 3) {
        std::cerr << "Usage: ./xx <text_file> <pattern_file>" << std::endl;
    }
    mxx::section_timer t;

    // read input file into in-memory string
    std::string filename(argv[1]);
    std::string input_str = mxx::file_block_decompose(filename.c_str(), c);

    t.end_section("read input file");

    // construct SA and LCP
    using index_t = uint64_t;
    dist_desa<index_t> idx(c);
    using range_t = dist_desa<index_t>::range_t;
    idx.construct(input_str.begin(), input_str.end(), c);
    t.end_section("construct idx");

    std::string pattern_file(argv[2]);
    strings ss = strings::from_dfile(pattern_file, c);
    size_t total_num = mxx::allreduce(ss.nstrings, c);
    t.end_section("read patterns file");

    // run locate a couple times
    int reps = 10;
    for (int i = 0; i < reps; ++i) {
        std::vector<range_t> mysols = idx.bulk_locate(ss);
        t.end_section("bulk_locate");
    }

    return 0;
}
