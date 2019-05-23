
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <utility>

// using TCLAP for command line parsing
#include <tclap/CmdLine.h>

// mxx dependencies
#include <mxx/env.hpp>
#include <mxx/comm.hpp>
#include <mxx/file.hpp>
#include <mxx/utils.hpp>

#include <suffix_array.hpp>
#include <lcp.hpp>
#include <divsufsort_wrapper.hpp>

#include "seq_query.hpp"
#include "desa.hpp"
#include "tldt.hpp"

// define index and TLI types for experiment
using index_t = uint64_t;
using TLI_t = tldt<index_t>;
//using TLI_t = tllt<index_t>;
using desa_t = dist_desa<index_t, TLI_t>;

int main(int argc, char *argv[]) {
    // set up mxx / MPI
    mxx::env e(argc, argv);
    mxx::env::set_exception_on_error();
    mxx::comm c;
    mxx::print_node_distribution(c);

    // given a file, compute suffix array and lcp array, how to do count/locate query?
    // via sa_range()
    /*
    if (argc < 3) {
        std::cerr << "Usage: ./xx <text_file> <pattern_file>" << std::endl;
    }
    */
    try {
        // define commandline usage
        TCLAP::CmdLine cmd("Distributed Enhanced Suffix Array");
        TCLAP::ValueArg<std::string> fileArg("f", "file", "Input string filename.", true, "", "filename");
        cmd.add(fileArg);
        TCLAP::ValueArg<std::string> loadArg("l", "load", "Load index from given basename", false, "", "filename");
        TCLAP::SwitchArg constructArg("c", "construct", "Construct SA/LCP/Lc from input file", false);
        cmd.xorAdd(loadArg, constructArg); // either load or construct SA/LCP
        TCLAP::ValueArg<std::string> outArg("o", "outfile", "Output file base name. If --construct was used, this stores the resulting DESA.", false, "", "filename");
        cmd.add(outArg);

        TCLAP::ValueArg<std::string> queryArg("q", "query", "Query file for benchmarking querying.", false, "", "filename");
        cmd.add(queryArg);
        cmd.parse(argc, argv);


        mxx::section_timer t;

        // create distributed DESA class
        using range_t = desa_t::range_t;
        desa_t idx(c);

        if (constructArg.getValue()) {
            if (c.rank() == 0) {
                std::cout << "constructing DESA (SA+LCP+LC)..." << std::endl;
            }
            // read input file into in-memory string
            std::string input_str = mxx::file_block_decompose(fileArg.getValue().c_str(), c);
            t.end_section("read input file");
            // construct DESA from scratch
            idx.construct(input_str.begin(), input_str.end(), c);
            t.end_section("construct idx");

            if (outArg.getValue() != "") {
                // store DESA to given basename
                if (c.rank() == 0) {
                    std::cout << "saving DESA to basename `" << outArg.getValue() << "` ..." << std::endl;
                }
                idx.write(outArg.getValue(), c);
            }
        } else {
            if (outArg.getValue() != "") {
                if (c.rank() == 0) {
                    std::cerr << "WARNING: --outfile argument will be ignored since the input is loaded from file (don't use in conjuction with --load)." << std::endl;
                }
            }
            if (c.rank() == 0) {
                std::cout << "loading DESA (SA+LCP+LC) from basename `" << loadArg.getValue() << "` ..." << std::endl;
            }
            idx.read(fileArg.getValue(), loadArg.getValue(), c);

        }

        // query benchmarking?
        if (queryArg.getValue() != "") {
            strings ss = strings::from_dfile(queryArg.getValue(), c);
            t.end_section("read patterns file");

            // run locate a couple of times
            int reps = 10;
            for (int i = 0; i < reps; ++i) {
                std::vector<range_t> mysols = idx.bulk_locate(ss);
                t.end_section("bulk_locate");
            }
        }

    } catch (TCLAP::ArgException& e) {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
        exit(EXIT_FAILURE);
    }

    return 0;
}
