
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

#include <seq_query.hpp>
#include <rmq.hpp>
#include <lcp.hpp>
#include <divsufsort_wrapper.hpp>
//#include <suffix_array.hpp>
//#include <ansv.hpp>

#include "seq_query.hpp"
#include "desa.hpp"


template <typename index_t>
struct tl_desa {
    index_t total_size;
    std::vector<char> lc;
    std::vector<index_t> lcp;
    std::vector<index_t> off; // offsets

    using range_t = std::pair<index_t, index_t>;

    using it_t = typename std::vector<index_t>::const_iterator;
    rmq<it_t, index_t> minq;

    void construct(const std::vector<index_t>& LCP, const std::vector<char> Lc, index_t maxsize) {
        // local ansv !?
        index_t n = LCP.size();
        total_size = n;

        struct node {
            index_t lcp;
            index_t pos;
            index_t l;
        };

        std::stack<node> st;
        st.push(node{0,0,0,0});

        index_t total_out = 1;
        std::vector<bool> do_output(n, false);
        do_output[0] = true;

        for (index_t i = 1; i < n; ++i) {
            if (LCP[i] == 0) {
                do_output[i] = true;
                continue;
            }
            while (!st.empty() && st.top().lcp > LCP[i]) {
                node u = st.top();
                st.pop();
                // u.pos has range [u.l, .. , i)
                index_t parent_size = i - u.l;
                if (parent_size > maxsize) {
                    // output but in inverse order !?
                    do_output[u.pos] = true;
                    ++total_out;
                }
            }

            if (st.empty()) {
                // cant happen
                assert(false);
            } else if (st.top().lcp == LCP[i]) {
                st.push(node{LCP[i], i, st.top().l});
            } else {
                assert(st.top().lcp < LCP[i]);
                st.push(node{LCP[i], i, st.top().pos});
            }
        }

        std::cout << "creating TL DT with " << total_out << " els" << std::endl;
        lcp.resize(total_out);
        lc.resize(total_out);
        off.resize(total_out);

        index_t j = 0;
        for (index_t i = 0; i < n; ++i) {
            if (do_output[i]) {
                lcp[j] = sa.LCP[i];
                lc[j] = sa.Lc[i];
                off[j] = i;
                ++j;
            }
        }

        // constrct RMQ over new sampled LCP
        minq = rmq<it_t,index_t>(lcp.begin(), lcp.end());
    }


    range_t locate(const std::string& P) {

        size_t n = lcp.size();
        size_t m = P.size();
        size_t l = 0;
        size_t r = n-1;

        // get first child interval and depth
        size_t i = this->minq(l+1, r);
        index_t q = this->lcp[i];

        // blind search
        while (q < m && l < r) {

            // NOTE: LCP[i] = lcp(SA[i-1],SA[i]), LCP[0] = 0
            // using [l,r] as an inclusive SA range
            // corresponding to LCP query range [l+1,r]

            // check if we've reached the end of the pattern
            if (q >= m) {
                break;
            }

            do {
                // `i` is the lcp(SA[i-1],SA[i])
                char lc = this->lc[i]; // == S[SA[l]+lcpv] for first iter
                if (lc == P[q]) {
                    r = i-1;
                    break;
                }
                l = i;
                if (l == r)
                    break;

                i = this->minq(l+1, r);
            } while (l < r && this->lcp[i] == q);

            if (this->lcp[i] == q) {
                if (l+1 < r) {
                    i = this->minq(l+1, r);
                } else {
                    i = l;
                }
            }
            q = this->lcp[i];
        }

        // return the range using offsets
        if (r == n-1) {
            return range_t(off[l], total_size);
        } else {
            return range_t(off[l], off[r+1]);
        }
    }
};


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
        using index_t = uint64_t;
        using range_t = dist_desa<index_t>::range_t;
        dist_desa<index_t> idx(c);

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
