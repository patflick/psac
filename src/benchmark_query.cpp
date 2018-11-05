#include <vector>
#include <string>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>

// mine
#include "seq_query.hpp"

// SDSL
#include <sdsl/suffix_arrays.hpp>


std::string file2string(const std::string& filename) {
    // read input file into in-memory string
    std::string input_str;
    std::ifstream t(filename.c_str());
    std::stringstream buffer;
    buffer << t.rdbuf();
    input_str = buffer.str();
    return input_str;
}


template <typename Idx>
void construct(Idx& idx, const std::string& filename, int x = 1) {
    assert(x==1);
    std::string input_str = file2string(filename);
    // construct SA and LCP
    idx.construct(input_str.begin(), input_str.end());
}

std::vector<std::string> rand_patterns(const std::string& str, size_t len, size_t num) {
    assert(str.size() > len);
    std::vector<std::string> patterns;
    patterns.reserve(num);
    std::minstd_rand g;
    std::uniform_int_distribution<size_t> d(0, str.size()-len-1);
    for (size_t i = 0; i < num; ++i) {
        size_t start = d(g);
        patterns.emplace_back(str.begin() + start, str.begin() + start + len);
    }
    return patterns;
}

template <typename Idx>
inline void locate(Idx& idx, const std::string& P) {
    auto occ = idx.locate(P);
}


// TODO:
template <typename Idx>
void benchmark(const std::string& input_str, const std::vector<std::string>& Ps) {
    my::timer t;
    t.tic();
    Idx idx;
    idx.construct(input_str.begin(), input_str.end());
    //construct(idx, input_str);
    t.toc();
    std::cout << "construct " << t.get_ns() << " ns" << std::endl;

    t.tic();
    for (size_t i = 0; i < Ps.size(); ++i) {
        idx.locate(Ps[i]);
    }
    t.toc();
    std::cout << "query avg " << t.get_ns()/Ps.size() << " ns" << std::endl;
    std::cout << "avg rmq time " << idx.rmqtimer.get_total_ns()/Ps.size() << " ns" << std::endl;
}

template <typename csa_t>
void benchmark_sdsl(const std::string& filename, const std::vector<std::string>& Ps) {
    my::timer t;
    t.tic();
    csa_t csa;
    sdsl::construct(csa, filename, 1);
    t.toc();
    std::cout << "construct " << t.get_ns() << " ns" << std::endl;

    t.tic();
    for (size_t i = 0; i < Ps.size(); ++i) {
        //auto occ = sdsl::locate(csa, Ps[i].begin(), Ps[i].end());
        size_t numocc = sdsl::count(csa, Ps[i].begin(), Ps[i].end());
    }
    t.toc();
    std::cout << "query avg " << t.get_ns()/Ps.size() << " ns" << std::endl;

}

void benchmark_all(const std::string& filename) {
    // generate patterns
    size_t len = 20;
    size_t num = 100;
    std::string input_str = file2string(filename);
    std::vector<std::string> Ps = rand_patterns(input_str, len, num);
    using index_t = uint64_t;

    std::cout << "Benchmarking sa_index<index_t>" << std::endl;
    benchmark<sa_index<index_t>>(input_str, Ps);

    std::cout << "Benchmarking bi_esa_index<index_t>" << std::endl;
    benchmark<bs_esa_index<index_t>>(input_str, Ps);

    std::cout << "Benchmarking esa_index<index_t>" << std::endl;
    benchmark<esa_index<index_t>>(input_str, Ps);

    std::cout << "Benchmarking desa_index<index_t>" << std::endl;
    benchmark<desa_index<index_t>>(input_str, Ps);

    std::cout << "Benchmarking lookup_desa_index<index_t>" << std::endl;
    benchmark<lookup_desa_index<index_t>>(input_str, Ps);

    std::cout << "Benchmarking sdsl::csa_wt<>" << std::endl;
    using csa_t = sdsl::csa_wt<>;
    benchmark_sdsl<csa_t>(filename, Ps);

    std::cout << "Benchmarking sdsl::csa_sada<>" << std::endl;
    benchmark_sdsl<sdsl::csa_sada<>>(filename, Ps);

    std::cout << "Benchmarking sdsl uncompressed SA" << std::endl;
    benchmark_sdsl<sdsl::csa_bitcompressed<sdsl::byte_alphabet>>(filename, Ps);


    std::cout << "Benchmarking sdsl csa_wt 1x1" << std::endl;
    constexpr int sa_sample = 1;
    constexpr int isa_sample = 1;
    using csa2_t = sdsl::csa_wt<sdsl::wt_huff<>, sa_sample, isa_sample>;
    benchmark_sdsl<csa2_t>(filename, Ps);
}

void benchmark_desa(const std::string& filename) {
    size_t len = 20;
    size_t num = 100;
    size_t reps = 100000;
    std::string input_str = file2string(filename);
    std::vector<std::string> Ps = rand_patterns(input_str, len, num);

    my::timer t;
    t.tic();
    desa_index<uint64_t> idx;
    idx.construct(input_str.begin(), input_str.end());
    //construct(idx, input_str);
    t.toc();
    std::cout << "construct " << t.get_ns()/1.e9 << " s" << std::endl;

    t.tic();
    for (size_t rep = 0; rep < reps; ++rep) {
        for (size_t i = 0; i < Ps.size(); ++i) {
            idx.locate(Ps[i]);
        }
    }
    t.toc();
    std::cout << "query total time: " << t.get_ns()/1.e9 << " s" << std::endl;
    std::cout << "query avg " << t.get_ns()/Ps.size()/reps << " ns" << std::endl;
    std::cout << "avg rmq time " << idx.rmqtimer.get_total_ns()/Ps.size()/reps << " ns" << std::endl;
}


int main(int argc, char *argv[])
{
    std::string filename(argv[1]);

    //benchmark_desa(filename);
    benchmark_all(filename);
}
