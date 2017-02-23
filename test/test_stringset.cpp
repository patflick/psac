
#include <iostream>
#include <vector>
#include <string>

#include <stringset.hpp>
#include <shifting.hpp>
#include <kmer.hpp>
#include <mxx/env.hpp>
#include <mxx/comm.hpp>
#include <mxx/distribution.hpp>

#include <cxx-prettyprint/prettyprint.hpp>

void test_kmer() {
    std::vector<std::string> vec = {"abc", "cba", "bbbb", "a"};
    // expected coding: a: 01, b: 10, c: 11
    // "abc" -> 00011011 -> 1b
    //          00101100 -> 2c
    //          00110000 -> 30
    // "cba" -> 00111001 -> 39
    //          00100100 -> 24
    //          00010000 -> 10
    // "bbbb"-> 00101010 -> 2a 2a 28 20
    // "a"   -> 00010000 -> 10
    // => [1b, 2c, 30, 39, 24, 10, 2a, 2a, 28, 20, 10] (hex)
    std::vector<uint16_t> ex_kmers = {0x1b, 0x2c, 0x30, 0x39, 0x24,0x10,
                                      0x2a, 0x2a, 0x28, 0x20, 0x10};
    vstringset ss(vec);


    std::string unique_chars = "abc";
    alphabet<char> alpha = alphabet<char>::from_sequence(unique_chars.begin(), unique_chars.end()); // TODO from vec of chars
    std::cout << "Alphabet: " << alpha << std::endl;
    std::vector<uint16_t> kmers = kmer_gen_stringset<uint16_t>(ss, 3, alpha);

    std::cout << std::hex << kmers << std::endl;
}


void test_shift() {
    // generate input
    mxx::comm c;
    std::vector<int> x;
    int n = 23;
    if (c.rank() == 0) {
        x.resize(n);
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] = i;
        }
    }
    // distribute equally
    mxx::stable_distribute_inplace(x, c);

    for (int i = 0; i < n; ++i) {
        std::vector<int> r = left_shift_dvec(x, c, i);

        std::vector<int> g = mxx::gatherv(r, 0, c);
        // check g
        if (c.rank() == 0) {
            std::cout << "shift " << i << ": " << g << std::endl;
            for (unsigned int j = 0; j < g.size(); ++j) {
                if (g[j] != (int)j + i && !(i+(int)j >= n && g[j] == 0))  {
                    std::cout << "ERROR ERROR" << std::endl;
                }
            }
        }
    }

}

void test_global_copy() {
    // generate input
    mxx::comm c;
    std::vector<int> x;
    int n = 20;
    if (c.rank() == 0) {
        x.resize(n);
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] = i;
        }
    }
    // distribute equally
    mxx::stable_distribute_inplace(x, c);

    // TODO: create ranges + test global copies
    dvector_const_wrapper<int, blk_dist> src(x, c);
    dvector<int, blk_dist> dst(c, x.size());
    copy_global_range(src, 4, 13, dst, 1, 10);
    mxx::sync_cout(c) << dst.vec << std::endl;
}


std::vector<int> rand_buckets(size_t n, size_t max_bs, const mxx::comm& c) {
    std::vector<int> seq;
    if (c.rank() == 0) {
        seq.resize(n);
        size_t i = 0;
        while (i < n) {
            size_t bs = (rand() % std::min(max_bs, n-i)) + 1;
            for (size_t j = i; j < bs; ++j) {
                seq[j] = i;
            }
        }
    }
    mxx::stable_distribute_inplace(seq, c);
    return seq;
}

void test_shift_buckets() {
    mxx::comm c;

    // 1) generate random buckets
    size_t n = 20;
    size_t max_bs = 10;
    std::vector<int> b = rand_buckets(n, max_bs, c);

    // 2) wrap as distributed range (blk_dist)
    // 3) init distributed buckets by value
}

int main(int argc, char *argv[]) {

    mxx::env e(argc, argv);

    //test_shift();
    test_global_copy();

    return 0;
}
