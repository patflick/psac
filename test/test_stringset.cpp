
#include <gtest/gtest.h>
#include <mxx/comm.hpp>
#include <mxx/distribution.hpp>

#include <iostream>
#include <vector>
#include <string>

#include <stringset.hpp>
#include <shifting.hpp>
#include <kmer.hpp>

#include <cxx-prettyprint/prettyprint.hpp>


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

#define SDEBUG(x) mxx::sync_cerr(c) << "[" << c.rank() << "]: " #x " = " << (x) << std::endl

std::string rand_dna(size_t size) {
    std::string result;
    result.resize(size);
    char alpha[4] = {'a', 'c', 't', 'g'};
    for (size_t i = 0; i < size; ++i) {
        result[i] = alpha[rand() % sizeof(alpha)];
    }
    return result;
}


std::string randseq_1(const mxx::comm& c) {
    std::vector<size_t> ssizes;
    if (c.rank() == 0) {
        ssizes = {88, 57, 8, 20, 3, 4, 1, 2, 3, 1, 1, 11};
    }
    std::string randseq;
    if (c.rank() == 0) {
        std::vector<std::string> strs;
        for (size_t s : ssizes) {
            strs.emplace_back(rand_dna(s));
        }

        std::cout << strs << std::endl;
        std::string flatstr = flatten_strings(strs);
        std::cout << "Flat str: \"" << flatstr << "\"" << std::endl;
        // vec of string to strings seperated by $
        randseq = flatstr;
    }
    randseq = mxx::stable_distribute(randseq, c);
    return randseq;
}

std::string randseq_2(const mxx::comm& c) {
    std::vector<size_t> ssizes;
    if (c.rank() == 0) {
        ssizes = {88, 57, 8, 20, 3, 4, 1, 2, 3, 1, 1, 11};
    }
    mxx::stable_distribute_inplace(ssizes, c);
    std::string randseq;
    std::vector<std::string> strs;
    for (size_t s : ssizes) {
        strs.emplace_back(rand_dna(s));
    }

    std::cout << strs << std::endl;
    std::string flatstr = flatten_strings(strs);
    mxx::sync_cout(c) << "Flat str: \"" << flatstr << "\"" << std::endl;
    // vec of string to strings seperated by $
    randseq = flatstr;
    //randseq = mxx::stable_distribute(randseq, c);
    return randseq;
}

TEST(PsacDistStringSet, TestSimpleKmers) {
    mxx::comm comm;
    // max with 5 processes:
    for (int comm_size = 1; comm_size <= 5; ++comm_size) {
        comm.with_subset(comm.rank() < comm_size, [&](const mxx::comm& c){
            std::vector<std::string> strs = {"abc", "cba", "bbbb", "a", "c", "abab", "cb"};
            std::string flatstrs;
            if (c.rank() == 0) {
                //strs = {"accctgca", "aca", "t", "gct"};
                flatstrs = flatten_strings(strs);
            }
            flatstrs = mxx::stable_distribute(flatstrs, c);

            // create stringset, dist_seq
            simple_dstringset ss(flatstrs.begin(), flatstrs.end(), c);

            // expected coding: a: 01, b: 10, c: 11
            // "abc" -> 00011011 -> 1b
            //          00101100 -> 2c
            //          00110000 -> 30
            // "cba" -> 00111001 -> 39
            //          00100100 -> 24
            //          00010000 -> 10
            // "bbbb"-> 10101010 -> 2a 2a 28 20
            // "a"   -> 00010000 -> 10
            // "c"   -> 00110000 -> 30
            // "abab"-> 01100110 -> 19 26 18 20
            // "cb"  -> 00111000 -> 38 20
            // => [1b, 2c, 30, 39, 24, 10, 2a, 2a, 28, 20, 19, 26, 18, 20, 10, 30, 20] (hex)
            std::vector<uint16_t> ex_kmers = {0x1b, 0x2c, 0x30, 0x39, 0x24,0x10,
                                              0x2a, 0x2a, 0x28, 0x20, 0x10, 0x30,
                                              0x19, 0x26, 0x18, 0x20, 0x38, 0x20};
            alphabet<char> alpha = alphabet<char>::from_string("abc", c);
            std::vector<uint16_t> kmers = kmer_gen_stringset<uint16_t>(ss, 3, alpha, c);

            std::vector<uint16_t> all_kmers = mxx::gatherv(kmers, 0, c);
            if (c.rank() == 0) {
                EXPECT_EQ(ex_kmers, all_kmers) << " with comm size: " << comm_size;
            }
        });
        comm.barrier();
    }
}

void test_dist_ss() {
    mxx::comm c;
    //std::string randseq = random_dstringset(20, c);
    std::string randseq = randseq_1(c);

    // generate strings of given sizes on master node, join into single string
    // with '$' as seperator and distribute equally among processors

    // construct distribute stringset by parsing the string according to
    // '$' separating character
    simple_dstringset ss(randseq.begin(), randseq.end(), c);

    SDEBUG(randseq);
    SDEBUG(ss.sizes);

    // TODO; create vector from stringset (everything but the separating characters)
    size_t num_b = std::accumulate(ss.sizes.begin(), ss.sizes.end(), 0);
    //std::string vec;
    std::vector<char> vec;
    vec.resize(num_b);
    auto oit = vec.begin();
    for (size_t i = 0; i < ss.sizes.size(); ++i) {
        oit = std::copy(ss.str_begins[i], ss.str_begins[i]+ss.sizes[i], oit);
    }

    // create the distributed sequences prefix_sizes format (with shadow els)
    dist_seqs ds = dist_seqs::from_dss(ss, c);

    // gather sizes and print
    std::vector<size_t> all_sizes = mxx::gatherv(ds.sizes(), 0, c);
    mxx::sync_cout(c) << ds << std::endl;
    if (c.rank() == 0) {
        std::cout << all_sizes << std::endl;
    }

    mxx::stable_distribute_inplace(vec, c);

    // try printing out original strings
    mxx::sync_cout(c) << vec << std::endl;
    // TODO: try to re-create strings via the dist_seqs object
    // send string to owner of the sequence (where it starts)
    // or rather: insert '$' at end of every string, and then just allgather?
    std::vector<char> allstr = mxx::gatherv(vec, 0, c);
    if (c.rank() == 0) {
        std::vector<std::string> strings;
        auto it = allstr.begin();
        for (size_t i = 0; i < all_sizes.size(); ++i) {
            strings.emplace_back(it, it + all_sizes[i]);
            it += all_sizes[i];
        }
        std::cout << strings << std::endl;
    }
}


TEST(PsacDistStringSet, DSKmerGen) {
    mxx::comm c;

    // create input
    std::vector<std::string> strs;
    std::string flatstrs;
    if (c.rank() == 0) {
        strs = {"cctgtggtataagagctttgggctttcgcagtcccgactagtctgaacttacccagactcccagtctgtagtgaataaggtgaaaaga", "tttggtttgcctcaaacatcccagacgccgcgcggacctctggaagacggtaagaca", "gtctgcgg", "aaactcataatgagggcgaa", "gca", "ggtc", "t", "gc", "cgc", "t", "a", "ggacaaggctt"};
        flatstrs = flatten_strings(strs);
    }
    flatstrs = mxx::stable_distribute(flatstrs, c);

    // create stringset, dist_seq
    simple_dstringset ss(flatstrs.begin(), flatstrs.end(), c);
    dist_seqs ds = dist_seqs::from_dss(ss, c);

    unsigned int k = 4;
    // create 4-mers
    alphabet<char> a = alphabet<char>::from_string("actg", c);
    std::vector<uint16_t> kmers = kmer_gen_stringset<uint16_t>(ss, k, a, c);

    mxx::stable_distribute_inplace(kmers, c);

    // construct kmers per string and also gather to root
    std::vector<uint16_t> skmers;
    for (auto s : strs) {
        std::vector<uint16_t> sk = kmer_generation<uint16_t>(s.begin(), s.end(), k, a);
        skmers.insert(skmers.end(), sk.begin(), sk.end());
    }

    // distribute skmers similar to kmers and then they should be equal
    mxx::stable_distribute_inplace(skmers, c);
    EXPECT_EQ(skmers, kmers);

    // next up: test shifting of kmers utilizing the dist_seqs representation
    size_t shift_by = 3;
    std::vector<uint16_t> b = shift_buckets_ds(ds, kmers, shift_by, c);
    //
    std::vector<std::vector<uint16_t>> kmer_vecs = gather_dist_seq(ds, kmers, c);
    std::vector<std::vector<uint16_t>> shift_vecs = gather_dist_seq(ds, b, c);

    ASSERT_EQ(kmer_vecs.size(), shift_vecs.size());
    if (c.rank() == 0) {
        for (size_t i = 0; i < kmer_vecs.size(); ++i) {
            ASSERT_EQ(kmer_vecs[i].size(), shift_vecs[i].size());
            for (size_t j = shift_by; j < shift_vecs[i].size(); ++j) {
                EXPECT_EQ(kmer_vecs[i][j], shift_vecs[i][j-shift_by]);
            }
            for (size_t j = 0; j < std::min(shift_vecs[i].size(), shift_by); ++j) {
                EXPECT_EQ(shift_vecs[i][shift_vecs[i].size()-j-1], 0);
            }
        }
    }
}

TEST(PsacDistStringSet, DSbuckets) {
    mxx::comm c;
    std::vector<int> b;
    if (c.is_first()) {
        b = {0, 0, 1, 1, 1, 1, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3};
    }
    b = mxx::stable_distribute(b, c);

    dist_seqs_buckets db = dist_seqs_buckets::from_func(b, c);

    SDEBUG(b);
    SDEBUG(db);
}


