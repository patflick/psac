
#include <gtest/gtest.h>
#include <mxx/env.hpp>
#include <mxx/comm.hpp>

#include <suffix_array.hpp>
#include <stringset.hpp>

#include <cxx-prettyprint/prettyprint.hpp>

#define SDEBUG(x) mxx::sync_cerr(c) << "[" << c.rank() << "]: " #x " = " << (x) << std::endl

std::string repeat(const std::string& str, size_t rep) {
    std::string res;
    res.resize(str.size()*rep);
    auto oit = res.begin();
    for (size_t i = 0; i < rep; ++i) {
        oit = std::copy(str.begin(), str.end(), oit);
    }
    return res;
}

std::vector<std::string> repeat_inc_seq(const std::string& seq = "ab", size_t reps = 3) {
    std::vector<std::string> res;
    for (size_t i = 0; i < reps; ++i) {
        res.emplace_back(repeat(seq, i+1));
    }
    return res;
}

std::vector<size_t> repeat_inc_gsa(size_t slen, size_t reps) {
    size_t m = reps*(reps+1)/2;
    size_t n = slen*m;
    std::vector<size_t> gsa(n);
    for (size_t i = 0; i < slen; ++i) {
        auto oit = gsa.begin()+i*m;
        for (size_t j = 0; j < reps; ++j) {
            size_t ss = i + slen*(j*(j+1))/2;
            *oit = ss;
            ++oit;
            for (size_t k = j+2; k <= reps; ++k) {
                *oit = *(oit-1) + k*slen;
                ++oit;
            }
        }
    }
    return gsa;
}

TEST(TestGSA, SimpleTiny) {
    mxx::comm comm;
    comm.with_subset(comm.rank() < 3, [](const mxx::comm& c) {

        // create input
        std::vector<std::string> strs;
        std::string flatstrs;
        if (c.rank() == 0) {
            strs = {"abab", "baba"};
            flatstrs = flatten_strings(strs);
        }
        flatstrs = mxx::stable_distribute(flatstrs, c);

        // create stringset, dist_seq
        simple_dstringset ss(flatstrs.begin(), flatstrs.end(), c);

        // construct GSA
        alphabet<char> a = alphabet<char>::from_string("ab", c);
        suffix_array<char, uint64_t> sa(c);
        sa.construct_ss(ss, a);
        SDEBUG(sa.local_SA);

        std::vector<uint64_t> sar = mxx::gatherv(sa.local_SA, 0, c);
        // check equal
        if (c.rank() == 0) {
            std::vector<uint64_t> ex_gsa = {7, 2, 5, 0, 3, 6, 1, 4};
            EXPECT_EQ(ex_gsa, sar);
        }
    });
}

void test_repeats(const std::string& seq, size_t reps, const mxx::comm& comm) {
    // create input
    size_t slen = seq.size()*reps*(reps+1)/2;
    ASSERT_GT(slen/4,0ul);

    comm.with_subset((size_t)comm.rank() < slen/4, [seq, reps](const mxx::comm& c) {

        std::vector<std::string> strs;
        std::string flatstrs;
        if (c.rank() == 0) {
            strs = repeat_inc_seq(seq, reps);
            flatstrs = flatten_strings(strs);
        }
        flatstrs = mxx::stable_distribute(flatstrs, c);

        // create stringset, dist_seq
        simple_dstringset ss(flatstrs.begin(), flatstrs.end(), c);

        // construct SA
        alphabet<char> a = alphabet<char>::from_string(seq, c);
        suffix_array<char, uint64_t> sa(c);
        sa.construct_ss(ss, a);

        // check equal
        std::vector<uint64_t> sar = mxx::gatherv(sa.local_SA, 0, c);
        if (c.rank() == 0) {
            std::vector<uint64_t> ex_gsa = repeat_inc_gsa(seq.size(), reps);
            EXPECT_EQ(ex_gsa, sar);
        }
    });
}

TEST(TestGSA, IncRepeatsAB3) {
    mxx::comm c;
    test_repeats("ab", 3, c);
}

TEST(TestGSA, IncRepeatsA20) {
    mxx::comm c;
    test_repeats("a", 20, c);
}

TEST(TestGSA, IncRepeatsABC10) {
    mxx::comm c;
    test_repeats("abc", 10, c);
}

TEST(TestGSA, IncRepeatsAF50) {
    mxx::comm c;
    test_repeats("abcdef", 50, c);
}
