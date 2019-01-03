/*
 * Copyright 2015 Georgia Institute of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>
#include <mxx/comm.hpp>
#include <mxx/distribution.hpp>

// disable timer output during testing
#define MXX_DISABLE_TIMER 1

#include <alphabet.hpp>
#include <suffix_array.hpp>
#include <divsufsort_wrapper.hpp>
#include <check_suffix_array.hpp>
#include <lcp.hpp>


template <typename char_t, typename index_t, bool _LCP>
bool check_sa_dss(suffix_array<char_t, index_t, _LCP>& sa, const std::string& str, const mxx::comm& c) {
    // gather SA back to root process
    std::vector<index_t> gsa = mxx::gatherv(sa.local_SA, 0, c);

    // check correctness using dss
    bool dss_correct = true;
    if (c.rank() == 0) {
        dss_correct = dss::check(str.begin(), str.end(), gsa);
    }

    return mxx::all_of(dss_correct, c);

    // check correctness using own method
    // TODO return bool value instead of exit(FAILURE) in that function
    //bool correct = gl_check_correct(sa, local_str.begin(), local_str.end(), c);
}

template <typename char_t, typename index_t, bool _LCP>
bool check_lcp_eq(suffix_array<char_t, index_t, _LCP>& sa, const std::string& local_str, const mxx::comm& c) {
    // gather LCP back to root process
    std::vector<index_t> gsa = mxx::gatherv(sa.local_SA, 0, c);
    std::vector<index_t> gisa = mxx::gatherv(sa.local_B, 0, c);
    std::vector<index_t> glcp;
    if (_LCP)
        glcp = mxx::gatherv(sa.local_LCP, 0, c);
    // gather string
    // TODO: use iterator or std::string version for mxx?
    std::vector<char> global_str_vec = mxx::gatherv(&(*local_str.begin()), local_str.size(), 0, c);
    std::string gstr(global_str_vec.begin(), global_str_vec.end());
    bool sa_correct = true;
    if (c.rank() == 0) {
        sa_correct = dss::check(gstr.begin(), gstr.end(), gsa);
    }

    // check LCP
    bool lcp_correct = true;
    if (_LCP) {
        if (c.rank() == 0) {
            lcp_correct = check_lcp(gstr, gsa, gisa, glcp);
        }
    }
    return sa_correct && lcp_correct;
}

template <typename char_t, typename index_t, bool _LCP>
bool check_sa_eqdss(suffix_array<char_t, index_t, _LCP>& sa, const std::string& str, const mxx::comm& c) {
    // gather SA back to root process
    std::vector<index_t> gsa = mxx::gatherv(sa.local_SA, 0, c);

    // check correctness using dss
    bool dss_correct = true;
    if (c.rank() == 0) {
        std::vector<index_t> true_sa;
        dss::construct(str.begin(), str.end(), true_sa);
        if (true_sa.size() != gsa.size()) {
            dss_correct = false;
        } else {
            for (size_t i = 0; i < true_sa.size(); ++i) {
                if (true_sa[i] != gsa[i]) {
                    dss_correct = false;
                    break;
                }
            }
        }
    }
    return mxx::all_of(dss_correct, c);
}


TEST(PSAC, Mississippi) {
    mxx::comm c;


    std::vector<uint32_t> exp = {10, 7, 4, 1, 0, 9, 8, 6, 3, 5, 2};
    std::string str;
    if (c.rank() == 0) {
        // generate some random input string
        str = "mississippi";
    }
    size_t size = 11;

    bool correct = true;
    c.with_subset((size_t)c.rank() < size, [&](const mxx::comm& subcomm) {
        // distribute string equally
        std::string local_str = mxx::stable_distribute(str, subcomm);

        // create suffix array w/o LCP
        suffix_array<char, uint32_t, false> sa(subcomm);
        sa.construct(local_str.begin(), local_str.end());

        //correct = check_sa_dss(sa, str, subcomm);
        std::vector<uint32_t> glsa = mxx::gatherv(sa.local_SA, 0, subcomm);
        if (subcomm.rank() == 0) {
            correct = (glsa == exp);
        }
    });
    ASSERT_TRUE(mxx::all_of(correct, c));
}

TEST(PSAC, RandAll) {
    mxx::comm c;

    size_t size = 130370;

    std::string str;
    if (c.rank() == 0) {
        // generate some random input string
        str = rand_dna(size, 7);
    }
    // distribute string equally
    std::string local_str = mxx::stable_distribute(str, c);

    // create suffix array w/o LCP
    suffix_array<char, uint32_t, false> sa(c);

    // construct suffix array
    sa.construct(local_str.begin(), local_str.end());
    ASSERT_TRUE(check_sa_dss(sa, str, c));
    ASSERT_TRUE(check_sa_eqdss(sa, str, c));

    // construct with custom `k` to force early bucket chaising
    sa.construct(local_str.begin(), local_str.end(), true, 3);
    ASSERT_TRUE(check_sa_dss(sa, str, c));
    ASSERT_TRUE(check_sa_eqdss(sa, str, c));

    // construct without bucket chaising
    sa.construct(local_str.begin(), local_str.end(), false, 2);
    ASSERT_TRUE(check_sa_dss(sa, str, c));

    // construct with std::array based construction
    sa.construct_arr<2>(local_str.begin(), local_str.end(), true);
    ASSERT_TRUE(check_sa_dss(sa, str, c));

    // construct with prefix-tripling
    sa.construct_arr<3>(local_str.begin(), local_str.end(), true);
    ASSERT_TRUE(check_sa_dss(sa, str, c));

    // construct with prefix-quadrupling
    sa.construct_arr<4>(local_str.begin(), local_str.end(), false);
    ASSERT_TRUE(check_sa_dss(sa, str, c));

    //TODO this one fails
    //sa.construct_fast();
    //ASSERT_TRUE(check_sa_dss(sa, str, c));
}

TEST(PSAC, RepeatsAll) {
    mxx::comm c;

    // create stirng with many repeats
    std::string str;
    if (c.rank() == 0) {
        std::vector<std::string> strings = {"helloworld", "blahlablah", "ellow", "worldblah", "rld", "hello" };
        // generate some random input string
        size_t numstr = 15000;
        std::stringstream ss;
        for (size_t i = 0; i < numstr; ++i) {
            ss << strings[std::rand() % strings.size()];
        }
        str = ss.str();
    }
    // distribute string equally
    std::string local_str = mxx::stable_distribute(str, c);

    // create suffix array w/o LCP
    suffix_array<char, uint64_t, false> sa(c);

    // construct suffix array
    sa.construct(local_str.begin(), local_str.end());
    ASSERT_TRUE(check_sa_dss(sa, str, c));
    ASSERT_TRUE(check_sa_eqdss(sa, str, c));

    // construct with custom `k` to force early bucket chaising
    sa.construct(local_str.begin(), local_str.end(), true, 3);
    ASSERT_TRUE(check_sa_dss(sa, str, c));
    ASSERT_TRUE(check_sa_eqdss(sa, str, c));

    // construct without bucket chaising
    sa.construct(local_str.begin(), local_str.end(), false, 2);
    ASSERT_TRUE(check_sa_dss(sa, str, c));

    // construct with std::array based construction
    sa.construct_arr<2>(local_str.begin(), local_str.end(), true);
    ASSERT_TRUE(check_sa_dss(sa, str, c));

    // construct with prefix-tripling
    sa.construct_arr<3>(local_str.begin(), local_str.end(), true);
    ASSERT_TRUE(check_sa_dss(sa, str, c));

    // construct with prefix-quadrupling
    sa.construct_arr<3>(local_str.begin(), local_str.end(), false);
    ASSERT_TRUE(check_sa_dss(sa, str, c));
}

TEST(PSAC, SmallStrings) {
    mxx::comm c;

    size_t size = 9;

    std::string str;
    if (c.rank() == 0) {
        // generate some random input string
        str = rand_dna(size, 13);
    }
    bool correct = true;
    c.with_subset((size_t)c.rank() < size, [&](const mxx::comm& subcomm) {
        // distribute string equally
        std::string local_str = mxx::stable_distribute(str, subcomm);

        // create suffix array w/o LCP
        suffix_array<char, uint32_t, false> sa(subcomm);
        sa.construct(local_str.begin(), local_str.end());

        correct = check_sa_dss(sa, str, subcomm);
    });
    ASSERT_TRUE(mxx::all_of(correct, c));
}

TEST(PSAC, Lcp1) {
    mxx::comm c;

    size_t size = 66763;

    std::string str;
    if (c.rank() == 0) {
        // generate some random input string
        str = rand_dna(size, 23);
    }
    // distribute string equally
    std::string local_str = mxx::stable_distribute(str, c);

    // create suffix array with LCP
    suffix_array<char, uint64_t, true> sa(c);

    // construct suffix  and LCP array
    sa.construct(local_str.begin(), local_str.end());
    EXPECT_TRUE(check_sa_dss(sa, str, c));
    EXPECT_TRUE(check_lcp_eq(sa, local_str, c));
    // construct suffix and LCP array using artificial small `k` to force bucket chaising
    sa.construct(local_str.begin(), local_str.end(), true, 3);
    EXPECT_TRUE(check_sa_dss(sa, str, c));
    EXPECT_TRUE(check_lcp_eq(sa, local_str, c));
}


TEST(PSAC, IntAlphabetMiss) {
    mxx::comm c;

    std::vector<unsigned int> exp;
    std::vector<int> str;
    if (c.rank() == 0) {
        // generate some random input string
        // mississippi: i = 3, m = 128, p = 66000, s = 12345678
        exp = {10, 7, 4, 1, 0, 9, 8, 6, 3, 5, 2};
        str = std::vector<int>({128, 3, 12345678, 12345678, 3, 12345678, 12345678, 3, 66000, 66000, 3});
    }
    bool correct = true;
    c.with_subset((size_t)c.rank() < 11, [&](const mxx::comm& subcomm) {
        // distribute string equally
        std::vector<int> local_str = mxx::stable_distribute(str, subcomm);
        std::vector<unsigned int> local_exp_sa = mxx::stable_distribute(exp, subcomm);

        // create suffix array with LCP
        suffix_array<int, unsigned int, true> sa(subcomm);

        // construct suffix  and LCP array
        sa.construct(local_str.begin(), local_str.end());

        // check correctness
        correct = (local_exp_sa == sa.local_SA);
    });
    ASSERT_TRUE(mxx::all_of(correct, c));
}

TEST(PSAC, FileIO) {
    mxx::comm gc;
    std::string s;
    if (gc.rank() == 0) {
        s = "mississippi";
    }
    gc.with_subset(gc.rank() < 11, [&](const mxx::comm& c) {
        std::string local_s = mxx::stable_distribute(s, c);

        suffix_array<char, size_t, true, true> sa(c);
        sa.construct(local_s.begin(), local_s.end());

        sa.write("miss");

        suffix_array<char, size_t, true, true> sa2(c);
        sa2.read("miss");

        EXPECT_EQ(sa.local_SA, sa2.local_SA);
        EXPECT_EQ(sa.local_LCP, sa2.local_LCP);
        EXPECT_EQ(sa.local_Lc, sa2.local_Lc);

        EXPECT_EQ(sa.alpha, sa2.alpha);

        std::vector<size_t> gsa = mxx::gatherv(sa.local_SA, 0, c);
        std::vector<size_t> glcp = mxx::gatherv(sa.local_LCP, 0, c);
        std::vector<char> glc = mxx::gatherv(sa.local_Lc, 0, c);

        if (c.size() > 2) {
            c.with_subset(c.rank() < c.size() - 2, [&](const mxx::comm& sc) {
                suffix_array<char, size_t, true, true> sa3(sc);
                sa3.read("miss");
                EXPECT_EQ(sa.alpha, sa3.alpha);
                // gather and check
                EXPECT_EQ(gsa, mxx::gatherv(sa3.local_SA, 0, sc));
                EXPECT_EQ(glcp, mxx::gatherv(sa3.local_LCP, 0, sc));
                EXPECT_EQ(glc, mxx::gatherv(sa3.local_Lc, 0, sc));
            });
        }
    });

    // try with different number of processors for write/read
}
