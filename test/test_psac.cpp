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

#include <alphabet.hpp>
#include <suffix_array.hpp>
#include <divsufsort_wrapper.hpp>
#include <check_suffix_array.hpp>

template <typename Iterator, typename index_t, bool _LCP>
bool check_sa_dss(suffix_array<Iterator, index_t, _LCP>& sa, const std::string& str, const mxx::comm& c) {
    // gather SA back to root process
    std::vector<index_t> gsa = mxx::gatherv(sa.local_SA, 0, c);

    // check correctness using dss
    bool dss_correct = true;
    if (c.rank() == 0) {
        dss_correct = dss::check(str.begin(), str.end(), gsa);
    }
    return mxx::all_of(dss_correct);

    // check correctness using own method
    // TODO return bool value instead of exit(FAILURE) in that function
    //bool correct = gl_check_correct(sa, local_str.begin(), local_str.end(), c);
}

template <typename Iterator, typename index_t, bool _LCP>
bool check_sa_eqdss(suffix_array<Iterator, index_t, _LCP>& sa, const std::string& str, const mxx::comm& c) {
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
    suffix_array<std::string::iterator, uint32_t, false> sa(local_str.begin(), local_str.end(), c);

    // construct suffix array
    sa.construct();
    ASSERT_TRUE(check_sa_dss(sa, str, c));
    ASSERT_TRUE(check_sa_eqdss(sa, str, c));

    // construct with custom `k` to force early bucket chaising
    sa.construct(true, 3);
    ASSERT_TRUE(check_sa_dss(sa, str, c));
    ASSERT_TRUE(check_sa_eqdss(sa, str, c));

    // construct without bucket chaising
    sa.construct(false, 2);
    ASSERT_TRUE(check_sa_dss(sa, str, c));

    // construct with std::array based construction
    sa.construct_arr<2>(true);
    ASSERT_TRUE(check_sa_dss(sa, str, c));

    // construct with prefix-tripling
    sa.construct_arr<3>(true);
    ASSERT_TRUE(check_sa_dss(sa, str, c));

    // construct with prefix-quadrupling
    sa.construct_arr<3>(false);
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
    suffix_array<std::string::iterator, uint64_t, false> sa(local_str.begin(), local_str.end(), c);

    // construct suffix array
    sa.construct();
    ASSERT_TRUE(check_sa_dss(sa, str, c));
    ASSERT_TRUE(check_sa_eqdss(sa, str, c));

    // construct with custom `k` to force early bucket chaising
    sa.construct(true, 3);
    ASSERT_TRUE(check_sa_dss(sa, str, c));
    ASSERT_TRUE(check_sa_eqdss(sa, str, c));

    // construct without bucket chaising
    sa.construct(false, 2);
    ASSERT_TRUE(check_sa_dss(sa, str, c));

    // construct with std::array based construction
    sa.construct_arr<2>(true);
    ASSERT_TRUE(check_sa_dss(sa, str, c));

    // construct with prefix-tripling
    sa.construct_arr<3>(true);
    ASSERT_TRUE(check_sa_dss(sa, str, c));

    // construct with prefix-quadrupling
    sa.construct_arr<3>(false);
    ASSERT_TRUE(check_sa_dss(sa, str, c));
}
