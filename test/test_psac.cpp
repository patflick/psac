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


TEST(PSAC, Rand1) {
    mxx::comm c;

    size_t size = 1337;

    std::string str;
    if (c.rank() == 0) {
        // generate some random input string
        str = rand_dna(size, 7);
    }
    // distribute string equally
    std::string local_str = mxx::stable_distribute(str, c);

    // create suffix array w/o LCP
    suffix_array<std::string::iterator, size_t, false> sa(local_str.begin(), local_str.end(), c);
    // construct suffix array
    sa.construct();

    // gather SA back to root process
    std::vector<size_t> gsa = mxx::gatherv(sa.local_SA, 0, c);

    // check correctness using dss
    bool dss_correct = true;
    if (c.rank() == 0) {
        dss::check(str.begin(), str.end(), gsa);
    }

    ASSERT_TRUE(dss_correct);

    // check correctness using own method
    // TODO return bool value instead of exit(FAILURE) in that function
    //bool correct = gl_check_correct(sa, local_str.begin(), local_str.end(), c);
}
