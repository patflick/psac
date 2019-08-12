/*
 * Copyright 2019 Georgia Institute of Technology
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

/**
 * @brief   Unit tests for sampling the LCP
 */

#include <gtest/gtest.h>

#include <stack>

#include <mxx/comm.hpp>
#include <mxx/distribution.hpp>

#include <tldt.hpp> // TODO: renmae/move code to different file?
#include <alphabet.hpp> // for rand_dna
#include <seq_query.hpp> // for sequential SA/LCP construction
#include <cxx-prettyprint/prettyprint.hpp>



TEST(PsacSampleLCP, SeqSmall) {
    // sequentially get SA and LCP for random or easy input!?
    size_t n = 10000;
    size_t maxsize = 78;
    std::string s = rand_dna(n, 13);


    salcp_index<size_t> idx;
    idx.construct(s.begin(), s.end());


    std::vector<size_t> off = sample_lcp_indirect(idx.LCP, maxsize);


    // check correctness
    // (EXPECTED, ACTUAL)
    seq_check_sample(idx.LCP, off, maxsize);
}

TEST(PsacSampleLCP, ParSmall) {
    // sequentially get SA and LCP for random or easy input!?
    size_t n = 100000;
    size_t maxsize = 123;

    mxx::comm c;

    salcp_index<size_t> idx;
    if (c.rank() == 0) {
        std::string s = rand_dna(n, 13);
        idx.construct(s.begin(), s.end());
    }

    std::vector<size_t> local_LCP = mxx::stable_distribute(idx.LCP, c);

    std::vector<size_t> local_off = sample_lcp_distr(local_LCP, maxsize, c);

    std::vector<size_t> off = mxx::gatherv(local_off, 0, c);

    if (c.rank() == 0) {
        seq_check_sample(idx.LCP, off, maxsize);
    }
}
