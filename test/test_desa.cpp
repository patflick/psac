
#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <utility>

#include <mxx/env.hpp>
#include <mxx/comm.hpp>
#include <mxx/file.hpp>
#include <mxx/distribution.hpp>

#include <suffix_array.hpp>
#include <lcp.hpp>
#include <divsufsort_wrapper.hpp>

#include "seq_query.hpp"
#include "desa.hpp"
#include "tldt.hpp"


using index_t = uint64_t;
using range_t = dist_desa<index_t>::range_t;


inline std::vector<std::pair<std::string, range_t>> mississippi_testcase() {
    using r = range_t;

    std::vector<std::pair<std::string, range_t>> patterns;
    patterns.emplace_back("i", r(0,4));
    patterns.emplace_back("mississip", r(4,5));
    patterns.emplace_back("ss", r(9,11));
    patterns.emplace_back("pp", r(6,7));

    patterns.emplace_back("m", r(4,5));
    patterns.emplace_back("pi", r(5,6));
    patterns.emplace_back("is", r(2,4));
    patterns.emplace_back("ssi", r(9,11));

    patterns.emplace_back("p", r(5,7));
    patterns.emplace_back("ississippi", r(3,4));
    patterns.emplace_back("si", r(7,9));
    patterns.emplace_back("miss", r(4,5));

    patterns.emplace_back("s", r(7,11));
    patterns.emplace_back("ip", r(1,2));
    patterns.emplace_back("ppi", r(6,7));
    patterns.emplace_back("mississippi", r(4,5));

    return patterns;
}


TEST(DesaTest,MississippiLocatePossible) {
    mxx::comm c;
    ASSERT_TRUE(c.size() <= 7) << "the `mississippi` test shouldn't run with more processors than character pairs";

    std::string s;
    if (c.rank() == 0) {
        s = "mississippi";
    }
    std::string input_str = mxx::stable_distribute(s, c);
    // construct DESA
    dist_desa<index_t> idx(c);
    idx.construct(input_str.begin(), input_str.end(), c);


    auto patterns = mississippi_testcase();

    // locate single pattern in distributed representation (collective op)
    for (auto pat : patterns) {
        std::string P = pat.first;
        range_t exp_res = pat.second;

        range_t r = idx.locate_possible(P);
        // (EXPECTED, ACTUAL)
        EXPECT_EQ(exp_res, r);
    }
}

TEST(DesaTest, FileIO) {
    mxx::comm c;
    ASSERT_TRUE(c.size() <= 7) << "the `mississippi` test shouldn't run with more processors than character pairs";

    std::string s;
    if (c.rank() == 0) {
        s = "mississippi";
        std::ofstream f("miss.str");
        f.write(&s[0],s.size());
    }
    std::string input_str = mxx::stable_distribute(s, c);
    // construct DESA
    dist_desa<index_t> desa(c);
    desa.construct(input_str.begin(), input_str.end(), c);

    desa.write("desa_miss", c);

    dist_desa<index_t> desa2(c);
    desa2.read("miss.str", "desa_miss", c);
    EXPECT_EQ(desa.sa.local_SA, desa2.sa.local_SA);
    EXPECT_EQ(desa.sa.local_LCP, desa2.sa.local_LCP);
    EXPECT_EQ(desa.sa.local_Lc, desa2.sa.local_Lc);
    EXPECT_EQ(desa.tli.idx.table, desa2.tli.idx.table);
    EXPECT_EQ(desa.tli.idx.alpha, desa2.tli.idx.alpha);

    // TODO: implement test for loading with smaller/larger nuber of processors
    // and still check that querying works correctly
    // -> introduce function to test queries for a given desa
}

#define PV(x, c) mxx::sync_cout(c) << "[" << c.rank() << "] " << "" # x  " = " << x << std::endl;

TEST(DesaTest,MississippiBulkLocatePossible) {
    mxx::comm c;

    ASSERT_TRUE(c.size() <= 7) << "the `mississippi` test shouldn't run with more processors than character pairs";

    std::string s;
    if (c.rank() == 0) {
        s = "mississippi";
    }
    std::string input_str = mxx::stable_distribute(s, c);
    // construct DESA
    dist_desa<index_t, tldt<index_t>> idx(c);
    idx.construct(input_str.begin(), input_str.end(), c);

    PV(input_str, c);
    PV(idx.sa.local_LCP, c);
    PV(idx.sa.local_SA, c);
    PV(idx.tli.prefix(), c);
    PV(idx.tli.idx.LCP, c);


    // create `strings` from patterns
    auto patterns = mississippi_testcase();
    mxx::blk_dist dist(patterns.size(), c.size(), c.rank());


    std::vector<std::string> my_patterns(dist.local_size());

    for (size_t i = 0; i < dist.local_size(); ++i) {
        my_patterns[i] = patterns[i+dist.eprefix_size()].first;
    }

    strings ss = strings::from_vec(my_patterns);
    std::vector<range_t> mysols = idx.bulk_locate(ss);

    // check results
    for (size_t i = 0; i < dist.local_size(); ++i) {
        EXPECT_EQ(patterns[i+dist.eprefix_size()].second, mysols[i]);
    }
}
