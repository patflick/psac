
#include <gtest/gtest.h>

#include <prettyprint.hpp>

#include <seq_query.hpp>
#include <partition.hpp> // TODO: move partition tests elsewhere


/* type parameterized index test */

template <typename IdxType>
class SeqIndexTest : public ::testing::Test {
     public:
    IdxType idx;
};

using index_t = uint64_t;
using MyTypes = ::testing::Types<sa_index<index_t>, esa_index<index_t>, bs_esa_index<index_t>, desa_index<index_t>, lookup_desa_index<index_t>>;
TYPED_TEST_CASE(SeqIndexTest, MyTypes);

// search for patterns inside `mississippi`
TYPED_TEST(SeqIndexTest, Locate_mississippi) {
    std::string input_str = "mississippi";
    using r = std::pair<index_t,index_t>;

    TypeParam idx;
    idx.construct(input_str.begin(), input_str.end());

    // TODO: flip arguments to: (EXPECTED, ACTUAL)
    EXPECT_EQ(idx.locate("i"), r(0,4));
    EXPECT_EQ(idx.locate("is"), r(2,4));
    EXPECT_EQ(idx.locate("ip"), r(1,2));
    EXPECT_EQ(idx.locate("ississippi"), r(3,4));

    EXPECT_EQ(idx.locate("m"), r(4,5));
    EXPECT_EQ(idx.locate("miss"), r(4,5));
    EXPECT_EQ(idx.locate("mississip"), r(4,5));
    EXPECT_EQ(idx.locate("mississippi"), r(4,5));

    EXPECT_EQ(idx.locate("p"), r(5,7));
    EXPECT_EQ(idx.locate("pi"), r(5,6));
    EXPECT_EQ(idx.locate("pp"), r(6,7));
    EXPECT_EQ(idx.locate("ppi"), r(6,7));

    EXPECT_EQ(idx.locate("s"), r(7,11));
    EXPECT_EQ(idx.locate("ss"), r(9,11));
    EXPECT_EQ(idx.locate("si"), r(7,9));
    EXPECT_EQ(idx.locate("ssi"), r(9,11));

    // expect not to find these:
    for (auto P : {"misx", "ississippii", "issississi", "mippi", "piss"}) {
        r res = idx.locate(P);
        EXPECT_EQ(res.first, res.second);

    }
}

// TODO: test case for testing random string and random patterns
//       check correcness by checking left and right next SA suffixes



TEST(Partition, part0) {
    lookup_desa_index<index_t> idx;
    std::string input_str = "mississippi";
    idx.construct(input_str.begin(), input_str.end());

    int p = 6;
    std::vector<size_t> part = partition(idx.tl.table, p);
    std::cout << "part: " << part << std::endl;

    // print size stats
    for (int i = 0; i < p; ++i) {
        size_t start = part[i];
        size_t end = i+1 < p ? part[i+1] : idx.tl.table.size()-1;
        size_t num = idx.tl.table[end] - idx.tl.table[start];
        std::cout << "[p " << i << "]: [" << start << "," << end << "):  " << num*100. / idx.tl.table.back() << " %  share" << std::endl;
    }
}



