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

/**
 * @brief   Unit tests for bit operations.
 */

#include <gtest/gtest.h>

#include <bitops.hpp>
#include <cstdlib>

TEST(PsacBitops, TrailingZeros) {
    ASSERT_EQ(3u, trailing_zeros(0x8));
    ASSERT_EQ(6u, trailing_zeros(0xffffff40));
    ASSERT_EQ(9u, trailing_zeros(0xbeef0132032e00ull));
    //ASSERT_EQ(32u, trailing_zeros(0x0));
    //ASSERT_EQ(8*sizeof(size_t), trailing_zeros((size_t)0));
    //ASSERT_EQ(64u, trailing_zeros((uint64_t)0));
    ASSERT_EQ(40u, trailing_zeros(0xabcdef0000000000ull));
}

TEST(PsacBitops, ReferenceTrailingZeros) {
    int num_tests = 100000;
    for (int i = 0; i < num_tests; ++i)
    {
        uint64_t x = 0;
        x |= static_cast<uint64_t>(rand()) << 32;
        x |= static_cast<uint64_t>(rand());

        unsigned int t1 = trailing_zeros(x);
        unsigned int t2 = reference_trailing_zeros(x);
        ASSERT_EQ(t2, t1);
    }
}

TEST(PsacBitops, LeadingZeros) {
    ASSERT_EQ(20u, leading_zeros(0x00000efc));
    ASSERT_EQ(52u, leading_zeros(0x0000000000000efcull));
    ASSERT_EQ(0u, leading_zeros(0xbbae0000));
    ASSERT_EQ(1u, leading_zeros(0x501340f0));
    ASSERT_EQ(2u, leading_zeros(0x201340f0));
    ASSERT_EQ(3u, leading_zeros(0x1fc3defe));
    ASSERT_EQ(0u, leading_zeros(0xbeefbeefbeefadadull));
    ASSERT_EQ(32u, leading_zeros(0x0));
    ASSERT_EQ(8*sizeof(size_t), leading_zeros((size_t)0x0));
}

TEST(PsacBitops, ReferenceLeadingZeros) {
    int num_tests = 100000;
    for (int i = 0; i < num_tests; ++i)
    {
        uint64_t x = 0;
        x |= static_cast<uint64_t>(rand()) << 32;
        x |= static_cast<uint64_t>(rand());

        unsigned int t1 = leading_zeros(x);
        unsigned int t2 = reference_leading_zeros(x);
        ASSERT_EQ(t2, t1);
    }
}

TEST(PsacBitops, LCPbitwise) {
    ASSERT_EQ(7u,lcp_bitwise(0xfecf0123, 0xfecdccce, 16, 2));
    ASSERT_EQ(9u,lcp_bitwise(0x01234567, 0x01234566, 10, 3));
    ASSERT_EQ(0u,lcp_bitwise(0x00001234, 0x00002234, 4, 4));
    ASSERT_EQ(13u,lcp_bitwise(0xbeefbeefbeefadadull, 0xbeefbeefbeacadadull, 21, 3));
    ASSERT_EQ(21u,lcp_bitwise(0xbeefbeefbeefadadull, 0xbeefbeefbeefadadull, 21, 3));
    ASSERT_EQ(0u,lcp_bitwise(0x8eefbeefbeefadadull, 0xbeefbeefbeefadadull, 21, 3));
}

// ceillog and floorlog functions (similar to leading zeros!!)
TEST(PsacBitops, CeilFloorlog2) {
    ASSERT_EQ(9u, floorlog2(0x00000321u));
    ASSERT_EQ(10u, ceillog2(0x00000321u));
    ASSERT_EQ(10u, reference_ceillog2(0x00000321u));

    ASSERT_EQ(0u, floorlog2(0x1));
    ASSERT_EQ(0u, ceillog2(0x1));
    ASSERT_EQ(0u, reference_ceillog2(0x1));
    ASSERT_EQ(1u, floorlog2(0x2));
    ASSERT_EQ(1u, ceillog2(0x2));
    ASSERT_EQ(1u, reference_ceillog2(0x2));
    ASSERT_EQ(1u, floorlog2(0x3u));
    ASSERT_EQ(2u, ceillog2(0x3u));
    ASSERT_EQ(2u, reference_ceillog2(0x3u));

    ASSERT_EQ(0u, floorlog2(0x1ull));
    ASSERT_EQ(0u, ceillog2(0x1ull));
    ASSERT_EQ(1u, floorlog2(0x2ull));
    ASSERT_EQ(1u, ceillog2(0x2ull));
    ASSERT_EQ(1u, floorlog2(0x3ull));
    ASSERT_EQ(2u, ceillog2(0x3ull));

    ASSERT_EQ(38u, floorlog2(0x0000007fbbbbffffull));
    ASSERT_EQ(39u, ceillog2(0x0000007ffcccccffull));

    ASSERT_EQ(56u, floorlog2(0x0100000000000000ull));
    ASSERT_EQ(56u, ceillog2(0x0100000000000000ull));

    ASSERT_EQ(63u, floorlog2(0xffffffffffffffffull));
    ASSERT_EQ(64u, ceillog2(0xffffffffffffffffull));
}
