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
 * @brief   Unit tests for sequential RMQ.
 */

#include <gtest/gtest.h>
#include <rmq.hpp>
#include <prettyprint.hpp>


// inherit from rmq in order to access the protected functions
template <typename Iterator, typename index_t = std::size_t>
class rmq_tester : public rmq<Iterator, index_t> {
public:
    // base class type
    typedef rmq<Iterator, index_t> base_class;

    // iterator value type
    typedef typename std::iterator_traits<Iterator>::value_type T;

    // constructor
    rmq_tester(Iterator begin, Iterator end) : base_class(begin, end) {}

    // check that the minimum per superblock is correct
    void check_superblock_correctness() {
        for (index_t d = 0; d < this->superblock_mins.size(); ++d) {
            // checking superblock correctness for block `d`
            index_t dist = 1<<d;
            ASSERT_EQ(this->n_superblocks - dist/2, this->superblock_mins[d].size()) << "Superblock minimum size is wrong for d=" << d;
            for (index_t i = 0; i < this->superblock_mins[d].size(); ++i) {
                Iterator minel_pos = std::min_element(this->_begin + i*base_class::SUPERBLOCK_SIZE, std::min(this->_begin + (i+dist)*base_class::SUPERBLOCK_SIZE, this->_end));
                ASSERT_EQ(*minel_pos, *(this->_begin + this->superblock_mins[d][i])) << "Superblock min is wrong for indeces: [d],[i]=" << d << "," << i;
            }
        }
    }

    // check that the minimum per block is correct
    void check_block_correctness() {
        for (index_t d = 0; d < this->block_mins.size(); ++d) {
            // checking block correctness for block `d`
            index_t dist = 1<<d;
            //assert(block_mins[d].size() == n_blocks - (n_superblocks)dist/2);
            for (index_t i = 0; i < this->block_mins[d].size(); ++i) {
                index_t n_sb = i / (base_class::NB_PER_SB - dist/2);
                index_t block_sb_idx = i % (base_class::NB_PER_SB - dist/2);
                index_t block_idx = base_class::NB_PER_SB*n_sb + block_sb_idx;
                index_t sb_end = base_class::SUPERBLOCK_SIZE*(n_sb+1);
                Iterator minel_pos = std::min_element(this->_begin + block_idx*base_class::BLOCK_SIZE, std::min(this->_begin + (block_idx+dist)*base_class::BLOCK_SIZE, std::min(this->_begin+sb_end,this->_end)));
                //index_t minel_idx = minel_pos - this->_begin;
                //index_t rmq_idx = base_class::SUPERBLOCK_SIZE*n_sb + this->block_mins[d][i];
                ASSERT_EQ(*minel_pos, *(this->_begin + base_class::SUPERBLOCK_SIZE*n_sb + this->block_mins[d][i]));
            }
        }
    }

    // in O(n^2)
    void check_all_subranges() {
        size_t n = std::distance(this->_begin, this->_end);

        for (size_t i = 0; i < n-1; ++i) {
            T min = *(this->_begin+i);
            for (size_t j = i+1; j < n; ++j) {
                if (*(this->_begin+j) < min)
                    min = *(this->_begin+j);
                ASSERT_EQ(min, *this->query(this->_begin+i, this->_begin+j+1)) << "wrong min for range (" << i << "," << j << ")";
            }
        }
    }
};

TEST(PsacRMQ, rmq1) {
    for (size_t size : {1, 13, 32, 64, 127, 233}) {
        std::vector<int> vec(size);
        std::generate(vec.begin(), vec.end(), [](){return std::rand() % 10;});
        // construct rmq
        rmq_tester<std::vector<int>::iterator> r(vec.begin(), vec.end());

        // check correctness
        r.check_block_correctness();
        r.check_superblock_correctness();
        r.check_all_subranges();
    }
}

TEST(PsacRMQ, rmq2) {
    for (size_t size : {123, 73, 88, 1025}) {
        std::vector<int> vec(size);
        std::generate(vec.begin(), vec.end(), [](){return 50 - std::rand() % 100;});
        // construct rmq
        rmq_tester<std::vector<int>::iterator> r(vec.begin(), vec.end());

        // check correctness
        r.check_block_correctness();
        r.check_superblock_correctness();
        r.check_all_subranges();
    }
}


TEST(PsacRMQ, rmqsmallblocks) {
    for (size_t size : {123, 73, 88, 1024, 1033}) {
        std::vector<unsigned int> vec(size);
        std::generate(vec.begin(), vec.end(), [](){return std::rand() % 100;});
        // construct rmq
        rmq_tester<std::vector<unsigned int>::iterator> r(vec.begin(), vec.end());

        // check correctness
        r.check_block_correctness();
        r.check_superblock_correctness();
        r.check_all_subranges();
    }
}

TEST(PsacRMQ, rmqbig) {
    std::vector<size_t> vec(1235);
    std::generate(vec.begin(), vec.end(), [](){return std::rand() % 1000;});
    // construct rmq
    rmq_tester<std::vector<size_t>::iterator> r(vec.begin(), vec.end());
    // check all queries
    r.check_all_subranges();
}

TEST(PsacRMQ, rmqmultimin) {

    std::vector<size_t> vec(1000);
    std::generate(vec.begin(), vec.end(), [](){return (8 + std::rand() % 10)/10;});
    rmq<std::vector<size_t>::const_iterator> minquery(vec.cbegin(), vec.cend());

    // check whether the min is the first min in the range
    // TODO: test for all partial ranges
    auto begin = vec.cbegin();
    auto min_it = minquery.query(vec.cbegin(), vec.cend());
    while (*min_it == 0) {
        if (min_it - begin > 0) {
            // assert the minimum of the range prior to the found min is larger
            auto min_it2 = minquery.query(begin, min_it);
            EXPECT_LT(*min_it, *min_it2) << " min for range [" << (begin-vec.cbegin()) << ",end] at pos " << (min_it - vec.cbegin()) << ", but there is a previous min of same value at pos " << (min_it2 - vec.cbegin());
        }
        // continue in remaining range:
        begin = min_it+1;
        if (begin == vec.cend())
            break;
        min_it = minquery.query(begin, vec.cend());
    }
    //std::cout << min_it - vec.cbegin() << std::endl;
}
