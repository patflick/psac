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
 * @brief   Unit tests for ANSV
 */

#include <gtest/gtest.h>

#define MXX_DISABLE_TIMER 1

#include <cxx-prettyprint/prettyprint.hpp>
#include <ansv.hpp>
#include <rmq.hpp>
#include <mxx/distribution.hpp>
#include <vector>
#include <algorithm>
#include <limits>
#include <utility>

// check the correctness of ansv via a rmq
template <typename T, int type = nearest_sm>
void check_ansv(const std::vector<T>& in, const std::vector<size_t>& nsv, bool left, size_t nonsv) {
    // construct RMQ
    rmq<typename std::vector<T>::const_iterator> minquery(in.cbegin(), in.cend());

    std::vector<T> sm_nsv;
    if (type == furthest_eq) {
        sm_nsv = ansv_sequential(in, left);
    }

    // for each position check the nsv and direction
    for (size_t i = 0; i < in.size(); ++i) {
        if (nsv[i] == nonsv) {
            if (left && i > 0) {
                // expect this is an overall minimum of the range [0, i]
                //T m = *minquery.query(in.cbegin(), in.cbegin()+i+1);
                auto mit = minquery.query(in.cbegin(), in.cbegin()+i+1);
                T m = *mit;
                EXPECT_TRUE(in[i] == m || in[0] == m) << " at i=" << i << ", in[i]=" << in[i] << ",m=" << m << ", mit=" << mit - in.begin();
            } else if (!left && i+1 < in.size()) {
                T m = *minquery.query(in.cbegin()+i, in.cend());
                EXPECT_TRUE(in[i] == m) << " at i=" << i;
            }
        } else {
            size_t s = nsv[i];
            if (left) {
                EXPECT_LT(s, i);
                if (s+1 < i) {
                    T m = *minquery.query(in.cbegin()+s+1, in.cbegin()+i);
                    if (type == nearest_sm) {
                        // check that there is nothing smaller than in[i] in the range
                        EXPECT_TRUE(in[i] <= m && in[s] < m) << " for range [" << s+1 << "," << i << "): " << "in[i]=" << in[i] << ", in[s]=" << in[s] << ", m=" << m;
                    } else if (type == furthest_eq) {
                        // test that no smaller values lay in between
                        EXPECT_TRUE(in[s] <= in[i] && in[s] <= m);
                        if (in[s] < in[i]) {
                            EXPECT_TRUE(s <= sm_nsv[i]);
                            // check that between right most of in[s] equal elements,
                            // everything is larger than in[i]
                            if (sm_nsv[i] + 1 < i) {
                                T m2 = *minquery.query(in.cbegin()+sm_nsv[i]+1, in.cbegin()+i);
                                EXPECT_GT(m2, in[i]) << " i=" << i << ",s=" << s;
                            }
                            // no other equal to in[i] in between
                            //EXPECT_TRUE(m > in[i]) << " for range [" << s+1 << "," << i << "], m=" << m << ", in[i]=" << in[i] << ", in[s]=" << in[s];
                        } else if (in[s] == in[i]) {
                        }

                        // in[s] is the furthest of its kind
                        // we check that the nsv for s is smaller, not equal
                        if (nsv[s] != nonsv) {
                            EXPECT_LT(in[nsv[s]], in[s]) << "i=" << i << ", s=" << s << ", nsv[s]=" << nsv[s];
                        }
                    } else { // type == nearest_eq
                        EXPECT_TRUE(in[i] < m && in[s] < m) << "i=" << i << ",s=" << s;
                    }
                }
                // element at `s` is smaller than `in[i]`
                if (type == nearest_sm) {
                    EXPECT_LT(in[s], in[i]);
                } else {
                    EXPECT_LE(in[s], in[i]);
                }

            } else {
                EXPECT_GT(s, i);
                // no other element can be smaller than `in[i]` before `in[s]`
                if (i < s-1) {
                    T m = *minquery.query(in.cbegin()+i+1, in.cbegin()+s);
                    if (type == nearest_sm) {
                        EXPECT_TRUE(in[i] <= m && in[s] < m) << " for range [" << i << "," << s-1 << "]";
                    } else if (type == furthest_eq) {
                        EXPECT_TRUE(in[s] <= in[i] && in[s] <= m);
                        // test if the nsv is actually the furthest
                        if (in[s] < in[i]) {
                            EXPECT_TRUE(sm_nsv[i] <= s);
                            if (i + 1 < sm_nsv[i]) {
                                T m2 = *minquery.query(in.cbegin()+i+1, in.cbegin()+sm_nsv[i]);
                                EXPECT_GT(m2, in[i]);
                            }
                        } else if (in[i] == in[s]) { }
                        // in[s] is the furthest of its kind
                        // we check that the nsv for s is smaller, not equal
                        if (nsv[s] != nonsv) {
                            EXPECT_LT(in[nsv[s]], in[i]) << " for i=" << i << ", in[i]=" << in[i] << ", in[s]=" << in[s] << ",s=" << s << ", nsv[s]=" << nsv[s] << ", in[nsv[s]]=" << in[nsv[s]] <<  ", in.size()=" << in.size();
                        }
                    } else { // type == nearest_eq
                        EXPECT_TRUE(in[i] < m && in[s] < m) << " for range [" << i << "," << s-1 << "]";
                    }
                }
                // element at `s` is smaller than in[i]
                if (type == nearest_sm) {
                    EXPECT_LT(in[s], in[i]);
                } else {
                    EXPECT_LE(in[s], in[i]);
                }
            }

        }
    }
}


// the API declaration of a generalized ANSV function
template <typename T>
using ansv_func_t=void (*)(const std::vector<T>&, std::vector<size_t>&, std::vector<size_t>&, std::vector<std::pair<T, size_t>>&, const mxx::comm& c, size_t);

// parallel test for neareat_smaller ANSV functions
template <typename T>
void par_test_ansv(const std::vector<T>& in, ansv_func_t<T> ansv_func, const mxx::comm& c) {
    // stably distribute input across processors
    std::vector<size_t> vec = mxx::stable_distribute(in, c);

    // prepare parameters for ANVS
    std::vector<size_t> left_nsv;
    std::vector<size_t> right_nsv;
    std::vector<std::pair<T, size_t>> lr_mins;
    const size_t nonsv = std::numeric_limits<size_t>::max();

    // call actual ANSV function
    ansv_func(vec, left_nsv, right_nsv, lr_mins, c, nonsv);

    // check that the output sizes are correct
    ASSERT_TRUE(mxx::all_of(vec.size() == left_nsv.size() && vec.size() == right_nsv.size(), c));

    // gather results
    left_nsv = mxx::gatherv(left_nsv, 0, c);
    right_nsv = mxx::gatherv(right_nsv, 0, c);

    // sequentially check the correctness on processor 0
    if (c.rank() == 0) {
        check_ansv<T,nearest_sm>(in, left_nsv, true, nonsv);
        check_ansv<T,nearest_sm>(in, right_nsv, false, nonsv);
    }
}

// parallel test for generalized ANSV
template <typename T, int left_type, int right_type, int indexing_type>
void par_test_gansv(const std::vector<T>& in, ansv_func_t<T> ansv_func, const mxx::comm& c) {
    // stably distribute input across processors
    std::vector<size_t> vec = mxx::stable_distribute(in, c);
    size_t prefix = mxx::exscan(vec.size(), c);

    // prepare parameters for ANVS
    std::vector<size_t> left_nsv;
    std::vector<size_t> right_nsv;
    std::vector<std::pair<T, size_t>> lr_mins;
    const size_t nonsv = std::numeric_limits<size_t>::max();

    // call actual ANSV function
    ansv_func(vec, left_nsv, right_nsv, lr_mins, c, nonsv);

    // check that the output sizes are correct
    ASSERT_TRUE(mxx::all_of(vec.size() == left_nsv.size() && vec.size() == right_nsv.size(), c));

    // resolve local indexing into global indexing prior to gathering the results
    if (indexing_type == local_indexing) {
        for (size_t i = 0; i < vec.size(); ++i) {
            // resolve left
            if (left_nsv[i] != nonsv) {
                if (left_nsv[i] < vec.size()) {
                    // this is a local element, add prefix
                    left_nsv[i] += prefix;
                } else {
                    EXPECT_TRUE(left_nsv[i] >= vec.size() && left_nsv[i] < vec.size()+lr_mins.size());
                    left_nsv[i] = lr_mins[left_nsv[i]-vec.size()].second;
                }
            }
            // resolve right
            if (right_nsv[i] != nonsv) {
                if (right_nsv[i] < vec.size()) {
                    // this is a local element, add prefix
                    right_nsv[i] += prefix;
                } else {
                    EXPECT_TRUE(right_nsv[i] >= vec.size() && right_nsv[i] < vec.size()+lr_mins.size());
                    right_nsv[i] = lr_mins[right_nsv[i]-vec.size()].second;
                }
            }
        }
    }

    // gather results
    left_nsv = mxx::gatherv(left_nsv, 0, c);
    right_nsv = mxx::gatherv(right_nsv, 0, c);

    // sequentially check the correctness on processor 0
    if (c.rank() == 0) {
        check_ansv<T,left_type>(in, left_nsv, true, nonsv);
        check_ansv<T,right_type>(in, right_nsv, false, nonsv);
    }
}


/*********************************************************************
 * Macros for generating all combination tests for generalized ANSV  *
 *********************************************************************/

#define PAR_TEST_GANSV(T, in, c, func_name, idx_type, left_type, right_type) \
        {std::string str = "" #func_name "<" #left_type ", " #right_type ", " #idx_type ">"; \
        SCOPED_TRACE(str); \
        par_test_gansv<T, left_type, right_type, idx_type>(in, &func_name<T,left_type,right_type,idx_type>, c); }

#define PAR_TEST_GANSV_ALLRIGHT(T, in , c, func_name, idx_type, left_type) \
        PAR_TEST_GANSV(T, in, c, func_name, idx_type, left_type, nearest_sm) \
        PAR_TEST_GANSV(T, in, c, func_name, idx_type, left_type, nearest_eq) \
        PAR_TEST_GANSV(T, in, c, func_name, idx_type, left_type, furthest_eq)

#define PAR_TEST_GANSV_ALLVAR(T, in, c, func_name, idx_type) \
        PAR_TEST_GANSV_ALLRIGHT(T, in , c, func_name, idx_type, nearest_sm) \
        PAR_TEST_GANSV_ALLRIGHT(T, in , c, func_name, idx_type, nearest_eq) \
        PAR_TEST_GANSV_ALLRIGHT(T, in , c, func_name, idx_type, furthest_eq)

// generates tests for all combinations of gANSV template parameters:
//    {nearest_sm, nearest_eq, furthest_eq}^2 x {local_indexing, global_indexing}
//    => 3*3*2 total
#define PAR_TEST_GANSV_ALL(T, in, c, func_name) \
        PAR_TEST_GANSV_ALLVAR(T, in, c, func_name, global_indexing) \
        PAR_TEST_GANSV_ALLVAR(T, in, c, func_name, local_indexing)


TEST(PsacANSV, SeqANSVrand) {

    for (size_t n : {8, 137, 1000, 4200, 13790}) {
        std::vector<size_t> vec(n);
        std::srand(0);
        std::generate(vec.begin(), vec.end(), [](){return std::rand() % 997;});
        // calc ansv
        std::vector<size_t> left_nsv = ansv_sequential(vec, true);
        std::vector<size_t> right_nsv = ansv_sequential(vec, false);

        check_ansv(vec, left_nsv, true, 0);
        check_ansv(vec, right_nsv, false, 0);
    }
}

#define PAR_GTEST_GANSV_RAND(func_name) \
TEST(PsacANSV, ParallelANSVrand_ ## func_name) {  \
    mxx::comm c; \
    for (size_t n : {13, 137, 1000, 26666}) { \
        std::vector<size_t> in; \
        if (c.rank() == 0) { \
            in.resize(n); \
            std::srand(7); \
            std::generate(in.begin(), in.end(), [](){return std::rand() % 100;}); \
        } \
        PAR_TEST_GANSV_ALL(size_t, in, c, func_name); \
    } \
}

// TODO: use google test test-case intialization instead of
//       generating input every time
#define PAR_GTEST_ANSV_RAND(func_name) \
TEST(PsacANSV, ParallelANSVrand_ ## func_name) {  \
    mxx::comm c; \
    for (size_t n : {13, 137, 1000, 26666}) { \
        std::vector<size_t> in; \
        if (c.rank() == 0) { \
            in.resize(n); \
            std::srand(7); \
            std::generate(in.begin(), in.end(), [](){return std::rand() % 100;}); \
        } \
        par_test_ansv(in, & func_name <size_t>, c); \
    } \
}


#define PAR_GTEST_ANSV_RAND_PERM(func_name) \
TEST(PsacANSV, ParallelANSVrandperm_ ## func_name) {  \
    mxx::comm c; \
    for (size_t n : {13, 137, 1000, 26666}) { \
        std::vector<size_t> in; \
        if (c.rank() == 0) { \
            in.resize(n); \
            std::srand(7); \
            for (size_t i = 0; i < n; ++i) in[i] = i; \
            std::random_shuffle(in.begin(), in.end()); \
        } \
        par_test_ansv(in, & func_name <size_t>, c); \
    } \
}

PAR_GTEST_GANSV_RAND(ansv);
PAR_GTEST_GANSV_RAND(gansv_impl);

PAR_GTEST_ANSV_RAND(my_ansv);
// NOTE: these two variants fail if there are equal elements in the range
//       -> they only work if all elements are unique
//PAR_GTEST_ANSV_RAND(my_ansv_minpair);
//PAR_GTEST_ANSV_RAND(hh_ansv);

PAR_GTEST_ANSV_RAND_PERM(my_ansv_minpair);
PAR_GTEST_ANSV_RAND_PERM(hh_ansv);

TEST(PsacANSV, ParallelANSVrand_special) {
    mxx::comm c;
    for (size_t n : {137}) { // {13, 137, 1000, 26666}) {
        std::vector<size_t> in;
        if (c.rank() == 0) {
            in.resize(n);
            std::srand(7);
            std::generate(in.begin(), in.end(), [](){return std::rand() % 100;});
        }
        PAR_TEST_GANSV(size_t, in, c, gansv_impl, global_indexing, nearest_eq, nearest_sm);
        //par_test_gansv<size_t, nearest_sm, nearest_eq, global_indexing>(in, &my_ansv_minpair_lbub<size_t,nearest_sm,nearest_eq,global_indexing>, c);
    }
}
