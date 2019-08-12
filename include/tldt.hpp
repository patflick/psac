/*
 * Copyright 2018 Georgia Institute of Technology
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

#ifndef TLDT_HPP
#define TLDT_HPP

#include <assert.h>

#include <vector>
#include <stack>
#include <algorithm>

#include <mxx/reduction.hpp>
#include <mxx/partition.hpp>

#include <seq_query.hpp>
#include <suffix_array.hpp>


template <typename index_t, typename lcp_t>
std::vector<index_t> sample_lcp(const std::vector<lcp_t>& LCP, index_t maxsize) {

    index_t n = LCP.size();

    struct node {
        index_t lcp; // == LCP[pos]
        index_t pos; // index of node
        index_t l;  // index of left parent
    };

    std::stack<node> st;
    st.push(node{0,0,0});

    index_t total_out = 1;
    std::vector<bool> do_output(n, false);
    do_output[0] = true;

    for (index_t i = 1; i < n; ++i) {

        while (!st.empty() && st.top().lcp > LCP[i]) {
            // new node is smaller LCP
            // -> "complete" all nodes which are on the stack
            node& u = st.top();

            // u.pos has range [u.l, .. , i)
            index_t parent_size = i - u.l;
            if (parent_size > maxsize) {
                // output but in inverse order !?
                do_output[u.pos] = true;
                ++total_out;
            }
            st.pop();
        }

        if (st.empty()) {
            // cant happen, because always: LCP[0] = 0
            assert(false);
        } else if (st.top().lcp == LCP[i]) {
            st.push(node{LCP[i], i, st.top().l});
            if (LCP[i] == 0) {
                do_output[i] = true;
                ++total_out;
            }
        } else {
            assert(st.top().lcp < LCP[i]);
            st.push(node{LCP[i], i, st.top().pos});
        }
    }
    // virtually, there is a 0 at the very end of the LCP array -> output them all
    while (!st.empty() && st.top().lcp > 0) {
        // new node is smaller LCP
        // -> "complete" all nodes which are on the stack
        node& u = st.top();

        // u.pos has range [u.l, .. , i)
        index_t parent_size = n - u.l;
        if (parent_size > maxsize) {
            do_output[u.pos] = true;
            ++total_out;
        }
        st.pop();
    }

    // TODO: output everything still in the queue?

    std::vector<index_t> idx(total_out);
    index_t j = 0;
    for (index_t i = 0; i < n; ++i) {
        if (do_output[i]) {
            idx[j] = i;
            ++j;
        }
    }
    return idx;
}


template <typename index_t, typename lcp_t>
std::vector<index_t> sample_lcp_indirect(const std::vector<lcp_t>& LCP, index_t maxsize) {
    assert(LCP[0] == 0);

    std::vector<index_t> samples; // all indices for output, eventually the result
    samples.push_back(0);
    size_t n = LCP.size();

    std::vector<index_t> st; // keep track of where each sequence starts? best to use indces into `samples`!?
    //st.push(0); // not necessary, 0 never gets "popped"

    //lcp_t minval = LCP[1];

    for (size_t i = 1; i < LCP.size(); ++i) {
        index_t l = LCP[i];
        index_t topl = LCP[samples.back()];

        while (!st.empty() && topl > l) {
            // l is rightmatch for topl and all its equals, their left match is in stack
            index_t lsidx = st.back(); // index of parent in `samples`
            index_t lnsv = samples[lsidx]; // index of parent in `LCP`
            st.pop_back();

            index_t parent_size = i - lnsv;
            if (parent_size > maxsize) {
                //  here i can output everything on the stack and quit the while loop
                // ie, everything in samples is safe, also `i` necessarily is safe and everything smaller to come after!
                // so i can push i, keep the stack empty. empty stack means: everything larger than samples.back() gets pushed to both, l == samples.back() -> nothing to stack, push samples. l < samples.back() -> output and safe, ie. not onto stack??
                // stack contains parent of those elements which may be removed at a later time
                //st = std::stack<index_t>();
                st.clear();
                break;
            } else {
                // elements are not output, but skipped
                // [x y z L L L]
                //      ^
                //      2       6, resize to 3
                samples.resize(lsidx + 1);
            }

            topl = LCP[samples.back()];
            // if i clear off the entire stack without outputting anything, what happens?
        }

        if (topl < l) { // only if new larger sequence do we push the stack
            st.push_back(samples.size()-1);
            samples.push_back(i);
        } else if (topl == l) {
            samples.push_back(i);
        } else {
            assert(st.empty());
            samples.push_back(i);
        }
    }

    // at the end we assume we get a 0 element. we go through the remaining stack
    // and remove everything that isn't large enough until the distance becomes big enough
    index_t topl = LCP[samples.back()];
    while (!st.empty() && topl > 0) {
        index_t lsidx = st.back(); // index of parent in `samples`
        index_t lnsv = samples[lsidx]; // index of parent in `LCP`
        st.pop_back();

        index_t parent_size = n - lnsv;
        if (parent_size > maxsize) {
            break;
        } else {
            samples.resize(lsidx + 1);
        }
    }

    return samples;
}

template <typename T>
std::vector<T> send_vec_left(const std::vector<T>& vec, const mxx::comm& c) {
    size_t send_size = vec.size();
    size_t recv_size = 0;
    mxx::datatype dt = mxx::get_datatype<T>();
    mxx::datatype size_dt = mxx::get_datatype<size_t>();
    int dst = c.rank() > 0 ? c.rank() - 1 : MPI_PROC_NULL;
    int src = c.rank() + 1 < c.size() ? c.rank() + 1 : MPI_PROC_NULL;
    MPI_Sendrecv(&send_size, 1, size_dt.type(), dst, 0,
                 &recv_size, 1, size_dt.type(), src, 0, c, MPI_STATUS_IGNORE);

    std::vector<T> res(recv_size);
    MPI_Sendrecv(const_cast<T*>(vec.data()), send_size, dt.type(), dst, 0,
                 res.data(), recv_size, dt.type(), src, 0, c, MPI_STATUS_IGNORE);

    return res;
}

// if this is not included as part of google test, define our own assert functions!
#ifndef GTEST_INCLUDE_GTEST_GTEST_H_
#define ASSERT_TRUE(x) {if (!(x)) { std::cerr << "[ERROR]: Assertion failed in " __FILE__ ":" << __LINE__ << std::endl;return ;}} std::cerr << ""
#define ASSERT_EQ(x, y) ASSERT_TRUE((x) == (y))
#define ASSERT_GT(x, y) ASSERT_TRUE((x) > (y))
#define ASSERT_LT(x, y) ASSERT_TRUE((x) < (y))
#endif

#define PV(x, c) mxx::sync_cout(c) << "[" << c.rank() << "] " << "" # x  " = " << x << std::endl;

template <typename index_t>
void seq_check_sample(const std::vector<index_t>& LCP, const std::vector<index_t>& off, index_t maxsize) {
    size_t n = LCP.size();
    // create sampling
    std::vector<size_t> lcp(off.size());
    for (size_t i = 0; i < off.size(); ++i) {
        ASSERT_TRUE(0 <= off[i] && off[i] < n);
        lcp[i] = LCP[off[i]];
    }
    // check correctness
    // (EXPECTED, ACTUAL)
    ASSERT_TRUE(off.size() > 0);
    ASSERT_EQ(0u, off[0]);

    for (size_t i = 1; i < off.size(); ++i) {
        // everything between off[i-1], .., off[i] should have LCP > than those two
        ASSERT_TRUE(off[i] - off[i-1] <= maxsize); // TODO: +-1 errors?
        size_t minlcp = std::max(LCP[off[i-1]], LCP[off[i]]);
        for (size_t j = off[i-1]+1; j < off[i]; ++j) {
            ASSERT_TRUE(0 <= j && j < n);
            ASSERT_TRUE(LCP[j] > minlcp); // << "LCP values skipped must be smaller than the larger of the two surrounding ones";
        }
    }

    // so far:
    // this tests the necessary conditions: intervals are valid, and not larger than the max

    // next:
    // need to also check they are the largest not exceeding the max,
    // ie. check that parent ranges of everything selected exceed the max

    std::stack<size_t> st;
    st.push(0);

    for (size_t i = 0; i < off.size(); ++i) {
        while (lcp[st.top()] > lcp[i]) {
            // skip all equal to find parent
            size_t l = lcp[st.top()];
            while (lcp[st.top()] == l) {
                st.pop();
            }

            // parent range should be larger than max size
            ASSERT_TRUE(off[i] - off[st.top()] > maxsize);
        }
        st.push(i);
    }


    while (lcp[st.top()] > 0) {
        // skip all equal to find parent
        size_t l = lcp[st.top()];
        while (lcp[st.top()] == l) {
            st.pop();
        }

        // parent range should be larger than max size
        ASSERT_TRUE(n - off[st.top()] > maxsize);
    }

}

// distributed version for sampling the LCP array
// -> used for TLDT construction
template <typename index_t, typename lcp_t>
std::vector<index_t> sample_lcp_distr(const std::vector<lcp_t>& local_LCP, index_t maxsize, const mxx::comm& comm) {

    // set up global sizes
    size_t n = mxx::allreduce(local_LCP.size(), comm);
    mxx::blk_dist dist(n, comm);
    size_t prefix = dist.eprefix_size();

    std::vector<index_t> samples; // all indices for output, eventually the result
    samples.push_back(prefix);

    std::vector<index_t> st; // stack of missing right matches, index into samples

    std::vector<index_t> left_st; // index into `samples`, stack of missing left matches
    left_st.push_back(0);

    // minimum seen so far
    lcp_t minval = local_LCP[0];

    for (size_t i = 1; i < local_LCP.size(); ++i) {
        index_t l = local_LCP[i];
        index_t topl = local_LCP[samples.back()-prefix];

        while (!st.empty() && topl > l) {
            // l is rightmatch for topl and all its equals, their left match is in stack
            index_t lsidx = st.back(); // index of parent in `samples`
            index_t lnsv = samples[lsidx]; // index of parent in `LCP`
            st.pop_back();

            index_t parent_size = i+prefix - lnsv;
            if (parent_size > maxsize) {
                //  here i can output everything on the stack and quit the while loop
                // ie, everything in samples is safe, also `i` necessarily is safe and everything smaller to come after!
                // so i can push i, keep the stack empty. empty stack means: everything larger than samples.back() gets pushed to both, l == samples.back() -> nothing to stack, push samples. l < samples.back() -> output and safe, ie. not onto stack??
                // stack contains parent of those elements which may be removed at a later time
                st.clear();
                break;
            } else {
                // elements are not output, but skipped
                samples.resize(lsidx + 1);
            }

            topl = local_LCP[samples.back()-prefix];
        }

        if (topl < l) { // only if new larger sequence do we push the stack
            st.push_back(samples.size()-1); // st always points toward right-most equal
            samples.push_back(i+prefix);
        } else if (topl == l) {
            samples.push_back(i+prefix);
        } else {
            assert(st.empty());
            samples.push_back(i+prefix);
            if (l < minval) {
                minval = l;
                if (samples[left_st.back()]-prefix <= maxsize) {
                    left_st.push_back(samples.size()-1);
                } else {
                    left_st.back() = samples.size() - 1;
                }
            }
        }
    }

    using pair_t = std::pair<lcp_t, index_t>;
    std::vector<pair_t> left(left_st.size());
    for (size_t i = 0; i < left_st.size(); ++i) {
        left[i].second = samples[left_st[i]];
        left[i].first = local_LCP[samples[left_st[i]]-prefix];
    }

    std::vector<pair_t> right = send_vec_left(left, comm);
    if (comm.rank() == comm.size()-1) {
        right.emplace_back(0, n);
    }

    // use `right` to solve remaining elements in stack `st`
    index_t first_out = right.size()-1;
    for (size_t i = 0; i < right.size(); ++i) {
        lcp_t l = right[i].first;
        index_t topl = local_LCP[samples.back()-prefix];
        while (!st.empty() && topl > l) { // right[i] is right match for samples.back()
            index_t lsidx = st.back(); // index of left parent in `samples`
            index_t lnsv = samples[lsidx]; // index of left parent in `LCP`
            st.pop_back();

            index_t parent_size = right[i].second - lnsv;
            if (parent_size > maxsize) {
                // here, I usually clear the stack, all remain!
                // thus right[i] as a parent is also a valid output!
                st.clear();
                break;
            } else {
                // remove from output
                samples.resize(lsidx + 1);
            }
            topl = local_LCP[samples.back()-prefix];
        }


        // what is right[i]'s parent range?
        //
        // 3 cases:
        if (topl < right[i].first) {
            // topl < right[i]  -> samples.back() is lnsv for right[i]
            if (i + 1 < right.size() && right[i+1].second - samples.back() > maxsize) {
                first_out = i;
                break;
            }
        } else if (topl == l) {
             // parent range is larger
            if (i + 1 < right.size() && right[i+1].second - samples[st.back()] > maxsize) {
                first_out = i;
                break;
             }
        } else if (st.empty()) {
            // empty and topl > right[i] => def output `i`
            first_out = i;
            break;
        }
    }

    // send back start point!
    index_t leftstart = mxx::right_shift(first_out, comm);

    // remove elements from front if they have a parent range smaller than maxsize
    if (comm.rank() > 0 && leftstart > 0) {
        std::vector<index_t> tmp;
        tmp.swap(samples);
        samples = std::vector<index_t>(tmp.begin() + left_st[leftstart], tmp.end());
    }

    return samples;
}

template <typename index_t>
struct tldt {
    desa_index<index_t> idx; // uses a sequential DESA underneath

    using range_t = std::pair<index_t,index_t>;

    std::vector<index_t> offsets;
    size_t n;

    template <typename char_t>
    void construct(const suffix_array<char_t, index_t, true, true>& sa, const std::string& local_str, const mxx::comm& comm) {
        // sample to 100
        n = sa.n;
        size_t prefix = sa.part.eprefix_size();
        index_t maxsize = sa.n / comm.size() / 128;
        if (maxsize < 2)
            maxsize = 2; // smallest parent interval
        std::vector<index_t> local_off = sample_lcp_distr(sa.local_LCP, maxsize, comm);

        // sample LCP and Lc at local_off
        std::vector<index_t> local_sample_lcp(local_off.size());
        std::vector<char_t> local_sample_lc(local_off.size());

        for (size_t i = 0; i < local_off.size(); ++i) {
            local_sample_lcp[i] = sa.local_LCP[local_off[i] - prefix];
            local_sample_lc[i] = sa.local_Lc[local_off[i] - prefix];
        }

        // allgather offsets and sampled lcp and Lc
        offsets = mxx::allgatherv(local_off, comm);
        idx.n = offsets.size();
        idx.LCP = mxx::allgatherv(local_sample_lcp, comm);
        idx.Lc = mxx::allgatherv(local_sample_lc, comm);

        // construct rmq
        idx.minq = rmq<typename std::vector<index_t>::const_iterator,index_t>(idx.LCP.begin(), idx.LCP.end());
    }

    inline index_t minmatch() const {
        return 1;
    }

    // for partitioning
    std::vector<index_t> prefix() const {
        // return inclusive prefix sum of bin size
        // (offsets is already the exclusive prefix sum, thus just need to shift by one)
        std::vector<index_t> ps(offsets.size());
        for (size_t i = 0; i < offsets.size()-1; ++i) {
            ps[i] = offsets[i+1];
        }
        ps.back() = n;
        return ps;
    }

    template <typename String>
    inline range_t lookup(const String& P) const {
        range_t r = idx.locate_possible(P);
        return range_t(offsets[r.first], r.second == offsets.size() ? n : offsets[r.second]);
    }
};

#endif // TLDT_HPP
