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
#ifndef PARTITION_HPP
#define PARTITION_HPP

#include <vector>

/**
 * @brief Simple recursive 1D-partitioning via bisection.
 *
 * Partitions bins given by an inclusive prefix sum of bin sizes.
 * Recursively partitions index range [l, r) of `tl` onto
 * processors (or partitons) [p0,p0+p).
 *
 * @param tl    Number of elements in each bin given as an incl. prefix sum
 *              (partial_sum) of bin sizes.
 * @param l     Left index (incl.) [l,r).
 * @param r     Right index (excl.) [l,r).
 * @param p0    First partition index.
 * @param p     Number of partitions
 * @param pstart    Returns the assigned bins in pstart[p0],...,pstart[p0+p-1]
 *                  where each `pstart[i]` is the index (number) of the first
 *                  bin assigned to partition `i`. I.e., each element is between
 *                  [0,tl.size()].
 *                  Partition `i` is assigned bins `pstart[i],..,pstart[i+1]`
 *                  or `pstart[i],...,tl.size()` for the last partition.
 */
template <typename index_t>
void rec_partition(const std::vector<index_t>& tl, size_t l, size_t r, int p0, int p, std::vector<size_t>& pstart) {
    if (p == 1) {
        return;
    }

    if (l == r) {
        for (int i = p0+1; i < p0+p; ++i) {
            pstart[i] = l;
        }
        return;
    }

    // p odd: (p+1)/2, (p-1)/2 weighted halfs (extreme case: 2,1)
    size_t excl = l == 0 ? 0 : tl[l-1];
    size_t num = tl[r-1] - excl;

    // compute how many elements for each side would be optimal
    size_t left_size;
    int left_p;
    if (p % 2 == 0) {
        left_size = num/2;
        left_p = p / 2;
    } else {
        // (p+1)/ 2 to the left, (p-1)/2 to the right
        left_size = (num*(p+1))/(2*p);
        left_p = (p+1)/2;
    }

    // find first index which is larger
    auto it = std::lower_bound(tl.begin()+l, tl.begin()+r, excl+left_size);
    size_t i = it - tl.begin();

    // decide whether this or the previous split is better (closer to `left size`)
    if (i > 0) {
        if (tl[i] - (left_size+excl) > (left_size+excl) - tl[i-1]) {
            --i;
        }
    }

    // left partition is now [l, i], convert to [l, i)
    ++i;

    // save the result
    pstart[p0+left_p] = i;

    // p even: find i in tl closest to (tl[l]+tl[r])/2
    rec_partition(tl, l, i, p0, left_p, pstart);
    rec_partition(tl, i, r, p0 + left_p, p - left_p, pstart);
}

/**
 * @brief Simple recursive 1D-partitioning via bisection.
 *
 * Partitions bins given by an inclusive prefix sum of bin sizes.
 *
 * @param tl    Number of elements in each bin given as an incl. prefix sum
 *              (partial_sum) of bin sizes.
 * @param p     Number of partitions
 * @return      Returns for each parititon, the assigned bins.
 *              Each `pstart[i]` is the index (number) of the first
 *              bin assigned to partition `i`.
 *              I.e., each element is between [0,tl.size()].
 *              Partition `i` is assigned bins `pstart[i],..,pstart[i+1]`
 *              or `pstart[i],...,tl.size()` for the last partition.
 */
template <typename index_t>
std::vector<size_t> partition(const std::vector<index_t>& tl, int p) {
    // partition by recursive weighted bisection
    std::vector<size_t> pstarts(p, 0);
    rec_partition(tl, 0, tl.size(), 0, p, pstarts);
    return pstarts;
}

#endif // PARTITION_HPP
