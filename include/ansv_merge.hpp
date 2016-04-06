/*
 * Copyright 2016 Georgia Institute of Technology
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

#ifndef ANSV_MERGE_HPP
#define ANSV_MERGE_HPP

#include <utility>
#include <iterator>
#include <vector>

#include "ansv_common.hpp"

template <int left_type, int right_type, typename Iterator, typename It2, typename It3>
std::pair<Iterator,Iterator> ansv_merge(Iterator left_begin, Iterator left_end, It2 left_ext_begin, It2 left_ext_end, Iterator right_begin, Iterator right_end, It3 right_ext_begin, It3 right_ext_end) {
    // starting with the largest value (right most in left sequence and leftmost in right sequence)
    typedef typename std::iterator_traits<Iterator>::value_type T;
    Iterator l = left_end;
    Iterator r = right_begin;
    while (l != left_begin && r != right_end) {
        Iterator l1 = l-1;
        if (l1->first < r->first) {
            // if furthest_eq: need to find furthest that is equal to l1
            if (left_type == furthest_eq) {
                // find the last of the equal range for l1
                Iterator li = l1;
                while (li != left_begin && (li-1)->first == l1->first)
                    --li;
                *r = *li;
            } else {
                *r = *l1;
            }
            ++r;
        } else if (r->first < l1->first) {
            // if furthest_eq: need to find furthest that is equal to r
            if (right_type == furthest_eq) {
                Iterator ri = r+1;
                while (ri != right_end && ri->first == r->first)
                    ++ri;
                *l1 = *(ri-1);
            } else {
                *l1 = *r;
            }
            --l;
        } else {
            // r == l
            // find end of equal range for both sides
            // equal range: [lsm, l1] = [lsm, l)
            Iterator lsm = l1;
            while (lsm != left_begin && l1->first == (lsm-1)->first)
                --lsm;

            // equal range: [r, rms)
            Iterator rsm = r+1;
            while (rsm != right_end && r->first == rsm->first)
                ++rsm;

            // saving first and last left elements since the underlying values
            // may be overwritten when merging into the left
            T low_left = *l1;
            T up_left = *lsm;

            Iterator lnext = lsm;
            Iterator rnext = rsm;
            bool end_loop = false;
            // assigning right matches to the left side
            if (right_type == nearest_sm) {
                if (rsm != right_end) {
                    for (Iterator li = l1; li != (lsm-1); --li) {
                        *li = *rsm;
                    }
                } else {
                    lnext = l;
                    end_loop = true;
                    // in read-only extension case: use that to find smaller!
                }
            } else if (right_type == nearest_eq) {
                // assign the next element to each internal and the first right to the last
                for (Iterator li = lsm; li != l1; ++li) {
                    *li = *(li+1);
                }
                *l1 = *r;
            } else if (right_type == furthest_eq) {
                // for all but the last of the equal range (which is on the
                // other side), the furthest eq is rms-1
                for (Iterator li = l1; li != (lsm-1); --li) {
                    *li = *(rsm-1);
                }
            }

            // assigning left matches to right side
            if (left_type == nearest_sm) {
                if (lsm != left_begin) {
                    for (Iterator ri = r; ri != rsm; ++ri) {
                        *ri = *(lsm-1);
                    }
                } else {
                    // didn't find a valid left-match for *r
                    rnext = r;
                    end_loop = true;
                }
            } else if (left_type == nearest_eq) {
                for (Iterator ri = rsm-1; ri != r; --ri) {
                    *ri = *(ri-1);
                }
                *r = low_left;
            } else if (left_type == furthest_eq) {
                for (Iterator ri = r; ri != rsm; ++ri) {
                    *ri = up_left;
                }
            }

            // continue with the next smaller element
            r = rnext;
            l = lnext;
            if (end_loop)
                break;
        }
    }


    // use right extension to merge remaining unsovled left elements
    if (l != left_begin && right_ext_begin != right_ext_end) {
        // extended sequence consists of a single equal range
        assert(right_ext_begin->first == (right_ext_end-1)->first);
        // find right matches for the remaining elements using the extended sequence
        if (right_type == nearest_sm) {
            for (Iterator li = l; li != left_begin; --li) {
                assert(right_ext_begin->first < (li-1)->first);
                *(li-1) = *right_ext_begin;
            }
            l = left_begin;
        } else if (right_type == nearest_eq) {
            for (Iterator li = l; li != left_begin; --li) {
                assert(right_ext_begin->first <= (li-1)->first);
                *(li-1) = *right_ext_begin;
            }
            l = left_begin;
        } else if (right_type == furthest_eq) {
            for (Iterator li = l; li != left_begin; --li) {
                assert(right_ext_begin->first <= (li-1)->first);
                *(li-1) = *(right_ext_end-1);
            }
            l = left_begin;
        }
    }

    // use left extension to merge remaining unsolved right elments
    if (r != right_end && left_ext_begin != left_ext_end) {
        // find left matches for unmatched right elements
        assert(left_ext_begin->first == (left_ext_end-1)->first);
        if (left_type == nearest_sm) {
            for (Iterator ri = r; ri != right_end; ++ri) {
                assert((left_ext_end-1)->first < ri->first);
                *ri = *(left_ext_end-1);
            }
            r = right_end;
        } else if (left_type == nearest_eq) {
            for (Iterator ri = r; ri != right_end; ++ri) {
                assert((left_ext_end-1)->first <= ri->first);
                *ri = *(left_ext_end-1);
            }
            r = right_end;
        } else if (left_type == furthest_eq) {
            // set to left most = left_ext_begin
            for (Iterator ri = r; ri != right_end; ++ri) {
                assert(left_ext_begin->first <= ri->first);
                *ri = *left_ext_begin;
            }
            r = right_end;
        }
    }

    return std::pair<Iterator, Iterator>(l-1, r);
}


// solve the right-matches for the left sequence
// right sequence is read-only
// left  sequence is solved with their right matches from the right sequence
template <int right_type, typename It1, typename It2>
It2 ansv_left_merge(It1 left_begin, It1 left_end, It2 right_begin, It2 right_end) {
    // right matches with right_type
    It1 l = left_end;
    It2 r = right_begin;
    while (l != left_begin && r != right_end) {
        It1 l1 = l-1;

        // find end of equal range for both sides
        // equal range: [lsm, l1] = [lsm, l)
        It1 lsm = l1;
        while (lsm != left_begin && l1->first == (lsm-1)->first)
            --lsm;

        if (right_type == nearest_sm) {
            // iterate through `r` to find the next smaller
            while (r != right_end && !(r->first < l1->first))
                ++r;
            if (r == right_end)
                break;
            for (It1 li = l1; li != (lsm-1); --li) {
                *li = *r;
            }
            l = lsm;
        } else if (right_type == nearest_eq) {
            // iterate through `r` to find the nearest non-larger
            while (r != right_end && r->first > l1->first)
                ++r;
            if (r == right_end)
                break;
            // assign nearest within equal range
            for (It1 li = lsm; li != l1; ++li) {
                *li = *(li+1);
            }
            *l1 = *r;
            l = lsm;
        } else if (right_type == furthest_eq) {
            // iterate through `r` to find the nearest non-larger
            while (r != right_end && r->first > l1->first)
                ++r;
            if (r == right_end)
                break;
            assert(r->first <= l1->first);
            // find end of equal range: [r, rms)
            It2 rsm = r+1;
            while (rsm != right_end && r->first == rsm->first)
                ++rsm;
            if (r->first == l1->first) {
                // assign this furthest element of its equal range to left
                for (It1 li = l1; li != (lsm-1); --li) {
                    *li = *(rsm-1);
                }
            } else {
                // only the right-most of the left equal range has r as its match
                for (It1 li = l1-1; li != (lsm-1); --li) {
                    *li = *l1;
                }
                *l1 = *(rsm-1);
            }
            l = lsm;
        }
    }
    // TODO: should I also return the right iterator?
    return l-1;
}


template <int left_type, typename It1, typename It2>
It2 ansv_right_merge(It1 left_begin, It1 left_end, It2 right_begin, It2 right_end) {
    // left matches with left_type
    It1 l = left_end;
    It2 r = right_begin;
    while (l != left_begin && r != right_end) {
        // find end of equal range: [r, rms)
        It2 rsm = r+1;
        while (rsm != right_end && r->first == rsm->first)
            ++rsm;

        if (left_type == nearest_sm) {
            // iterate through left to find the next smaller
            while (l != left_begin && !((l-1)->first < r->first))
                --l;
            if (l == left_begin) {
                break;
            }
            for (It2 ri = r; ri != rsm; ++ri) {
                *ri = *(l-1);
            }
            r = rsm;
        } else if (left_type == nearest_eq) {
            // iterate through left to find the nearest non-larger
            while (l != left_begin && r->first < (l-1)->first)
                --l;
            if (l == left_begin) {
                break;
            }
            // assign nearest within equal range
            for (It2 ri = rsm-1; ri != r; --ri) {
                *ri = *(ri-1);
            }
            *r = *(l-1);
            r = rsm;
        } else if (left_type == furthest_eq) {
            // iterate through left to find the nearest non-larger
            while (l != left_begin && r->first < (l-1)->first)
                --l;
            if (l == left_begin) {
                break;
            }
            // find end of equal range for both sides
            // equal range: [lsm, l1] = [lsm, l)
            It1 lsm = l-1;
            while (lsm != left_begin && (l-1)->first == (lsm-1)->first)
                --lsm;
            // assign this furthest element of its equal range to left
            if (r->first == (l-1)->first) {
                for (It2 ri = r; ri != rsm; ++ri) {
                    *ri = *lsm;
                }
            } else {
                for (It2 ri = r+1; ri != rsm; ++ri) {
                    *ri = *r;
                }
                *r = *lsm;
            }
            r = rsm;
        }
    }
    return r;
}

template <int left_type, int right_type, typename Iterator>
std::pair<Iterator,Iterator> ansv_merge(Iterator left_begin, Iterator left_end, Iterator right_begin, Iterator right_end) {
    return ansv_merge<left_type, right_type>(left_begin, left_end, left_begin, left_begin, right_begin, right_end, right_end, right_end);
}
#endif // ANSV_MERGE_HPP
