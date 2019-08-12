
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <string.h>

// psac stuff:
//#include <suffix_array.hpp>
#include <lcp.hpp>
#include <rmq.hpp>
#include <divsufsort_wrapper.hpp>
#include <lookup_table.hpp>

#include <prettyprint.hpp>

#define RMQ_USE_SDSL 0

#if RMQ_USE_SDSL
#include <sdsl/rmq_succinct_sada.hpp>
#endif

#ifndef SEQ_QUERY_HPP
#define SEQ_QUERY_HPP

namespace my {
struct timer {
    using clock_type = std::chrono::steady_clock;
    using time_point = clock_type::time_point;
    using duration = time_point::duration;

    time_point ts;
    duration elapsed;
    duration total;

    timer() : ts(), elapsed(0), total(0) {}

    inline void tic() {
      ts = clock_type::now();
    }

    inline void toc() {
      elapsed = clock_type::now() - ts;
      total += elapsed;
    }

    template <typename precision>
    inline typename precision::rep get_time() const {
      return std::chrono::duration_cast<precision>(elapsed).count();
    }

    template <typename precision>
    inline typename precision::rep get_total_time() const {
      return std::chrono::duration_cast<precision>(total).count();
    }

    inline std::chrono::nanoseconds::rep get_ns() const {
      return get_time<std::chrono::nanoseconds>();
    }

    inline std::chrono::nanoseconds::rep get_total_ns() const {
      return get_total_time<std::chrono::nanoseconds>();
    }
};
} // namepsace my

// LCP from SA and string??

// libdivsufsort wrapper
//#include <divsufsort_wrapper.hpp>

/// TODO: compare SA, SA+LCP, ESA, and DESA querying
///       (opt) compare against sdsl (although ours requires tons more mem...)


// TODO: speedup this comparison function using AVX etc
// compares pattern P lexicographically to S[pos...) and returns:
//  <0: P is smaller
//  =0: P is equal
//  >0: P is larger
inline int cmp_pattern(const std::string& P, const std::string& S, size_t pos) {
    if (pos >= S.size()) {
        return 1;
    }
    return strncmp(&P[0], &S[pos], P.size());
}

template <typename Iterator>
inline int cmp_pattern(const std::string& P, Iterator strbeg, size_t n, size_t pos) {
    if (pos >= n) {
        return 1;
    }
    return strncmp(&P[0], &strbeg[pos], P.size());
}


// simple binary search for given condition
// assuming for the sequence `data`, all cond(data[i]) == false come before all
// cond(data[i]) == true. This returns the smallest index i such that
// cond(data[i]] == true, and returns data.size() if none exist
template <typename T, typename Cond>
size_t binary_search(const std::vector<T>& data, Cond cond) {
    if (data.empty()) {
        return 0;
    }
    if (cond(data.front())) {
        return 0;
    }
    if (!cond(data.back())) {
        return data.size();
    }

    size_t l = 0; 
    size_t r = data.size();

    while (l + 1 < r) {
        size_t m = (l+r) >> 1;
        if (!cond(data[m])) {
            l = m;
        } else {
            r = m;
        }
    }
    return r;
}


// O(P log(n)) binary search for pattern using only string and SA
template <typename index_t>
std::pair<index_t, index_t> locate_sa(const std::string& S, const std::vector<index_t>& SA, const std::string& P) {
    // binary search using SA and S
    assert(S.size() == SA.size());

    // TODO: can be speed up by first searching until we hit a midpoint for which cmp == 0, then do sepearate lb and ub
    index_t lb = binary_search(SA, [&](index_t pos) { return cmp_pattern(P, S, pos) <= 0; });
    index_t ub = binary_search(SA, [&](index_t pos) { return cmp_pattern(P, S, pos) < 0; });
    return std::pair<index_t,index_t>(lb,ub);
}

// locate using LCP information (M&M use interval tree, we use RMQ !?)
// O(P + log(n))


// returns the offset for the first differening character between x and y
// this compares until at most n characters in. if x[0..n) == y[0..n), then
// this returns `n`
int my_stricmp(const char* x, const char* y, size_t n, size_t& len) {
    len = 0;
    while (len < n && x[len] == y[len]) {
        ++len;
    }
    if (len == n)
        return 0;
    else if (x[len] < y[len])
        return -1;
    else
        return 1;
}

// computes the lcp between the suffix S[pos+offset...) and the pattern P[offset...)
inline size_t lcp_offset(const std::string& P, const std::string& S, size_t pos, size_t offset) {
    assert(pos+offset <= S.size());
    size_t max = std::min(S.size() - pos, P.size());
    size_t i = offset;
    while (i < max && P[i] == S[pos+i]) {
        ++i;
    }
    return i;
}

inline size_t lcp_offset(const std::string& P, const char* strbegin, size_t n, size_t pos, size_t offset) {
    assert(pos+offset <= n);
    size_t max = std::min(n - pos, P.size());
    size_t i = offset;
    while (i < max && P[i] == strbegin[pos+i]) {
        ++i;
    }
    return i;
}

template <typename index_t, typename RMQ>
size_t lb_rmq(const std::vector<index_t>& LCP, const RMQ& minq, size_t l, size_t r, index_t q) {
    // we have a `q` match at `r` and trying to find the lower bound
    // LCP[r] = lcp(SA[r-1],SA[r])
    if (LCP[r] < q)
        return r; // `r` is the lower bound

    //typename std::vector<index_t>::const_iterator b = LCP.begin();

    while (l + 1 < r) {
        size_t mid = (l+r)/2;
        index_t x = LCP[minq(mid+1, r)];
        if (x < q) {
            l = mid;
        } else {
            r = mid;
        }
    }

    return r;
}

template <typename index_t, typename RMQ>
size_t ub_rmq(const std::vector<index_t>& LCP, const RMQ& minq, size_t l, size_t r, index_t q) {
    // we have a `q` match at `l` and trying to find the upper bound
    // LCP[l+1] = lcp(SA[l],SA[l+1])
    if (LCP[l+1] < q)
        return l; // `l` is the upper bound


    //typename std::vector<index_t>::const_iterator b = LCP.begin();

    // one-sided binary search
    while (l + 1 < r) {
        size_t mid = (l+r)/2;
        index_t x = LCP[minq(l+1, mid)];
        if (x < q) {
            r = mid;
        } else {
            l = mid;
        }
    }

    return l;
}

template <typename index_t>
struct sa_index {

    size_t n;
    const char* strbegin;
    const char* strend;
    std::vector<index_t> SA;

    my::timer rmqtimer;

    template <typename Iterator>
    void construct(Iterator begin, Iterator end) {
        strbegin = &(*begin);
        strend = &(*end);
        dss::construct(begin, end, SA);
        n = SA.size();
    }

    std::pair<index_t, index_t> locate(const std::string& P) {
        // TODO: can be speed up by first searching until we hit a midpoint for which cmp == 0, then do sepearate lb and ub
        index_t lb = binary_search(SA, [&](index_t pos) { return cmp_pattern(P, strbegin, n, pos) <= 0; });
        index_t ub = binary_search(SA, [&](index_t pos) { return cmp_pattern(P, strbegin, n, pos) < 0; });
        return std::pair<index_t,index_t>(lb,ub);
    }
};

template <typename index_t>
struct salcp_index : public sa_index<index_t> {

    std::vector<index_t> LCP;

    template <typename Iterator>
    void construct(Iterator begin, Iterator end) {
        sa_index<index_t>::construct(begin, end);
        // construct LCP from SA:
        // create ISA needed for constructing LCP
        std::vector<index_t> ISA(this->n);
        for (size_t i = 0; i < this->n; ++i) {
            ISA[this->SA[i]] = i;
        }
        std::vector<index_t> LCP;
        lcp_from_sa(std::string(this->strbegin, this->strend), this->SA, ISA, this->LCP);
    }
};


// O(sigma*P) using repeated RMQ on LCP [Fischer & Heun 2007]
template <typename index_t>
struct esa_index : public salcp_index<index_t> {
    using it_t = typename std::vector<index_t>::const_iterator;

#if RMQ_USE_SDSL
    sdsl::rmq_succinct_sct<> minq;
#else
    rmq<it_t, index_t> minq;
#endif

    template <typename Iterator>
    void construct(Iterator begin, Iterator end) {
        salcp_index<index_t>::construct(begin, end);
        // construct RMQ ontop of LCP
#if RMQ_USE_SDSL
        minq = sdsl::rmq_succinct_sct<>(&this->LCP);
#else
        minq = rmq<it_t,index_t>(this->LCP.begin(), this->LCP.end());
#endif
    }

    std::pair<index_t,index_t> locate(const std::string& P) const {

        size_t n = this->n;
        size_t m = P.size();
        size_t l = 0;
        size_t r = n-1;

        size_t q = 0; // size of current match

        while (q < m && l < r) {

            // NOTE: LCP[i] = lcp(SA[i-1],SA[i]), LCP[0] = 0
            // using [l,r] as an inclusive SA range
            // corresponding to LCP query range [l+1,r]


            // l, r <- getChild(q, c, l, r):

            // get first child interval and depth
            //it_t ii = minq.query(b+l+1, b+r+1); // this may have already been run by the previous step
            //size_t i = ii - b;
            size_t i = minq(l+1, r);
            index_t lcpv = this->LCP[i];
            assert(lcpv >= q);

            // check skipped characters (XXX optional?)
            // characters in P[q..lcpv), compare to [l]
            bool match = true;
            for (size_t j = q; q < lcpv; ++q) {
                match &= (P[j] == this->strbegin[this->SA[l]+j]);
            }
            if (!match) {
                return std::pair<index_t, index_t>(l,l);
            }

            // check if we've reached the end of the pattern
            if (lcpv >= m) {
                return std::pair<index_t,index_t>(l, r+1);
            }

            char c = P[lcpv];
            do {
                // `i` is the lcp(SA[i-1],SA[i])
                char Lc = this->strbegin[this->SA[i-1]+lcpv]; // == S[SA[l]+lcpv] for first iter
                //char Rc = S[SA[i]+lcpv];
                if (Lc == c) {
                    r = i-1;
                    break;
                }
                l = i;
                if (l == r)
                    break;
                i = minq(l+1, r);
            } while (l < r && this->LCP[i] == lcpv);

            if (this->strbegin[this->SA[l]+lcpv] == c) {
                // found the interval we were looking for
                q = lcpv+1;
            } else {
                return std::pair<index_t,index_t>(l,l);
            }

        }
        return std::pair<index_t,index_t>(l, r+1);
    }
};

// O(P + log(n)) via binary search range queries on LCP
// original M&M does this via pre-computing the RLCP and LLCP queries for
// all possible (l,mid,r) points of the binary search
//
// this version uses binary search on an underlying RMQ
template <typename index_t>
struct bs_esa_index : public esa_index<index_t> {
    using it_t = typename std::vector<index_t>::const_iterator;

    std::pair<index_t, index_t> locate(const std::string& P) const {
        size_t n = this->n;
        size_t m = P.size();
        size_t l = 0;
        size_t r = n-1;
        size_t llcp = lcp_offset(P, this->strbegin, n, this->SA[0], 0);
        size_t rlcp = lcp_offset(P, this->strbegin, n, this->SA[n-1], 0);

        index_t q = 0; // num of chars matched

        if (llcp < m && P[llcp] < this->strbegin[this->SA[0]+llcp]) {
            // first suffix is larger than `P` -> no match
            return std::pair<index_t,index_t>(0,0);
        } else if (llcp == m) {
            // first suffixes matches, just find ub
            r = ub_rmq(this->LCP, this->minq, 0, n-1, m);
            return std::pair<index_t,index_t>(0,r+1);
        }

        if (rlcp < m && P[rlcp] > this->strbegin[this->SA[n-1]+rlcp]) {
            // last suffix is smaller than `P` -> no match
            return std::pair<index_t,index_t>(n,n);
        } else if (rlcp == m) {
            // last suffix matches `P` -> find lb
            l = lb_rmq(this->LCP, this->minq, 0, n-1, m);
            return std::pair<index_t, index_t>(l, n);
        }


        while (l+1 < r) {

            size_t mid = (l+r) / 2;
            index_t pos = this->SA[mid];

            if (llcp >= rlcp) {
                // sharing more with left than right
                size_t i = this->minq(l+1, mid);
                q = this->LCP[i]; // minLCP for suffixes in SA[l..m]
                if (q >= llcp) {
                    q = lcp_offset(P, this->strbegin, n, pos, llcp);
                }
            } else {
                size_t i = this->minq(mid+1, r); // minLCP for suffixes in SA[m..r]
                q = this->LCP[i];
                if (q >= rlcp) {
                    q = lcp_offset(P, this->strbegin, n, pos, rlcp);
                }
            }

            if (q == m) {
                // found _a_ match, now find left and right boundaries
                assert(llcp < m && rlcp < m);

                // find ub
                r = ub_rmq(this->LCP, this->minq, mid, r, m);

                // find lb
                l = lb_rmq(this->LCP, this->minq, l, mid, m);

                return std::pair<index_t, index_t>(l, r+1);

            } else if (P[q] <= this->strbegin[pos+q]) {
                r = mid;
                rlcp = q;
            } else {
                l = mid;
                llcp = q;
            }
        }
        // found no match
        size_t lb = r;
        return std::pair<index_t,index_t>(lb, lb);
    }
};

template <typename index_t>
struct desa_index : public esa_index<index_t> {
    using it_t = typename esa_index<index_t>::it_t;
    // adds Lc array of characters
    std::vector<char> Lc;

    template <typename Iterator>
    void construct(Iterator begin, Iterator end) {
        esa_index<index_t>::construct(begin, end);

        // NOTE:
        //   LCP[i] = lcp(SA[i-1],SA[i])
        //   differing chars (needed for decision)
        //     left:  S[SA[i-1]+LCP[i]]   (need only one of these)
        //     right: S[SA[i]  +LCP[i]]
        // construct char array Lc[i] = S[SA[i-1]+LCP[i]], i=1,...n-1
        Lc.resize(this->n);
        for (size_t i = 1; i < this->n; ++i) {
            Lc[i] = this->strbegin[this->SA[i-1]+this->LCP[i]];
        }
    }

    // a ST node is virtually represented by it's interval [l,r] and it's first
    // child split point `i1`, where LCP[i1] = minLCP[l..r] is the string
    // depths `q` of the node. `c` is P[q], the (q+1)th char in P
    inline void find_child(size_t& l, size_t& i1, size_t& r, size_t& q, char c) const {
        assert(l < r);
        assert(l <= i1);
        assert(i1 <= r);
        do {
            // `i` is the lcp(sa[i-1],sa[i])
            char lc = this->lc[i1]; // == s[sa[l]+lcpv] for first iter
            if (lc == c) {
                r = i1-1;
                break;
            }
            l = i1;
            if (l == r)
                break;

            this->rmqtimer.tic();
            i1 = this->minq(l+1, r);
            this->rmqtimer.toc();
        } while (l < r && this->lcp[i1] == q);

        if (this->lcp[i1] == q) {
            if (l+1 < r) {
                this->rmqtimer.tic();
                i1 = this->minq(l+1, r);
                this->rmqtimer.toc();
            } else {
                i1 = l;
            }
        }
        q = this->lcp[i1];
    }

    /*
    std::pair<index_t,index_t> locate_possible(const std::string& P) {
        size_t n = this->n;
        size_t m = P.size();

        size_t l = 0;
        size_t r = n-1;
        //size_t q = 0; // size of current match

        if (l == r) {
            return std::pair<index_t, index_t>(l, r+1);
        }


        this->rmqtimer.tic();
        size_t i = this->minq(l+1, r);
        this->rmqtimer.toc();
        size_t q = this->LCP[i];

        while (q < m && l < r) {
            find_child(l, i, r, q, P[q]);
        }

        return std::pair<index_t, index_t>(l, r+1);
    }

    std::pair<index_t,index_t> locate(const std::string& P) {
        std::pair<index_t, index_t> res = locate_possible(P);
        // check if pattern matches
        if (res.first < res.second) {
            int cmp = strncmp(&this->strbegin[this->SA[res.first]], &P[0], P.size());
            if (cmp == 0) {
                return res;
            } else {
                return std::pair<index_t,index_t>(res.first, res.first);
            }
        }
        return res;
    }
    */

    /*
    std::pair<index_t,index_t> locate(const std::string& P) {

        size_t n = this->n;
        size_t m = P.size();
        size_t l = 0;
        size_t r = n-1;

        size_t q = 0; // size of current match

        while (q < m && l < r) {

            // NOTE: LCP[i] = lcp(SA[i-1],SA[i]), LCP[0] = 0
            // using [l,r] as an inclusive SA range
            // corresponding to LCP query range [l+1,r]

            // get first child interval and depth
            size_t i = this->minq(l+1, r);
            index_t lcpv = this->LCP[i];
            assert(lcpv >= q);

            // check if we've reached the end of the pattern
            if (lcpv >= m) {
                break;
            }

            char c = P[lcpv];
            do {
                // `i` is the lcp(SA[i-1],SA[i])
                char lc = this->Lc[i]; // == S[SA[l]+lcpv] for first iter
                if (lc == c) {
                    r = i-1;
                    break;
                }
                l = i;
                if (l == r)
                    break;

                i = this->minq(l+1, r);
            } while (l < r && this->LCP[i] == lcpv);

            if (this->strbegin[this->SA[l]+lcpv] == c) {
                // found the interval we were looking for
                q = lcpv+1;
            } else {
                return std::pair<index_t,index_t>(l,l);
            }
        }
        // check if pattern matches
        if (l <= r) {
            int cmp = strncmp(&this->strbegin[this->SA[l]], &P[0], P.size());
            if (cmp != 0) {
                return std::pair<index_t,index_t>(l, l);
            }
        }
        return std::pair<index_t,index_t>(l, r+1);
    }
    */


    template <typename String>
    inline std::pair<index_t,index_t> locate_possible(const String& P) const {

        size_t n = this->n;
        size_t m = P.size();
        size_t l = 0;
        size_t r = n-1;

        // get first child interval and depth
        size_t i = this->minq(l+1, r);
        index_t q = this->LCP[i];

        // blind search
        while (q < m && l < r) {

            // NOTE: LCP[i] = lcp(SA[i-1],SA[i]), LCP[0] = 0
            // using [l,r] as an inclusive SA range
            // corresponding to LCP query range [l+1,r]

            // check if we've reached the end of the pattern
            if (q >= m) {
                break;
            }

            do {
                // `i` is the lcp(SA[i-1],SA[i])
                char lc = this->Lc[i]; // == S[SA[l]+lcpv] for first iter
                if (lc == P[q]) {
                    r = i-1;
                    break;
                }
                l = i;
                if (l == r)
                    break;

                i = this->minq(l+1, r);
            } while (l < r && this->LCP[i] == q);

            if (this->LCP[i] == q) {
                if (l+1 < r) {
                    i = this->minq(l+1, r);
                } else {
                    i = l;
                }
            }
            q = this->LCP[i];
        }
        return std::pair<index_t,index_t>(l, r+1);
    }


    std::pair<index_t,index_t> locate(const std::string& P) const {

        size_t n = this->n;
        size_t m = P.size();
        size_t l = 0;
        size_t r = n-1;

        // get first child interval and depth
        size_t i = this->minq(l+1, r);
        index_t q = this->LCP[i];

        // blind search
        while (q < m && l < r) {

            // NOTE: LCP[i] = lcp(SA[i-1],SA[i]), LCP[0] = 0
            // using [l,r] as an inclusive SA range
            // corresponding to LCP query range [l+1,r]

            // check if we've reached the end of the pattern
            if (q >= m) {
                break;
            }

            do {
                // `i` is the lcp(SA[i-1],SA[i])
                char lc = this->Lc[i]; // == S[SA[l]+lcpv] for first iter
                if (lc == P[q]) {
                    r = i-1;
                    break;
                }
                l = i;
                if (l == r)
                    break;

                i = this->minq(l+1, r);
            } while (l < r && this->LCP[i] == q);

            if (this->LCP[i] == q) {
                if (l+1 < r) {
                    i = this->minq(l+1, r);
                } else {
                    i = l;
                }
            }
            q = this->LCP[i];
        }

        // check if pattern matches the string
        if (l <= r) {
            int cmp = strncmp(&this->strbegin[this->SA[l]], &P[0], P.size());
            if (cmp != 0) {
                return std::pair<index_t,index_t>(l, l);
            }
        }
        return std::pair<index_t,index_t>(l, r+1);
    }
};


template <typename index_t>
struct lookup_desa_index : public desa_index<index_t> {
    lookup_index<index_t> tl;

    template <typename Iterator>
    void construct(Iterator begin, Iterator end) {
        desa_index<index_t>::construct(begin, end);
        tl.construct(begin, end, 16); // automatically size `k` given table size and use alphabet size
    }

    std::pair<index_t,index_t> locate(const std::string& P) const {
        size_t m = P.size();

        index_t l, r;
        std::tie(l, r) = tl.lookup(P);
        if (l == r) {
            return std::pair<index_t, index_t>(l,l);
        }
        --r; // convert [l,r) to [l,r]

        if (P.size() > tl.k && l <= r) {
            // further narrow down search space
            if (l < r) {

                size_t i = this->minq(l+1, r);
                index_t q = this->LCP[i];
                assert(q >= tl.k);

                // blind search
                while (q < m && l < r) {

                    // NOTE: LCP[i] = lcp(SA[i-1],SA[i]), LCP[0] = 0
                    // using [l,r] as an inclusive SA range
                    // corresponding to LCP query range [l+1,r]

                    // check if we've reached the end of the pattern
                    if (q >= m) {
                        break;
                    }

                    do {
                        // `i` is the lcp(SA[i-1],SA[i])
                        char lc = this->Lc[i]; // == S[SA[l]+q] for first iter
                        if (lc == P[q]) {
                            r = i-1;
                            break;
                        }
                        l = i;
                        if (l == r)
                            break;
                        i = this->minq(l+1, r);
                    } while (l < r && this->LCP[i] == q);

                    if (this->LCP[i] == q) {
                        if (l+1 < r) {
                            i = this->minq(l+1, r);
                        } else {
                            i = l;
                        }
                    }
                    q = this->LCP[i];
                }
            }

            // check if pattern matches
            if (l <= r) {
                int cmp = strncmp(&this->strbegin[this->SA[l]], &P[0], P.size());
                if (cmp == 0) {
                    return std::pair<index_t, index_t>(l, r+1);
                } else {
                    // no match
                    return std::pair<index_t,index_t>(l, l);
                }
            }
        }
        return std::pair<index_t,index_t>(l, r+1);
    }
    /*
    std::pair<index_t,index_t> locate(const std::string& P) {
        size_t n = this->n;
        size_t m = P.size();

        index_t l, r;
        std::tie(l, r) = tl.lookup(P);
        if (l == r) {
            return std::pair<index_t, index_t>(l,l);
        }
        --r; // convert [l,r) to [l,r]

        if (P.size() > tl.k && l <= r) {
            // further narrow down search space
            if (l < r) {

                size_t i = this->minq(l+1, r);
                size_t q = this->LCP[i];
                assert(q >= tl.k);

                while (q < m && l < r) {

                    // NOTE: LCP[i] = lcp(SA[i-1],SA[i]), LCP[0] = 0
                    // using [l,r] as an inclusive SA range
                    // corresponding to LCP query range [l+1,r]

                    // get first child interval and depth
                    size_t i = this->minq(l+1, r);
                    index_t lcpv = this->LCP[i];
                    assert(lcpv >= q);

                    // check if we've reached the end of the pattern
                    if (lcpv >= m) {
                        break;
                    }

                    char c = P[lcpv];
                    do {
                        // `i` is the lcp(SA[i-1],SA[i])
                        char lc = this->Lc[i]; // == S[SA[l]+lcpv] for first iter
                        if (lc == c) {
                            r = i-1;
                            break;
                        }
                        l = i;
                        if (l == r)
                            break;
                        i = this->minq(l+1, r);
                    } while (l < r && this->LCP[i] == lcpv);

                    if (this->strbegin[this->SA[l]+lcpv] == c) {
                        // found the interval we were looking for
                        q = lcpv+1;
                    } else {
                        return std::pair<index_t,index_t>(l,l);
                    }
                }
            }

            // check if pattern matches
            if (l <= r) {
                int cmp = strncmp(&this->strbegin[this->SA[l]], &P[0], P.size());
                if (cmp == 0) {
                    return std::pair<index_t, index_t>(l, r+1);
                } else {
                    // no match
                    return std::pair<index_t,index_t>(l, l);
                }
            }
        }
        return std::pair<index_t,index_t>(l, r+1);
    }
    */
    /*
    std::pair<index_t,index_t> locate(const std::string& P) {
        size_t m = P.size();

        index_t l, r;
        std::tie(l, r) = tl.lookup(P);
        if (l == r) {
            return std::pair<index_t, index_t>(l,l);
        }
        --r; // convert [l,r) to [l,r]

        if (P.size() > tl.k && l <= r) {
            // further narrow down search space
            if (l < r) {
                this->rmqtimer.tic();
                size_t i = this->minq(l+1, r);
                this->rmqtimer.toc();
                size_t q = this->LCP[i];
                assert(q >= tl.k);

                while (q < m && l < r) {
                    this->find_child(l, i, r, q, P[q]);
                }
            }

            // check if pattern matches
            if (l <= r) {
                int cmp = strncmp(&this->strbegin[this->SA[l]], &P[0], P.size());
                if (cmp == 0) {
                    return std::pair<index_t, index_t>(l, r+1);
                } else {
                    // no match
                    return std::pair<index_t,index_t>(l, l);
                }
            }
        }
        return std::pair<index_t,index_t>(l, r+1);
    }
    */
};



#endif // SEQ_QUERY_HPP
