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

#ifndef SUFFIX_TREE_HPP
#define SUFFIX_TREE_HPP

#include <vector>

#include <mxx/comm.hpp>
#include <mxx/timer.hpp>

#include <suffix_array.hpp>
#include <ansv.hpp>

#include <bulk_rma.hpp>

template <typename Func, typename InputIterator, typename index_t = std::size_t>
void for_each_parent(const suffix_array<InputIterator, index_t, true>& sa, Func func, const mxx::comm& comm) {
    mxx::section_timer t(std::cerr, comm);
    // get input sizes
    size_t local_size = sa.local_SA.size();
    size_t global_size = mxx::allreduce(local_size, comm);
    size_t prefix = mxx::exscan(local_size, comm);
    // assert n >= p, or rather at least one element per process
    MXX_ASSERT(mxx::all_of(local_size >= 1, comm));
    // ansv of lcp!
    // TODO: use index_t instead of size_t
    std::vector<size_t> left_nsv;
    std::vector<size_t> right_nsv;
    std::vector<std::pair<index_t, size_t>> lr_mins;

    const size_t nonsv = std::numeric_limits<size_t>::max();
    t.end_section("pre ansv");

    // ANSV with furthest eq for left and smallest for right
    ansv<index_t, furthest_eq, nearest_sm, local_indexing>(sa.local_LCP, left_nsv, right_nsv, lr_mins, comm, nonsv);
    t.end_section("ansv");

    // each SA[i] lies between two LCP values
    // LCP[i] = lcp(S[SA[i-1]], S[SA[i]])
    // leaf nodes are the suffix array positions. Their parent is the either their left or their right
    // LCP, depending on which one is larger

    // get the first LCP value of the next processor
    index_t next_first_lcp = mxx::left_shift(sa.local_LCP[0], comm);
    for (size_t i = 0; i < local_size; ++i) {
        // for each suffix array position SA[i], we check the longest-common-prefix
        // with the neighboring suffixes SA[i-1] and SA[i+1]. Whichever one it
        // shares the larger common prefix with, is its sibling in the ST and
        // they share a parent at the depth given by the larger LCP value. The
        // index of the LCP that has that value will be the index of the parent
        // node.
        //
        // This means for every `i`, we need argmax_i {LCP[i], LCP[i+1]}, where
        // `i+1` might be on the next processor.
        //
        // If there are multiple leafs > 2 for an internal node, the parent
        // will be the index of the furthest equal element. We thus need
        // to use the NSV for determining the left parent.
        // If the right LCP is larger, then that one is the direct parent,
        // since there can't be any equal elements to the left (since the
        // right one was larger).

        // parent will be an index into LCP
        size_t parent = std::numeric_limits<size_t>::max();
        index_t lcp_val;

        // the globally first element has parent 1
        if (comm.rank() == 0 && i == 0) {
            // globally first leaf: SA[0]
            if (local_size > 1) {
                lcp_val = sa.local_LCP[1];
            } else {
                MXX_ASSERT(global_size > 1);
                lcp_val = next_first_lcp;
            }
            // -> parent = 1, since it is the common prefix between SA[0] and SA[1]
            // unless the lcp is 0, then this leaf is connected
            // directly to the root node (parent = 0)
            parent = lcp_val > 0 ? 1 : 0;
        } else {
            // To determine whether the left or right LCP is the parent,
            // we take the max of LCP[i]=lcp(SA[i-1],SA[i]) and LCP[i+1]=lcp(SA[i], SA[i+1])
            // There are two special cases to handle:
            // 1) locally last element: we need to use the first LCP value of the next processor
            //    in place of LCP[i+1]
            // 2) globally last element: parent is always the left furthest eq nsv
            if ((i == local_size-1
                 && (comm.rank() == comm.size() || sa.local_LCP[local_size-1] >= next_first_lcp))
                || (i < local_size-1 && sa.local_LCP[i] >= sa.local_LCP[i+1])) {
                // the parent is the left furthest eq or nearest sm
                size_t nsv;
                if (left_nsv[i] < local_size) {
                    nsv = prefix + left_nsv[i];
                    lcp_val = sa.local_LCP[left_nsv[i]];
                } else {
                    nsv = lr_mins[left_nsv[i] - local_size].second;
                    lcp_val = lr_mins[left_nsv[i] - local_size].first;
                }
                if (lcp_val == sa.local_LCP[i]) {
                    parent = nsv;
                } else {
                    parent = prefix + i;
                    lcp_val = sa.local_LCP[i];
                }
            } else {
                // SA[i] shares a longer prefix with its right neighbor SA[i+1]
                // they converge at internal node prefix+i+1
                parent = prefix + i + 1;
                if (i == local_size - 1)
                    lcp_val = next_first_lcp;
                else
                    lcp_val = sa.local_LCP[i+1];
            }
        }
        func(i, global_size + prefix + i, parent, lcp_val);
    }

    // get parents of internal nodes (via LCP)
    for (size_t i = 0; i < local_size; ++i) {
        size_t parent = std::numeric_limits<size_t>::max();
        index_t lcp_val;
        // for each LCP position, get ANSV left-furthest-eq and right-nearest-sm
        // and the max of the two is the parent
        // Special cases: first (LCP[0]) and globally last LCP
        if (comm.rank() == 0 && i == 0) {
            // this is the root node and it has no parent!
            continue;

        //} else if (comm.rank() == comm.size() - 1 && i == local_size - 1) {
            // globally last element (no right ansv)
            // this case is identical to the regular case, since for the right
            // most element, right_nsv[i] will be == nonsv
            // and as such is handled in the corresponding case below
        } else {
            if (sa.local_LCP[i] == 0) {
                // this is a dupliate of the root node which is located at
                // position 0 on processor 0
                continue;
            } else {
                // left NSV can't be non-existant because LCP[0] = 0
                assert(left_nsv[i] != nonsv);
                if (right_nsv[i] == nonsv) {
                    // use left one
                    size_t nsv;
                    if (left_nsv[i] < local_size) {
                        nsv = prefix + left_nsv[i];
                        lcp_val = sa.local_LCP[left_nsv[i]];
                    } else {
                        nsv = lr_mins[left_nsv[i] - local_size].second;
                        lcp_val = lr_mins[left_nsv[i] - local_size].first;
                    }
                    if (lcp_val == sa.local_LCP[i]) {
                        // duplicate node, don't add!
                        continue;
                    }
                    parent = nsv;
                } else {
                    // get left NSV index and value
                    size_t lnsv;
                    index_t left_lcp_val;
                    if (left_nsv[i] < local_size) {
                        lnsv = prefix + left_nsv[i];
                        left_lcp_val = sa.local_LCP[left_nsv[i]];
                    } else {
                        lnsv = lr_mins[left_nsv[i] - local_size].second;
                        left_lcp_val = lr_mins[left_nsv[i] - local_size].first;
                    }
                    // get right NSV index and value
                    size_t rnsv;
                    index_t right_lcp_val;
                    if (right_nsv[i] < local_size) {
                        rnsv = prefix + right_nsv[i];
                        right_lcp_val = sa.local_LCP[right_nsv[i]];
                    } else {
                        rnsv = lr_mins[right_nsv[i] - local_size].second;
                        right_lcp_val = lr_mins[right_nsv[i] - local_size].first;
                    }
                    // parent is the NSV for which LCP is larger.
                    // if same, use left furthest_eq
                    if (left_lcp_val >= right_lcp_val) {
                        if (left_lcp_val == sa.local_LCP[i]) {
                            // this is a duplicate node, and won't be added
                            continue;
                        }
                        parent = lnsv;
                        lcp_val = left_lcp_val;
                    } else {
                        parent = rnsv;
                        lcp_val = right_lcp_val;
                    }
                }
            }
        }
        func(i, prefix + i, parent, lcp_val);
    }
}

constexpr int edgechar_twophase_all2all = 1;
constexpr int edgechar_bulk_rma = 2;
constexpr int edgechar_mpi_osc_rma = 3;
constexpr int edgechar_rma_shared = 4;
constexpr int edgechar_posix_sm = 5;
constexpr int edgechar_posix_sm_split = 6;

//#if SHARED_MEM
constexpr int edgechar_default = edgechar_bulk_rma;
/*
#else
constexpr int edgechar_default = edgechar_bulk_rma;
#endif
*/


/*
 * A rather in-efficient method, that requires two complete global shuffles
 * to request the edge character data
 *
 * This can be deleted eventually.
 */
template <typename InputIterator, typename index_t = std::size_t>
std::vector<size_t> construct_st_2phase(const suffix_array<InputIterator, index_t, true>& sa, const mxx::comm& comm) {
    mxx::section_timer t(std::cerr, comm);
    // get input sizes
    size_t local_size = sa.local_SA.size();
    size_t global_size = mxx::allreduce(local_size, comm);
    size_t prefix = mxx::exscan(local_size, comm);
    // assert n >= p, or rather at least one element per process
    MXX_ASSERT(mxx::all_of(local_size >= 1, comm));

    std::vector<std::tuple<size_t, size_t, size_t>> parent_reqs;
    parent_reqs.reserve(2*local_size);
    // parent request where the character is the last `$`/`0` character
    // these don't have to be requested, but are locally fulfilled
    std::vector<std::tuple<size_t, size_t, size_t>> dollar_reqs;

    for_each_parent(sa, [&](size_t i, size_t gidx, size_t parent, size_t lcp_val) {
        if (sa.local_SA[i] + lcp_val >= global_size) {
            MXX_ASSERT(sa.local_SA[i] + lcp_val == global_size);
            dollar_reqs.emplace_back(parent, gidx, 0);
        } else {
            parent_reqs.emplace_back(parent, gidx, sa.local_SA[i] + lcp_val);
        }
    }, comm);
    t.end_section("locally calc parents");

    typedef typename std::iterator_traits<InputIterator>::value_type CharT;
    std::vector<CharT> edge_chars;

    // This is a slower method, because it sends all edges to the position
    // of the character first, and then back to the position of the parent.
    // Most parents are on the same processor as the child node, thus
    // this requires a lot more communication then necessary
    // 1) send tuples (parent, i, SA[i]+LCP[i]) to 3rd index)
    mxx::partition::block_decomposition_buffered<size_t> part(global_size, comm.size(), comm.rank());
    // send all requests to the process on which the character for the
    // character request lies
    mxx::all2all_func(parent_reqs, [&part](const std::tuple<size_t,size_t,size_t>& t) {return part.target_processor(std::get<2>(t));}, comm);
    t.end_section("all2all_func: req characters");

    // replace string request with character from original string
    for (size_t i = 0; i < parent_reqs.size(); ++i) {
        size_t offset = std::get<2>(parent_reqs[i]);
        if (offset == global_size) {
            // the artificial last `$` character is mapped to 0
            std::get<2>(parent_reqs[i]) = 0;
        } else {
            // get character from that global string position
            std::get<2>(parent_reqs[i]) = sa.alphabet_mapping[static_cast<size_t>(*(sa.input_begin+(std::get<2>(parent_reqs[i])-prefix)))];
        }
    }
    // append the "dollar" requests
    parent_reqs.insert(parent_reqs.end(), dollar_reqs.begin(), dollar_reqs.end());
    dollar_reqs.clear(); dollar_reqs.shrink_to_fit();

    t.end_section("locally answer char queries");

    // 2) send tuples (parent, i, S[SA[i]+LCP[i]) to 1st index) [to parent]
    mxx::all2all_func(parent_reqs, [&part](const std::tuple<size_t,size_t,size_t>& t) {return part.target_processor(std::get<0>(t));}, comm);
    t.end_section("all2all_func: send to parent");

    // one internal node for each LCP entry, each internal node is sigma cells
    std::vector<size_t> internal_nodes((sa.sigma+1)*local_size);
    for (size_t i = 0; i < parent_reqs.size(); ++i) {
        size_t parent = std::get<0>(parent_reqs[i]);
        size_t node_idx = (parent - prefix)*(sa.sigma+1);
        uint16_t c = std::get<2>(parent_reqs[i]);
        MXX_ASSERT(0 <= c && c < sa.sigma+1);
        size_t cell_idx = node_idx + c;
        internal_nodes[cell_idx] = std::get<1>(parent_reqs[i]);
    }

    t.end_section("locally: create internal nodes");

    return internal_nodes;
}

// original implementation used for SC16 and IPDPS17 papers
template <typename InputIterator, typename index_t = std::size_t, int edgechar_method = edgechar_default>
std::vector<size_t> construct_suffix_tree(const suffix_array<InputIterator, index_t, true>& sa, const mxx::comm& comm) {
    mxx::section_timer t(std::cerr, comm);
    // get input sizes
    size_t local_size = sa.local_SA.size();
    size_t global_size = mxx::allreduce(local_size, comm);
    size_t prefix = mxx::exscan(local_size, comm);
    // assert n >= p, or rather at least one element per process
    MXX_ASSERT(mxx::all_of(local_size >= 1, comm));

    std::vector<std::tuple<size_t, size_t, size_t>> parent_reqs;
    parent_reqs.reserve(2*local_size);
    // parent request where the character is the last `$`/`0` character
    // these don't have to be requested, but are locally fulfilled
    std::vector<std::tuple<size_t, size_t, size_t>> dollar_reqs;
    std::vector<std::tuple<size_t, size_t, size_t>> remote_reqs;

    for_each_parent(sa, [&](size_t i, size_t gidx, size_t parent, size_t lcp_val) {
        if (prefix <= parent && parent < prefix + local_size) {
            parent_reqs.emplace_back(parent, gidx, sa.local_SA[i] + lcp_val);
        } else {
            remote_reqs.emplace_back(parent, gidx, sa.local_SA[i] + lcp_val);
        }
    }, comm);
    t.end_section("locally calc parents");

    // TODO: plus distinguish between dollar/parent req only for the first method
    typedef typename std::iterator_traits<InputIterator>::value_type CharT;
    std::vector<CharT> edge_chars;
    if (edgechar_method == edgechar_bulk_rma) {
        mxx::partition::block_decomposition_buffered<size_t> part(global_size, comm.size(), comm.rank());
        // send those edges for which the parent lies on a remote processor
        typedef std::tuple<size_t, size_t, size_t> Tp;
        mxx::all2all_func(remote_reqs, [&part](const Tp& t) {return part.target_processor(std::get<0>(t));}, comm);
        parent_reqs.insert(parent_reqs.end(), remote_reqs.begin(), remote_reqs.end());
        remote_reqs = std::vector<Tp>();
        t.end_section("bulk_rma: send to parent");

        // only query for those with offset != global_size
        // bucket by target processor of the character request
        auto dollar_begin = std::partition(parent_reqs.begin(), parent_reqs.end(), [&global_size](const Tp& x){return std::get<2>(x) < global_size;});
        dollar_reqs = std::vector<Tp>(dollar_begin, parent_reqs.end());
        parent_reqs.resize(std::distance(parent_reqs.begin(), dollar_begin));
        t.end_section("bulk_rma: partition dollars");

        // bucket the String index by target processor (the character we need for this edge)
        // as a pre-step for the bulk_rma (which requires things to be bucketed by target processor)
        std::vector<size_t> send_counts = mxx::bucketing(parent_reqs, [&part](const std::tuple<size_t,size_t,size_t>& t) { return part.target_processor(std::get<2>(t));}, comm.size());
        t.end_section("bulk_rma: bucketing by char index");
        // create request address vector
        std::vector<size_t> global_indexes(parent_reqs.size());
        for (size_t i = 0; i < parent_reqs.size(); ++i) {
            global_indexes[i] = std::get<2>(parent_reqs[i]);
        }
        t.end_section("bulk_rma: create global_indexes");
        // use global bulk RMA for getting the corresponding characters
        edge_chars = bulk_rma(sa.input_begin, sa.input_end, global_indexes, send_counts, comm);
        t.end_section("bulk_rma: bulk_rma");
    } else {
        mxx::partition::block_decomposition_buffered<size_t> part(global_size, comm.size(), comm.rank());
        // send those edges for which the parent lies on a remote processor
        mxx::all2all_func(remote_reqs, [&part](const std::tuple<size_t,size_t,size_t>& t) {return part.target_processor(std::get<0>(t));}, comm);
        parent_reqs.insert(parent_reqs.end(), remote_reqs.begin(), remote_reqs.end());
        t.end_section("all2all_func: send to parent");

        std::vector<size_t> global_indexes(parent_reqs.size());
        for (size_t i = 0; i < parent_reqs.size(); ++i) {
            global_indexes[i] = std::get<2>(parent_reqs[i]);
        }
        t.end_section("create global_indexes");

        // TODO: bulk_rma_mpi only for non-dollar
        if (edgechar_method == edgechar_mpi_osc_rma) {
            edge_chars = bulk_rma_mpiwin(sa.input_begin, sa.input_end, global_indexes, comm);
        } else if (edgechar_method == edgechar_rma_shared) {
            edge_chars = bulk_rma_shm_mpi(sa.input_begin, sa.input_end, global_indexes, comm);
        } else if (edgechar_method == edgechar_posix_sm) {
            edge_chars = bulk_rma_shm_posix(sa.input_begin, sa.input_end, global_indexes, comm);
        } else if (edgechar_method == edgechar_posix_sm_split) {
            edge_chars = bulk_rma_shm_posix_split(sa.input_begin, sa.input_end, global_indexes, comm);
        }
        t.end_section("RMA read chars");
    }

    // TODO: (alternatives for full lookup table in each node:)
    // local hashing key=(node-idx, char), value=(child idx)
    //            or multimap key=(node-idx), value=(char, child idx)
    //            2nd enables iteration over children, but not direct lookup
    //            of specific child
    //            2nd no different than fixed std::vector<std::list>

    // one internal node for each LCP entry, each internal node is sigma cells
    std::vector<size_t> internal_nodes((sa.sigma+1)*local_size);
    for (size_t i = 0; i < parent_reqs.size(); ++i) {
        size_t parent = std::get<0>(parent_reqs[i]);
        size_t node_idx = (parent - prefix)*(sa.sigma+1);
        uint16_t c;
        CharT x = edge_chars[i];
        if (x == 0) {
            c = 0;
        } else {
            c = sa.alphabet_mapping[x];
        }
        MXX_ASSERT(0 <= c && c < sa.sigma+1);
        size_t cell_idx = node_idx + c;
        internal_nodes[cell_idx] = std::get<1>(parent_reqs[i]);
    }
    if (edgechar_method == edgechar_bulk_rma) {
        for (size_t i = 0; i < dollar_reqs.size(); ++i) {
            size_t parent = std::get<0>(dollar_reqs[i]);
            size_t node_idx = (parent - prefix)*(sa.sigma+1);
            internal_nodes[node_idx] = std::get<1>(dollar_reqs[i]);
        }
    }

    t.end_section("locally: create internal nodes");

    return internal_nodes;
}

struct edge {
    size_t parent;
    size_t gidx;

    edge() = default;
    edge(const edge& o) = default;
    edge(edge&& o) = default;
    edge(size_t parent, size_t gidx) : parent(parent), gidx(gidx) {};

    edge& operator=(const edge& o) = default;
    edge& operator=(edge&& o) = default;
};

std::ostream& operator<<(std::ostream& os, const edge& e) {
    return os << "(" << e.parent << "," << e.gidx << ")";
}

MXX_CUSTOM_STRUCT(edge, parent, gidx);

template <typename InputIterator, typename index_t = std::size_t, int edgechar_method = edgechar_default>
std::vector<size_t> construct_suffix_tree_edges(const suffix_array<InputIterator, index_t, true>& sa, const mxx::comm& comm) {
    mxx::section_timer t(std::cerr, comm);

    // get input sizes
    size_t local_size = sa.local_SA.size();
    size_t global_size = mxx::allreduce(local_size, comm);
    size_t prefix = mxx::exscan(local_size, comm);
    // assert n >= p, or rather at least one element per process
    MXX_ASSERT(mxx::all_of(local_size >= 1, comm));

    std::vector<edge> edges;
    edges.reserve(2*local_size);

    std::vector<size_t> char_indexes;
    char_indexes.reserve(2*local_size);

    // parent request where the character is the last `$`/`0` character
    // these don't have to be requested, but are locally fulfilled
    std::vector<edge> dollar_edges;
    std::vector<std::pair<edge, size_t>> remote_edges;
    //std::vector<size_t> remote_char_indexes;

    for_each_parent(sa, [&](size_t i, size_t gidx, size_t parent, size_t lcp_val) {
        size_t char_idx = sa.local_SA[i] + lcp_val;
        // remote or local?
        if (prefix <= parent && parent < prefix + local_size) {
            if (char_idx < global_size) {
                edges.emplace_back(parent, gidx);
                char_indexes.push_back(char_idx);
            } else {
                dollar_edges.emplace_back(parent, gidx);
            }
        } else {
            remote_edges.emplace_back(edge(parent, gidx), char_idx);
        }
    }, comm);
    t.end_section("locally calc parents");

    mxx::sync_cout(comm) << "regular edges: " << edges.size() << "/" << 2*local_size << ", dollar: " << dollar_edges.size() << ", remote: " << remote_edges.size() << std::endl;

    // TODO: plus distinguish between dollar/parent req only for the first method
    typedef typename std::iterator_traits<InputIterator>::value_type CharT;
    std::vector<CharT> edge_chars;

    mxx::partition::block_decomposition_buffered<size_t> part(global_size, comm.size(), comm.rank());
    // send those edges for which the parent lies on a remote processor
    mxx::all2all_func(remote_edges, [&part](const std::pair<edge,size_t>& e) {return part.target_processor(e.first.parent);}, comm);
    for (auto& p : remote_edges) {
        if (p.second < global_size) {
            edges.emplace_back(p.first);
            char_indexes.push_back(p.second);
        } else {
            dollar_edges.emplace_back(p.first);
        }
    }
    t.end_section("bulk_rma: send to parent");

    if (edgechar_method == edgechar_bulk_rma) {
        // use global bulk RMA for getting the corresponding characters
        edge_chars = bulk_rma(sa.input_begin, sa.input_end, char_indexes, comm);
        t.end_section("bulk_rma: bulk_rma");
    } else {
        // shared memory versions
        if (edgechar_method == edgechar_mpi_osc_rma) {
            edge_chars = bulk_rma_mpiwin(sa.input_begin, sa.input_end, char_indexes, comm);
        } else if (edgechar_method == edgechar_rma_shared) {
            edge_chars = bulk_rma_shm_mpi(sa.input_begin, sa.input_end, char_indexes, comm);
        } else if (edgechar_method == edgechar_posix_sm) {
            edge_chars = bulk_rma_shm_posix(sa.input_begin, sa.input_end, char_indexes, comm);
        } else if (edgechar_method == edgechar_posix_sm_split) {
            edge_chars = bulk_rma_shm_posix_split(sa.input_begin, sa.input_end, char_indexes, comm);
        }
        t.end_section("RMA read chars");
    }

    // one internal node for each LCP entry, each internal node is sigma cells
    std::vector<size_t> internal_nodes((sa.sigma+1)*local_size);
    for (size_t i = 0; i < edges.size(); ++i) {
        size_t node_idx = (edges[i].parent - prefix)*(sa.sigma+1);
        CharT x = edge_chars[i];
        uint16_t c = sa.alphabet_mapping[x];
        MXX_ASSERT(0 <= c && c < sa.sigma+1);
        size_t cell_idx = node_idx + c;
        internal_nodes[cell_idx] = edges[i].gidx;
    }

    // process dollar edges
    for (size_t i = 0; i < dollar_edges.size(); ++i) {
        size_t parent = dollar_edges[i].parent;
        size_t node_idx = (parent - prefix)*(sa.sigma+1);
        internal_nodes[node_idx] = dollar_edges[i].gidx;
    }

    t.end_section("locally: create internal nodes");

    return internal_nodes;
}

// interleave SA and LCP and get index after
inline size_t interleaved_val(size_t idx, size_t global_size) {
    return (idx >= global_size) ? (2*(idx-global_size)+1) : 2*idx;
}

class stopwatch {
    std::chrono::steady_clock::time_point start_time;
    typedef std::chrono::steady_clock::time_point::duration duration;
    duration elapsed;

public:
    stopwatch() : elapsed(duration::zero()) {};

    void start() {
        start_time = std::chrono::steady_clock::now();
    }

    void pause() {
        elapsed += duration(std::chrono::steady_clock::now()-start_time);
    }

    template <typename dur = std::chrono::duration<double, std::milli>>
    typename dur::rep total() const {
        return dur(elapsed).count();
    }
};

// experimental shared memory implementation for suffix tree construction
template <typename InputIterator, typename index_t = std::size_t, int edgechar_method = edgechar_default>
std::vector<size_t> construct_suffix_tree_sm(const suffix_array<InputIterator, index_t, true>& sa, const mxx::comm& comm) {
    mxx::section_timer t(std::cerr, comm);
    typedef typename std::iterator_traits<InputIterator>::value_type CharT;

    // get input sizes
    size_t local_size = sa.local_SA.size();
    size_t global_size = mxx::allreduce(local_size, comm);
    size_t prefix = mxx::exscan(local_size, comm);
    // assert n >= p, or rather at least one element per process
    MXX_ASSERT(mxx::all_of(local_size >= 1, comm));

    std::vector<edge> edges;
    //edges.reserve(2*local_size);

    std::vector<size_t> char_indexes;
    //char_indexes.reserve(2*local_size);

    // parent request where the character is the last `$`/`0` character
    // these don't have to be requested, but are locally fulfilled
    std::vector<edge> dollar_edges;
    std::vector<std::pair<edge, size_t>> remote_edges;

    // create shared memory window over input string
    shmem_window_posix_split<CharT> win(sa.input_begin, sa.input_end, comm);
    t.end_section("create shared mem window");

    size_t sigma = sa.sigma+1;
    std::vector<size_t> internal_nodes(sigma*local_size);

    for_each_parent(sa, [&](size_t i, size_t gidx, size_t parent, size_t lcp_val) {
        size_t char_idx = sa.local_SA[i] + lcp_val;
        // remote or local?
        if (prefix <= parent && parent < prefix + local_size) {
            if (char_idx < global_size) {
                // fill node without internal node ordering
                size_t local_nodeidx = sigma*(parent-prefix);

                CharT x = win.get(char_idx);
                uint16_t cval = sa.alphabet_mapping[x];
                internal_nodes[local_nodeidx+cval] = gidx;
                // insert into next open slot in node
                /*
                for (size_t a = 1; a < sigma; ++a) {
                    if (internal_nodes[local_nodeidx+a] == 0) {
                        internal_nodes[local_nodeidx+a] = gidx;
                        break;
                    }
                }
                */
            } else {
                size_t local_nodeidx = sigma*(parent-prefix);
                internal_nodes[local_nodeidx + 0] = gidx;
            }
        } else {
            remote_edges.emplace_back(edge(parent, gidx), char_idx);
        }
    }, comm);
    t.end_section("locally calc parents");


    /* process all nodes and request chars if needed */
    /*
    std::vector<CharT> chars(sigma);
    std::vector<size_t> node_copy(sigma-1);
    stopwatch timer_find;
    stopwatch timer_sort;
    stopwatch timer_get_char;
    stopwatch timer_fill;
    for (size_t node = 0; node < local_size*sigma; node+=sigma) {
        // skip empty nodes
        if (internal_nodes[node+1] == 0)
            continue;
        unsigned int a = 1; // count characters (must be at least 2)
        for (; a < sigma; ++a) {
            if (internal_nodes[node+a] == 0)
                break;
        }
        if (a == sigma) {
            // full node, don't need to request characters: sort by interleaved value
            std::sort(internal_nodes.begin()+node+1, internal_nodes.begin()+node+a,
                [global_size](size_t i1, size_t i2) {
                    return interleaved_val(i1, global_size) < interleaved_val(i2, global_size);
            });
        } else {
            // get all characters
            std::copy(internal_nodes.begin()+node+1, internal_nodes.begin()+node+sigma, node_copy.begin());
            for (size_t c = 1; c < a; ++c) {
                size_t gidx = internal_nodes[node+c];
                size_t lidx = (gidx >= global_size) ? gidx - global_size - prefix : gidx - prefix;
                chars[c] = win.get(sa.local_SA[lidx]+sa.local_LCP[node/sigma]);
            }
            std::fill(internal_nodes.begin()+node+1, internal_nodes.begin()+node+sigma, 0);

            for (size_t c = 1; c < a; ++c) {
                CharT x = chars[c];
                uint16_t cval = sa.alphabet_mapping[x];
                internal_nodes[node+cval] = node_copy[c-1];
            }
        }
    }
    t.end_section("order edges in nodes");
    */
    //mxx::sync_cout(comm) << "time breakdown: find: " << timer_find.total() << "\tsort: " << timer_sort.total() << "\tget_char: " << timer_get_char.total() << "\tfill: " << timer_fill.total() << std::endl;

    /* process remote edges */

    mxx::partition::block_decomposition_buffered<size_t> part(global_size, comm.size(), comm.rank());
    // send those edges for which the parent lies on a remote processor
    mxx::all2all_func(remote_edges, [&part](const std::pair<edge,size_t>& e) {return part.target_processor(e.first.parent);}, comm);
    for (auto& p : remote_edges) {
        if (p.second < global_size) {
            size_t local_nodeidx = sigma*(p.first.parent-prefix);
            CharT x = win.get(p.second);
            uint16_t cval = sa.alphabet_mapping[x];
            internal_nodes[local_nodeidx + cval] = p.first.gidx;
        } else {
            size_t local_nodeidx = sigma*(p.first.parent-prefix);
            internal_nodes[local_nodeidx + 0] = p.first.gidx;
        }
    }
    t.end_section("send and process remote edges");

    return internal_nodes;
}


#endif // SUFFIX_TREE_HPP
