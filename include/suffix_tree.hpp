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

template <typename Q, typename Func>

std::vector<typename std::result_of<Func(Q)>::type> bulk_query(const std::vector<Q>& queries, Func f, const std::vector<size_t>& send_counts, const mxx::comm& comm) {
    // type of the query results
    typedef typename std::result_of<Func(Q)>::type T;
    mxx::section_timer t(std::cerr, comm);

    // get receive counts (needed as send counts for returning queries)
    std::vector<size_t> recv_counts = mxx::all2all(send_counts, comm);
    t.end_section("bulk_query: get recv_counts");

    // send all queries via all2all
    std::vector<Q> local_queries = mxx::all2allv(queries, send_counts, recv_counts, comm);
    t.end_section("bulk_query: all2all queries");

    // TODO: show load inbalance in queries and recv_counts?
    size_t recv_num = local_queries.size();
    std::pair<size_t, int> maxel = mxx::max_element(recv_num, comm);
    size_t total_queries = mxx::allreduce(queries.size(), comm);
    std::vector<size_t> recv_per_proc = mxx::gather(recv_num, 0, comm);
    if (comm.rank() == 0) {
        std::cerr << "Avg queries: " << total_queries * 1.0 / comm.size() << ", max queries on proc " << maxel.second << ": " << maxel.first << std::endl;
        std::cerr << "Inbalance factor: " << maxel.first * comm.size() * 1.0 / total_queries << "x" << std::endl;
        //std::cerr << "Queries received by each processor: " << recv_per_proc << std::endl;
    }

    // locally use query function for querying and save results
    std::vector<T> local_results(local_queries.size());
    for (size_t i = 0; i < local_queries.size(); ++i) {
        local_results[i] = f(local_queries[i]);
    }
    // now we can free the memory used for queries
    local_queries = std::vector<Q>();
    t.end_section("bulk_query: local query");

    // return all results, send_counts are the same as the recv_counts from the
    // previous all2all, and the other way around
    std::vector<T> results = mxx::all2allv(local_results, recv_counts, send_counts, comm);
    t.end_section("bulk_query: all2all query results");
    return results;
}

// global_adrs don't need to be sorted by address, but sorted by target processor
// TODO: generalize for where the global addresses/offsets are part of another data structure
template <typename InputIter>
std::vector<typename std::iterator_traits<InputIter>::value_type>
bulk_rma(InputIter local_begin, InputIter local_end,
         const std::vector<size_t>& queries, const std::vector<size_t>& send_counts, const mxx::comm& comm) {

    // get local and global size
    size_t local_size = std::distance(local_begin, local_end);
    size_t prefix = mxx::exscan(local_size, comm);

    return bulk_query(queries,
                      [&local_begin, &prefix](size_t gladr) {
                            return *(local_begin + (gladr - prefix));
                      }, send_counts, comm);
}


template <typename InputIter>
std::vector<typename std::iterator_traits<InputIter>::value_type>
bulk_rma(InputIter local_begin, InputIter local_end,
         const std::vector<size_t>& global_indexes, const mxx::comm& comm) {
    // get local and global size
    size_t local_size = std::distance(local_begin, local_end);
    size_t global_size = mxx::allreduce(local_size, comm);
    // get the block decomposition class and check that input is actuall block
    // decomposed
    // TODO: at one point, refactor this crap:
    mxx::partition::block_decomposition_buffered<size_t> part(global_size, comm.size(), comm.rank());
    MXX_ASSERT(part.local_size() == local_size);

    // get send counts by linear scan (TODO: could get for free with the bucketing)
    // or cheaper with p*log(n)
    std::vector<size_t> send_counts(comm.size(), 0);
    int cur_p = 0;
    for (size_t i = 0; i < local_size; ++i) {
        int t = part.target_processor(global_indexes[i]);
        MXX_ASSERT(cur_p <= t);
        ++send_counts[t];
        cur_p = t;
    }

    return bulk_rma(local_begin, local_end, global_indexes, send_counts, comm);
}


constexpr int edgechar_twophase_all2all = 1;
constexpr int edgechar_bulk_rma = 2;
constexpr int edgechar_mpi_osc_rma = 3;
constexpr int edgechar_rma_shared = 4;

#if SHARED_MEM
constexpr int edgechar_default = edgechar_rma_shared;
#else
constexpr int edgechar_default = edgechar_bulk_rma;
#endif

template <typename InputIterator, typename index_t = std::size_t, int edgechar_method = edgechar_default>
std::vector<size_t> construct_suffix_tree(const suffix_array<InputIterator, index_t, true>& sa, const mxx::comm& comm) {
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

    //const size_t sigma = 4; // TODO determine sigma or use local hashing
    // at most `n` internal nodes?
    // we represent only internal nodes. if a child pointer points to {total-size + X}, its a leaf
    // leafs are not represented as tree nodes inside the suffix tree structure
    // edges contain starting character? and/or length?
    //std::vector<size_t> tree_nodes(sigma*local_size); // TODO: determine real size

    // each SA[i] lies between two LCP values
    // LCP[i] = lcp(S[SA[i-1]], S[SA[i]])
    // leaf nodes are the suffix array positions. Their parent is the either their left or their right
    // LCP, depending on which one is larger

    std::vector<std::tuple<size_t, size_t, size_t>> parent_reqs;
    parent_reqs.reserve(2*local_size);
    // parent request where the character is the last `$`/`0` character
    // these don't have to be requested, but are locally fulfilled
    std::vector<std::tuple<size_t, size_t, size_t>> dollar_reqs;
    std::vector<std::tuple<size_t, size_t, size_t>> remote_reqs;

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
        if (edgechar_method == edgechar_twophase_all2all) {
            if (sa.local_SA[i] + lcp_val >= global_size) {
                MXX_ASSERT(sa.local_SA[i] + lcp_val == global_size);
                dollar_reqs.emplace_back(parent, global_size + prefix + i, 0);
            } else {
                parent_reqs.emplace_back(parent, global_size + prefix + i, sa.local_SA[i] + lcp_val);
            }
        } else {
            if (prefix <= parent && parent < prefix + local_size) {
                parent_reqs.emplace_back(parent, global_size + prefix + i, sa.local_SA[i] + lcp_val);
            } else {
                remote_reqs.emplace_back(parent, global_size + prefix + i, sa.local_SA[i] + lcp_val);
            }
        }
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
        if (edgechar_method == edgechar_twophase_all2all) {
            if (sa.local_SA[i] + lcp_val >= global_size) {
                MXX_ASSERT(sa.local_SA[i] + lcp_val == global_size);
                dollar_reqs.emplace_back(parent, prefix + i, 0);
            } else {
                parent_reqs.emplace_back(parent, prefix + i, sa.local_SA[i] + lcp_val);
            }
        } else {
            if (prefix <= parent && parent < prefix + local_size) {
                parent_reqs.emplace_back(parent, prefix + i, sa.local_SA[i] + lcp_val);
            } else {
                remote_reqs.emplace_back(parent, prefix + i, sa.local_SA[i] + lcp_val);
            }
        }
    }
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
    } else if (edgechar_method == edgechar_twophase_all2all) {
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
    } else if (edgechar_method == edgechar_mpi_osc_rma) {
        mxx::partition::block_decomposition_buffered<size_t> part(global_size, comm.size(), comm.rank());
        // send those edges for which the parent lies on a remote processor
        mxx::all2all_func(remote_reqs, [&part](const std::tuple<size_t,size_t,size_t>& t) {return part.target_processor(std::get<0>(t));}, comm);
        parent_reqs.insert(parent_reqs.end(), remote_reqs.begin(), remote_reqs.end());
        t.end_section("all2all_func: send to parent");

        // create MPI_Win for input string, create character array for size of parents
        // and use RMA to request (read) all characters which are not `$`
        MPI_Win win;
        MPI_Win_create(&(*sa.input_begin), local_size, 1, MPI_INFO_NULL, comm, &win);
        MPI_Win_fence(0, win);

        // read characters here!
        edge_chars.resize(parent_reqs.size());
        for (size_t i = 0; i < parent_reqs.size(); ++i) {
            size_t offset = std::get<2>(parent_reqs[i]);
            // read global index offset
            if (offset == global_size) {
                // TODO: handle specially?
                edge_chars[i] = 0;
            } else {
                int proc = part.target_processor(offset);
                size_t proc_offset = offset - part.excl_prefix_size(proc);
                // request proc_offset from processor `proc` in window win
                MPI_Get(&edge_chars[i], 1, MPI_CHAR, proc, proc_offset, 1, MPI_CHAR, win);
            }
        }
        // fence to complete all requests
        //MPI_Win_fence(MPI_MODE_NOSTORE | MPI_MODE_NOPUT | MPI_MODE_NOSUCCEED, win);
        MPI_Win_fence(0, win);
        MPI_Win_free(&win);

        t.end_section("RMA read chars");
    } else if (edgechar_method == edgechar_rma_shared) {
        // use MPI shared memory windows (only works on shared memory systems)
        mxx::partition::block_decomposition_buffered<size_t> part(global_size, comm.size(), comm.rank());
        // send those edges for which the parent lies on a remote processor
        mxx::all2all_func(remote_reqs, [&part](const std::tuple<size_t,size_t,size_t>& t) {return part.target_processor(std::get<0>(t));}, comm);
        parent_reqs.insert(parent_reqs.end(), remote_reqs.begin(), remote_reqs.end());
        t.end_section("all2all_func: send to parent");

        // create MPI_Win for input string, create character array for size of parents
        // and use RMA to request (read) all characters which are not `$`
        MPI_Win win;
        CharT* charwin;
        MPI_Win_allocate_shared (local_size, 1, MPI_INFO_NULL, comm, &charwin, &win);
        /* copy string into shared memory window */
        std::copy(sa.input_begin, sa.input_end, charwin);
        MPI_Win_sync(win);
        MPI_Win_fence(0, win);

        /* create table of pointers for shared memory access */
        std::vector<CharT*> shptrs(comm.size());
        for (int i = 0; i < comm.size(); ++i) {
            MPI_Aint winsize;
            int windispls;
            MPI_Win_shared_query(win, i, &winsize, &windispls, &shptrs[i]);
        }

        // read characters here!
        edge_chars.resize(parent_reqs.size());
        for (size_t i = 0; i < parent_reqs.size(); ++i) {
            size_t offset = std::get<2>(parent_reqs[i]);
            // read global index offset
            if (offset == global_size) {
                // TODO: handle specially?
                edge_chars[i] = 0;
            } else {
                int proc = part.target_processor(offset);
                size_t proc_offset = offset - part.excl_prefix_size(proc);
                // read at pos proc_offset from processor `proc` in window win
                edge_chars[i] = *(shptrs[proc] + proc_offset);
            }
        }
        // fence to complete all requests
        MPI_Win_sync(win);
        MPI_Win_fence(0, win);
        MPI_Win_free(&win);

        t.end_section("RMA shared read chars");
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
        if (edgechar_method == edgechar_twophase_all2all) {
            c = std::get<2>(parent_reqs[i]);
        } else {
            CharT x = edge_chars[i];
            if (x == 0) {
                c = 0;
            } else {
                c = sa.alphabet_mapping[x];
            }
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

    // This stuff is all done (look above)
    // TODO:
    // - get character for internal nodes (first character of suffix starting at
    //   the node S[SA[i]+LCP[i]]) [i.e. requesting a full permutation] needed for
    //   all edges, i.e. up to 2n-1 (SA and LCP aligned)
    //
    // TODO:
    // need character S[SA[i]+maxLCP[i]] for each edge (leaf + internal node)
    // maxLCP is equal to LCP[parent]
    // but `parent` might not be local
    // -> can also be retrieved at `parent` when sending edges
    // SA[i] is local, we can send that, along with `i` since edge is (parent,i)
    // so at least needs sending (parent, i, SA[i])
    // OR: (parent, i, c) if `c` is retrieved before sending edges to `parent`
    // clue: we are sending exactly as many edge elements as in ANSV
    // whereas requesting `c` is full all2all_msgs/queries/RMA?
    //


    // TODO: probably best to send a character S[SA[i]+maxLCP[i]] along for edge labeling
    // each internal node: at LCP index, LCP gives string depth already
    //                     children edges: either by fixed size sigma*n lookup table
    //                     or by hashtable (lcp-index, character)->child-index
    //                     The LCP indeces are predictable [n/p*rank,n/p*(rank+1))
    //                     -> be careful with hash function!

    return internal_nodes;
}


#endif // SUFFIX_TREE_HPP
