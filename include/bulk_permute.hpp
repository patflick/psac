#ifndef BULK_PERMUTE_HPP
#define BULK_PERMUTE_HPP

#include <mxx/comm.hpp>
#include <mxx/partition.hpp>

#include <assert.h>

/*
 * TODO: double check with mxx bucketing implemenetation
 */

template <typename T, typename index_t>
void bulk_permute_inplace(std::vector<T>& vec, std::vector<index_t>& idx, const mxx::blk_dist& part, const mxx::comm& comm) {
    assert(idx.size() == vec.size());

    //SAC_TIMER_START();
    // 1.) local bucketing for each processor
    //
    // counting the number of elements for each processor
    std::vector<size_t> send_counts(comm.size(), 0);
    for (index_t gi : idx) {
        int target_p = part.rank_of(gi);
        assert(0 <= target_p && target_p < comm.size());
        ++send_counts[target_p];
    }

    // get exclusive prefix sum
    std::vector<size_t> send_displs = mxx::local_exscan(send_counts);
    std::vector<size_t> upper_bound = mxx::local_scan(send_counts);

    // in-place bucketing
    int cur_p = 0;
    for (std::size_t i = 0; i < idx.size();) {
        // skip full buckets
        while (cur_p < comm.size()-1 && send_displs[cur_p] >= upper_bound[cur_p]) {
            // skip over full buckets
            i = send_displs[++cur_p];
        }
        // break if all buckets are done
        if (cur_p == comm.size()-1)
            break;
        int target_p = part.rank_of(idx[i]);
        assert(0 <= target_p && target_p < comm.size());
        if (target_p == cur_p) {
            // item correctly placed
            ++i;
        } else {
            // swap to correct bucket
            assert(target_p > cur_p);
            std::swap(idx[i], idx[send_displs[target_p]]);
            std::swap(vec[i], vec[send_displs[target_p]]);
        }
        send_displs[target_p]++;
    }

    //SAC_TIMER_END_SECTION("sa2isa_bucketing");

    // all2all communication for both vectors
    idx = mxx::all2allv(idx, send_counts, comm);
    std::vector<T> recv_vec = mxx::all2allv(vec, send_counts, comm);
    //SAC_TIMER_END_SECTION("sa2isa_all2all");

    // locally rearrange (assign to correct index)
    size_t prefix = part.eprefix_size();
    for (std::size_t i = 0; i < idx.size(); ++i) {
        index_t out_idx = idx[i] - prefix;
        assert(0 <= out_idx && out_idx < idx.size());
        vec[out_idx] = recv_vec[i];
    }

    //SAC_TIMER_END_SECTION("sa2isa_rearrange");
}


#endif // BULK_PERMUTE_HPP
