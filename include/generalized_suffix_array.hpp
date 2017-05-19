#ifndef GEN_SUFFIX_ARRAY_HPP
#define GEN_SUFFIX_ARRAY_HPP

#include <mpi.h>
#include <vector>
#include <cstring> // memcmp

#include "alphabet.hpp"
#include "kmer.hpp"
#include "par_rmq.hpp"
#include "shifting.hpp"

#include <mxx/comm.hpp>
#include <mxx/datatypes.hpp>
#include <mxx/shift.hpp>
#include <mxx/partition.hpp>
#include <mxx/sort.hpp>

#include <prettyprint.hpp>


template <typename Iterator>
struct string_set {
    Iterator lbegin;
    Iterator lend;


};


struct generalized_suffix_array {

    using alphabet_type = alphabet<char>;
    using index_t = uint64_t;

template <typename StringSet>
void construct_ss(const StringSet& ss, const mxx::comm& comm) {
    /***********************
     *  Initial bucketing  *
     ***********************/

    // detect alphabet and get encoding
    alphabet_type alpha = alphabet_type::from_string("abcd", comm);
    unsigned int bits_per_char = alpha.bits_per_char();
    unsigned int k = get_optimal_k(alpha);
    if(comm.rank() == 0) {
        INFO("Alphabet: " << alpha.unique_chars());
        INFO("Detecting sigma=" << alpha.sigma() << " => l=" << bits_per_char << ", k=" << k);
    }

    // create initial k-mers and use these as the initial bucket numbers
    // for each character position
    std::vector<index_t> local_B = kmer_generation<index_t>(ss, k, alpha, comm);

    size_t shift_by;
    for (shift_by = k; shift_by < n; shift_by <<= 1) {
        // 1) doubling by shifting into tuples (2BSA kind of structure)
        std::vector<index_t> B2 = shift_buckets_ss(ss);
        // 2) sort by (B1, B2)
        // 2) LCP
        // 3) rebucket (B1, B2) -> B1
        // 4) reverse order to SA order
    }
}
};

#endif // GEN_SUFFIX_ARRAY_HPP
