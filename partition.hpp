/**
 * @file    paritition.hpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements the partition API and different data partitions, most
 *          notably the block decomposition.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 *
 * TODO add Licence
 */

#ifndef PARTITION_HPP
#define PARTITION_HPP

#include <vector>
#include "assert.h"

// We want inlined non virtual functions for the partition. We thus need to use
// templating when these classes are used and no real inheritance to enforce
// the API.
namespace partition
{

template <typename index_t>
class block_decomposition
{
public:
    /// Constructor (no default construction)
    block_decomposition(index_t n, int p, int rank)
        : n(n), p(p), rank(rank)
    {
    }

    index_t local_size()
    {
        return n/p + ((static_cast<index_t>(rank) < (n % static_cast<index_t>(p))) ? 1 : 0);
    }

    index_t local_size(int rank)
    {
        return n/p + ((static_cast<index_t>(rank) < (n % static_cast<index_t>(p))) ? 1 : 0);
    }

    index_t prefix_size()
    {
        return (n/p)*(rank+1) + std::min<index_t>(n % p, rank + 1);
    }

    index_t prefix_size(int rank)
    {
        return (n/p)*(rank+1) + std::min<index_t>(n % p, rank + 1);
    }

    index_t excl_prefix_size()
    {
        return (n/p)*rank + std::min<index_t>(n % p, rank);
    }

    index_t excl_prefix_size(int rank)
    {
        return (n/p)*rank + std::min<index_t>(n % p, rank);
    }

    // which processor the element with the given global index belongs to
    int target_processor(index_t global_index)
    {
        if (global_index < ((n/p)+1)*(n % p))
        {
            // a_i is within the first n % p processors
            return global_index/((n/p)+1);
        }
        else
        {
            return n%p + (global_index - ((n/p)+1)*(n % p))/(n/p);
        }
    }

    /// Destructor
    virtual ~block_decomposition () {}
private:
    /* data */
    /// Number of elements
    const index_t n;
    /// Number of processors
    const int p;
    /// Processor rank
    const int rank;
};


template <typename index_t>
class block_decomposition_buffered
{
public:
    block_decomposition_buffered(index_t n, int p, int rank)
        : n(n), p(p), rank(rank), div(n / p), mod(n % p),
          loc_size(div + (static_cast<index_t>(rank) < mod ? 1 : 0)),
          prefix(div*rank + std::min<index_t>(mod, rank)),
          div1mod((div+1)*mod)
    {
    }

    index_t local_size()
    {
        return loc_size;
    }

    index_t local_size(int rank)
    {
        return div + (static_cast<index_t>(rank) < mod ? 1 : 0);
    }

    index_t prefix_size()
    {
        return prefix + loc_size;
    }

    index_t prefix_size(int rank)
    {
        return div*(rank+1) + std::min<index_t>(mod, rank + 1);
    }

    index_t excl_prefix_size()
    {
        return prefix;
    }

    index_t excl_prefix_size(int rank)
    {
        return div*rank + std::min<index_t>(mod, rank);
    }

    // which processor the element with the given global index belongs to
    int target_processor(index_t global_index)
    {
        // TODO: maybe also buffer (div+1)*mod, would save one multiplication
        // in each call to this
        if (global_index < div1mod)
        {
            // a_i is within the first n % p processors
            return global_index/(div+1);
        }
        else
        {
            return mod + (global_index - div1mod)/div;
        }
    }

    virtual ~block_decomposition_buffered () {}
private:
    /* data */
    /// Number of elements
    const index_t n;
    /// Number of processors
    const int p;
    /// Processor rank
    const int rank;

    // derived/buffered values (for faster computation of results)
    const index_t div; // = n/p
    const index_t mod; // = n%p
    // local size (number of local elements)
    const index_t loc_size;
    // the exclusive prefix (number of elements on previous processors)
    const index_t prefix;
    /// number of elements on processors with one more element
    const index_t div1mod; // = (n/p + 1)*(n % p)
};

} // namespace partition

/****************************************************
 *  Legacy code: (need to be removed step by step:  *
 ****************************************************/

/**
 * @brief Returns a block partitioning of an input of size `n` among `p` processors.
 *
 * @param n The number of elements.
 * @param p The number of processors.
 *
 * @return A vector of the number of elements for each processor.
 */
/*
std::vector<int> block_partition(int n, int p)
{
    // init result
    std::vector<int> partition(p);
    // get the number of elements per processor
    int local_size = n / p;
    // and the elements that are not evenly distributable
    int remaining = n % p;
    for (int i = 0; i < p; ++i) {
        if (i < remaining) {
            partition[i] = local_size + 1;
        } else {
            partition[i] = local_size;
        }
    }
    return partition;
}
*/

/*
inline int block_partition_local_size(int n, int p, int i)
{
    return n/p + ((i < (n % p)) ? 1 : 0);
}
inline std::size_t block_partition_local_size(std::size_t n, int p, int i)
{
    return n/p + ((static_cast<std::size_t>(i) < (n % static_cast<std::size_t>(p))) ? 1 : 0);
}
inline std::size_t block_partition_prefix_size(std::size_t n, int p, int i)
{
    return (n/p)*(i+1) + std::min<std::size_t>(n % p, i+1);
}

inline int block_partition_prefix_size(int n, int p, int i)
{
    return (n/p)*(i+1) + std::min(n % p, i+1);
}

inline std::size_t block_partition_excl_prefix_size(std::size_t n, int p, int i)
{
    return (n/p)*i + std::min<std::size_t>(n % p, i);
}

inline int block_partition_excl_prefix_size(int n, int p, int i)
{
    return (n/p)*i + std::min(n % p, i);
}

// returns the target processor id {0,..,p-1} for an element with index i
inline int block_partition_target_processor(std::size_t n, int p, std::size_t a_i)
{
    if (a_i < ((n/p)+1)*(n % p))
    {
        // a_i is within the first n % p processors
        return a_i/((n/p)+1);
    }
    else
    {
        return n%p + (a_i - ((n/p)+1)*(n % p))/(n/p);
    }
}

inline int block_partition_target_processor(int n, int p, int a_i)
{
    if (a_i < ((n/p)+1)*(n % p))
    {
        // a_i is within the first n % p processors
        return a_i/((n/p)+1);
    }
    else
    {
        return n%p + (a_i - ((n/p)+1)*(n % p))/(n/p);
    }
}
*/

#endif // PARTITION_HPP
