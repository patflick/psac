#ifndef BULK_RMA_HPP
#define BULK_RMA_HPP

#include <mxx/partition.hpp>
#include <mxx/comm.hpp>
#include <mxx/timer.hpp>

// for posix sm
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>

template <typename Q, typename Func>
std::vector<typename std::result_of<Func(Q)>::type> bulk_query(const std::vector<Q>& queries, Func f, const std::vector<size_t>& send_counts, const mxx::comm& comm) {
    // type of the query results
    using T = typename std::result_of<Func(Q)>::type;
    mxx::section_timer t(std::cerr, comm);

    // get receive counts (needed as send counts for returning queries)
    std::vector<size_t> recv_counts = mxx::all2all(send_counts, comm);
    t.end_section("bulk_query: get recv_counts");

    // send all queries via all2all
    std::vector<Q> local_queries = mxx::all2allv(queries, send_counts, recv_counts, comm);
    t.end_section("bulk_query: all2all queries");

    // show load inbalance in queries and recv_counts
    size_t recv_num = local_queries.size();
    std::pair<size_t, int> maxel = mxx::max_element(recv_num, comm);
    size_t total_queries = mxx::allreduce(queries.size(), comm);
    std::vector<size_t> recv_per_proc = mxx::gather(recv_num, 0, comm);
    if (comm.rank() == 0) {
        std::cerr << "Avg queries: " << total_queries * 1.0 / comm.size() << ", max queries on proc " << maxel.second << ": " << maxel.first << std::endl;
        std::cerr << "Inbalance factor: " << maxel.first * comm.size() * 1.0 / total_queries << "x" << std::endl;
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

// buckets inplace but keeps track of original index for each element
// returns the send_counts (number of elements in each bucket)
template <typename T, typename Func>
std::vector<size_t> idxbucketing(const std::vector<T>& vec, Func key_func, size_t num_buckets, std::vector<T>& bucketed_vec, std::vector<size_t>& original_pos) {
    std::vector<size_t> send_counts(num_buckets, 0);
    for (size_t i = 0; i < vec.size(); ++i) {
        ++send_counts[key_func(vec[i])];
    }
    std::vector<size_t> offsets = mxx::local_exscan(send_counts);

    // [2nd pass]: saving elements into correct position
    bucketed_vec.resize(vec.size());
    original_pos.resize(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        size_t pos = offsets[key_func(vec[i])]++;
        bucketed_vec[pos] = vec[i];
        original_pos[pos] = i;
    }
    return send_counts;
}

template <typename T, typename Func>
std::vector<size_t> idxbucketing_inplace(std::vector<T>& vec, Func key_func, size_t num_buckets, std::vector<size_t>& original_pos) {
    // TODO: inplace optimizations?
    std::vector<T> tmp_res;
    std::vector<size_t> send_counts = idxbucketing(vec, key_func, num_buckets, tmp_res, original_pos);
    vec.swap(tmp_res);
    return send_counts;
}

template <typename T>
std::vector<T> permute(const std::vector<T>& vec, const std::vector<size_t>& idx) {
    assert(vec.size() == idx.size());
    std::vector<T> results(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        // TODO: streaming write
        results[idx[i]] = vec[i];
    }
    return results;
}


template <typename InputIter>
std::vector<typename std::iterator_traits<InputIter>::value_type>
bulk_rma(InputIter local_begin, InputIter local_end,
         const std::vector<size_t>& global_indexes, const mxx::comm& comm) {

    using value_type = typename std::iterator_traits<InputIter>::value_type;
    // get local and global size
    size_t local_size = std::distance(local_begin, local_end);
    size_t global_size = mxx::allreduce(local_size, comm);
    // get the block decomposition class and check that input is actuall block
    // decomposed
    mxx::blk_dist part(global_size, comm.size(), comm.rank());
    MXX_ASSERT(part.local_size() == local_size);


    std::vector<size_t> bucketed_indexes;
    std::vector<size_t> original_pos;
    std::vector<size_t> send_counts = idxbucketing(global_indexes, [&part](size_t gidx) { return part.rank_of(gidx); }, comm.size(), bucketed_indexes, original_pos);

    std::vector<value_type> results = bulk_rma(local_begin, local_end, bucketed_indexes, send_counts, comm);
    bucketed_indexes = std::vector<size_t>();
    // reorder back into original order
    return permute(results, original_pos);
}


template <typename InputIter>
std::vector<typename std::iterator_traits<InputIter>::value_type>
bulk_rma_mpiwin(InputIter local_begin, InputIter local_end,
         const std::vector<size_t>& global_indexes, const mxx::comm& comm) {
    using value_type = typename std::iterator_traits<InputIter>::value_type;

    // get local and global size
    size_t local_size = std::distance(local_begin, local_end);
    size_t global_size = mxx::allreduce(local_size, comm);
    mxx::blk_dist part(global_size, comm.size(), comm.rank());

    // create MPI_Win for input string, create character array for size of parents
    // and use RMA to request (read) all characters which are not `$`
    MPI_Win win;
    MPI_Win_create(&(*local_begin), local_size, sizeof(value_type), MPI_INFO_NULL, comm, &win);
    MPI_Win_fence(0, win);

    mxx::datatype dt = mxx::get_datatype<value_type>();

    // read characters here!
    std::vector<value_type> results(global_indexes.size());
    for (size_t i = 0; i < results.size(); ++i) {
        size_t offset = global_indexes[i];
        int proc = part.rank_of(offset);
        size_t proc_offset = offset - part.eprefix_size(proc);
        // request proc_offset from processor `proc` in window win
        MPI_Get(&results[i], 1, dt.type(), proc, proc_offset, 1, dt.type(), win);
    }
    // fence to complete all requests
    //MPI_Win_fence(MPI_MODE_NOSTORE | MPI_MODE_NOPUT | MPI_MODE_NOSUCCEED, win);
    MPI_Win_fence(0, win);
    MPI_Win_free(&win);

    return results;
}

#if MPI_VERSION > 2
// TODO: separate further into Window and the global indexing stuff
//       ie: seprate into: global_array and backend implementation 
template <typename T>
class shmem_window_mpi {
// TODO: visablitiy
public:
    typedef T value_type;
    size_t global_size;
    size_t local_size;
    size_t prefix;
    const mxx::comm& comm;

    // private:
    MPI_Win win;
    value_type* charwin;
    value_type* shptr;

    template <typename Iterator>
    void init(Iterator local_begin, Iterator local_end) {
        mxx::section_timer t(std::cerr, comm);
        // get local and global size
        local_size = std::distance(local_begin, local_end);
        prefix = mxx::exscan(local_size, comm);
        global_size = mxx::allreduce(local_size, comm);
        mxx::blk_dist part(global_size, comm.size(), comm.rank());

        mxx::datatype dt = mxx::get_datatype<value_type>();

        // create MPI_Win for input string, create character array for size of parents
        // and use RMA to request (read) all characters which are not `$`
        MPI_Info info;
        MPI_Info_create(&info);
        MPI_Info_set(info, "alloc_shared_noncontig", "true");
        if (comm.rank() == 0) {
            MPI_Win_allocate_shared(global_size, sizeof(value_type), info, comm, &charwin, &win);
        } else {
            MPI_Win_allocate_shared(0, sizeof(value_type), info, comm, &charwin, &win);
        }
        t.end_section("alloc window");

        /* create table of pointers for shared memory access */
        //std::vector<CharT*> shptrs(comm.size());
        /*
           for (int i = 0; i < comm.size(); ++i) {
           MPI_Aint winsize;
           int windispls;
           MPI_Win_shared_query(win, i, &winsize, &windispls, &shptrs[i]);
           }
           */

        MPI_Aint winsize; int windispls;
        MPI_Win_shared_query(win, 0, &winsize, &windispls, &shptr);
        t.end_section("get ptrs via shared_query");

        /* copy string into shared memory window */
        memcpy(shptr+prefix, &(*local_begin), local_size*sizeof(value_type));
        t.end_section("copy input string into shared win");
        MPI_Win_sync(win);
        MPI_Win_fence(0, win);
        t.end_section("sync");
    }

    template <typename Iterator>
    shmem_window_mpi(Iterator local_begin, Iterator local_end, const mxx::comm& c) : comm(c) {
        init(local_begin, local_end);
    }

    inline T get(size_t gidx) const {
        return *(shptr + gidx);
    }

    virtual ~shmem_window_mpi() {
        MPI_Win_free(&win);
    }
};

#endif

template <typename T>
class shmem_window_posix {
// TODO: visablitiy
public:
    typedef T value_type;
    size_t global_size;
    size_t local_size;
    const mxx::comm& comm;

    // private:
    value_type* shptr;
    int sm_fd;

    template <typename Iterator>
    void init(Iterator local_begin, Iterator local_end) {
        mxx::section_timer t(std::cerr, comm);

        // get local and global size
        local_size = std::distance(local_begin, local_end);
        global_size = mxx::allreduce(local_size, comm);

        // create MPI_Win for input string, create character array for size of parents
        // and use RMA to request (read) all characters which are not `$`
        if (comm.rank() == 0) {
            sm_fd = shm_open("/my_shmem", O_CREAT | O_RDWR, 438);
            if (sm_fd == -1) {
                std::cerr << "couldn't open sm file" << std::endl;
                exit(EXIT_FAILURE);
            }
            if (ftruncate(sm_fd, sizeof(value_type)*global_size) == -1) {
                std::cerr << "couldn't truncate sm file" << std::endl;
                exit(EXIT_FAILURE);
            }
            shptr = (value_type*) mmap(NULL, sizeof(value_type)*global_size, PROT_READ | PROT_WRITE, MAP_SHARED, sm_fd, 0);
        }
        t.end_section("shm_open+alloc");

        // gather string into shmem page
        std::vector<size_t> recv_sizes = mxx::gather(local_size, 0, comm);
        mxx::gatherv(&(*local_begin), local_size, shptr, recv_sizes, 0, comm);
        t.end_section("gather string to master");

        // open shared memory pages for the string
        if (comm.rank() != 0) {
            sm_fd = shm_open("/my_shmem", O_RDONLY, 438);
            if (sm_fd == -1) {
                std::cerr << "couldn't open sm file on slave process" << std::endl;
                exit(EXIT_FAILURE);
            }
            shptr = (value_type*) mmap(NULL, sizeof(value_type)*global_size, PROT_READ, MAP_SHARED, sm_fd, 0);
        }
        comm.barrier();
        t.end_section("open shmem on rank != 0");
    }

    template <typename Iterator>
    shmem_window_posix(Iterator local_begin, Iterator local_end, const mxx::comm& c) : comm(c) {
        init(local_begin, local_end);
    }

    inline T get(size_t gidx) const {
        return *(shptr + gidx);
    }

    virtual ~shmem_window_posix() {
        // clean up shmem
        //munmap(shptr, sizeof(CharT)*global_size);
        if (comm.rank() == 0)
            shm_unlink("/my_shmem");
    }
};

template <typename T>
class shmem_window_posix_split {
// TODO: visablitiy
public:
    typedef T value_type;
    size_t global_size;
    size_t local_size;
    const mxx::comm& comm;

    // shmem files and mem-mapped ptrs
    std::vector<value_type*> shptrs;
    std::vector<size_t> group_data_sizes;
    std::vector<int> sm_fds;

    // group sizes (TODO: move into a separate hier-communicator object)
    mxx::comm subcomm;
    int num_groups;
    int group_size;
    int group_idx;

    template <typename Iterator>
    void init(Iterator local_begin, Iterator local_end) {
        mxx::section_timer t(std::cerr, comm);

        // get local and global size
        local_size = std::distance(local_begin, local_end);
        global_size = mxx::allreduce(local_size, comm);

        /* split communicator into groups */

        num_groups = 4; // split communicator into 4 subgroups
        if (num_groups > comm.size()) {
            num_groups = 1;
        }
        group_size = comm.size() / num_groups;
        // number of group
        group_idx = comm.rank() / group_size;
        // create subcommunicator (TODO: use hierarchical communicator/or 2D grid comm)
        subcomm = comm.split(group_idx);
        MXX_ASSERT(subcomm.rank() == comm.rank() % group_size);


        /* get data size for each group */
        shptrs.resize(num_groups);

        size_t group_data_size = mxx::allreduce(local_size, subcomm);
        // allgather group sizes
        mxx::comm span_comm = comm.split(subcomm.rank() == 0);
        if (subcomm.rank() == 0) {
            group_data_sizes = mxx::allgather(group_data_size, span_comm);
        }
        mxx::bcast(group_data_sizes, 0, subcomm);
        t.end_section("create groups");


        /* open shared memory on each group master */

        sm_fds.resize(num_groups);
        if (subcomm.rank() == 0) {
            char sh_name[20];
            sprintf(sh_name, "/my_shmem_%d", group_idx);

            sm_fds[group_idx] = shm_open(sh_name, O_CREAT | O_RDWR, 438);
            if (sm_fds[group_idx] == -1) {
                std::cerr << "couldn't open sm file" << std::endl;
                exit(EXIT_FAILURE);
            }
            if (ftruncate(sm_fds[group_idx], sizeof(value_type)*group_data_size) == -1) {
                std::cerr << "couldn't truncate sm file" << std::endl;
                exit(EXIT_FAILURE);
            }
            shptrs[group_idx] = (value_type*) mmap(NULL, sizeof(value_type)*group_data_size, PROT_READ | PROT_WRITE, MAP_SHARED, sm_fds[group_idx], 0);
        }
        t.end_section("shm_open+alloc");

        // gather string into shmem page
        std::vector<size_t> recv_sizes = mxx::gather(local_size, 0, subcomm);
        mxx::gatherv(&(*local_begin), local_size, shptrs[group_idx], recv_sizes, 0, subcomm);
        t.end_section("gather+convert string to group master");

        if (subcomm.rank() == 0) {
            char sh_name[20];
            sprintf(sh_name, "/my_shmem_%d", group_idx);
            // reopen shared mem in readonly mode
            munmap(shptrs[group_idx], sizeof(value_type)*group_data_size);
            close(sm_fds[group_idx]);
        }
        t.end_section("shm close on group-master");

        // open shared memory pages for the string
        for (int i = 0; i < num_groups; ++i) {
            char sh_name[20];
            sprintf(sh_name, "/my_shmem_%d", i);
            sm_fds[i] = shm_open(sh_name, O_RDONLY, 438);
            if (sm_fds[i] == -1) {
                std::cerr << "couldn't open sm file on slave process" << std::endl;
                exit(EXIT_FAILURE);
            }
            shptrs[i] = (value_type*) mmap(NULL, sizeof(value_type)*group_data_sizes[i], PROT_READ, MAP_SHARED, sm_fds[i], 0);
        }
        comm.barrier();
        t.end_section("open shmem on group masters");
    }

    template <typename Iterator>
    shmem_window_posix_split(Iterator local_begin, Iterator local_end, const mxx::comm& c) : comm(c) {
        init(local_begin, local_end);
    }

    inline T get(size_t gidx) const {
        for (int g = 0; g < num_groups; ++g) {
            if (gidx < group_data_sizes[g]) {
                return *(shptrs[g] + gidx);
            } else {
                gidx -= group_data_sizes[g];
            }
        }
        assert(false);
        return T();
    }

    virtual ~shmem_window_posix_split() {
        // clean up shmem
        //munmap(shptr, sizeof(CharT)*global_size);
        if (subcomm.rank() == 0) {
            char sh_name[20];
            sprintf(sh_name, "/my_shmem_%d", group_idx);
            shm_unlink(sh_name);
        }
    }
};

#if MPI_VERSION > 2
template <typename InputIter>
std::vector<typename std::iterator_traits<InputIter>::value_type>
bulk_rma_shm_mpi(InputIter local_begin, InputIter local_end,
         const std::vector<size_t>& global_indexes, const mxx::comm& comm) {
    mxx::section_timer t(std::cerr, comm);
    using value_type = typename std::iterator_traits<InputIter>::value_type;

    shmem_window_mpi<value_type> win(local_begin, local_end, comm);

    // read characters here!
    std::vector<value_type> results(global_indexes.size());
    for (size_t i = 0; i < results.size(); ++i) {
        size_t offset = global_indexes[i];
        results[i] = win.get(offset);
    }
    t.end_section("get all characters");

    comm.barrier();

    return results;
}
#endif

template <typename InputIter>
std::vector<typename std::iterator_traits<InputIter>::value_type>
bulk_rma_shm_posix(InputIter local_begin, InputIter local_end,
         const std::vector<size_t>& global_indexes, const mxx::comm& comm) {
    mxx::section_timer t(std::cerr, comm);
    using value_type = typename std::iterator_traits<InputIter>::value_type;

    shmem_window_posix<value_type> win(local_begin, local_end, comm);

    // read characters here!
    std::vector<value_type> results(global_indexes.size());
    for (size_t i = 0; i < results.size(); ++i) {
        // read global index offset
        results[i] = win.get(global_indexes[i]);
    }
    t.end_section("get all characters");

    comm.barrier();
    t.end_section(" barrier");

    return results;
}

// TODO: wrap the shared memory window thing into a class with deref opertor etc

template <typename InputIter>
std::vector<typename std::iterator_traits<InputIter>::value_type>
bulk_rma_shm_posix_split(InputIter local_begin, InputIter local_end,
         const std::vector<size_t>& global_indexes, const mxx::comm& comm) {
    mxx::section_timer t(std::cerr, comm);
    using value_type = typename std::iterator_traits<InputIter>::value_type;

    shmem_window_posix_split<value_type> win(local_begin, local_end, comm);

    // read characters here!
    std::vector<value_type> results(global_indexes.size());
    for (size_t i = 0; i < results.size(); ++i) {
        // read global index offset
        results[i] = win.get(global_indexes[i]);
    }
    t.end_section("get all characters");
    comm.barrier();
    t.end_section(" barrier");

    return results;
}


#endif // BULK_RMA_HPP
