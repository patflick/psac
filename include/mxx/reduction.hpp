/**
 * @file    reduction.hpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Reduction operations.
 *
 *
 * TODO add Licence
 */


#ifndef MXX_REDUCTION_HPP
#define MXX_REDUCTION_HPP

#include <mpi.h>

#include <vector>
#include <iterator>
#include <limits>
#include <functional>

// mxx includes
#include "datatypes.hpp"

namespace mxx {

/*********************************************************************
 *                     User supplied functions:                      *
 *********************************************************************/


// this wrapper for user functions is not thread-safe!
// TODO: use another approach in order to make this thread safe!
// TODO: count template intializations, and make each of these static classes
//       templated by a counter. thus each user function used in the code
//       will get its own compile time template instantiation of this static
//       class
template <typename T>
struct user_func_wrapper
{
    static bool active;
    static std::function<T (T& x, T& y)> func;
    static MPI_Datatype dt;
    static void user_function(void *invec, void *inoutvec, int *len, MPI_Datatype *)
    {
        if (!active) {
          throw std::runtime_error("calling inactive user function");
        }
        // check in and out type
        // TODO: check extend of all types!

        T* in = (T*) invec;
        T* inout = (T*) inoutvec;
        for (int i = 0; i < *len; ++i)
        {
            inout[i] = func(in[i], inout[i]);
        }
    }
};

template <typename T>
std::function<T (T& x, T& y)> user_func_wrapper<T>::func;
template <typename T>
MPI_Datatype user_func_wrapper<T>::dt;
template <typename T>
bool user_func_wrapper<T>::active;

// internal functions for creating the MPI_Op for user supplied operators
template <typename T, typename Func, bool IsCommutative = true>
MPI_Op create_user_op(Func& func) {
    // get type
    mxx::datatype<T> dt;
    // define C++ wrapper for function pointers
    typedef user_func_wrapper<T> wrapper;
    if (wrapper::active) {
      throw std::runtime_error("type wrapper still under use");
    }
    wrapper::active = true;
    wrapper::func = std::move(func);
    wrapper::dt = dt.type();

    // create MPI user defined operation
    MPI_Op op;
    int commute = IsCommutative ? 1 : 0;
    MPI_Op_create(&wrapper::user_function, commute, &op);
    return op;
}

// internal function to free user supplied ops
template <typename T>
void free_user_op(MPI_Op op)
{
    typedef user_func_wrapper<T> wrapper;
    if (!wrapper::active) {
      throw std::runtime_error("type wrapper not active");
    }
    wrapper::active = false;
    // cleanup
    MPI_Op_free(&op);
}

/*********************************************************************
 *                Reductions                                         *
 *********************************************************************/
// TODO: add more (vectorized, different reduce ops, etc)
// TODO: naming of functions !?
// TODO: template specialize for std::min, std::max, std::plus, std::multiply
//       etc for integers to use MPI builtin ops


template <typename T>
T allreduce(T& x, MPI_Comm comm = MPI_COMM_WORLD) {
    // get type
    mxx::datatype<T> dt;
    T result;
    MPI_Allreduce(&x, &result, 1, dt.type(), MPI_SUM, comm);
    return result;
}

template <typename T>
T reduce(T& x, MPI_Comm comm = MPI_COMM_WORLD, int root = 0)
{
    // get type
    mxx::datatype<T> dt;
    T result;
    MPI_Reduce(&x, &result, 1, dt.type(), MPI_SUM, root, comm);
    return result;
}

template <typename T, typename Func>
T allreduce(T& x, Func func, MPI_Comm comm = MPI_COMM_WORLD) {
    // get user op
    MPI_Op op = create_user_op<T, Func>(func);
    // get type
    mxx::datatype<T> dt;
    // perform reduction
    T result;
    MPI_Allreduce(&x, &result, 1, dt.type(), op, comm);
    // clean up op
    free_user_op<T>(op);
    // return result
    return result;
}

template <typename T>
T exscan(T& x, MPI_Comm comm = MPI_COMM_WORLD) {
    // get type
    mxx::datatype<T> dt;
    T result;
    MPI_Exscan(&x, &result, 1, dt.type(), MPI_SUM, comm);
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0)
      result = T();
    return result;
}

template <typename T, typename Func>
T exscan(T& x, Func func, MPI_Comm comm = MPI_COMM_WORLD) {
    // get user op
    MPI_Op op = create_user_op<T, Func>(func);
    // get type
    mxx::datatype<T> dt;
    // perform reduction
    T result;
    MPI_Exscan(&x, &result, 1, dt.type(), op, comm);
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0)
      result = T();
    // clean up op
    free_user_op<T>(op);
    return result;
}

template <typename T>
T scan(T& x, MPI_Comm comm = MPI_COMM_WORLD) {
    // get type
    mxx::datatype<T> dt;
    T result;
    MPI_Scan(&x, &result, 1, dt.type(), MPI_SUM, comm);
    return result;
}

template <typename T, typename Func>
T scan(T& x, Func func, MPI_Comm comm = MPI_COMM_WORLD) {
    // get user op
    MPI_Op op = create_user_op<T, Func>(func);
    // get type
    mxx::datatype<T> dt;
    T result;
    MPI_Scan(&x, &result, 1, dt.type(), op, comm);
    // clean up op
    free_user_op<T>(op);
    return result;
}

/****************************************************
 *  reverse reductions (with reverse communicator)  *
 ****************************************************/

void rev_comm(MPI_Comm comm, MPI_Comm& rev)
{
    // get MPI parameters
    int rank;
    int p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);
    MPI_Comm_split(comm, 0, p - rank, &rev);
}

template <typename T>
T reverse_exscan(T& x, MPI_Comm comm = MPI_COMM_WORLD) {
    MPI_Comm rev;
    rev_comm(comm, rev);
    T result = exscan(x, rev);
    MPI_Comm_free(&rev);
    return result;
}

template <typename T, typename Func>
T reverse_exscan(T& x, Func func, MPI_Comm comm = MPI_COMM_WORLD) {
    MPI_Comm rev;
    rev_comm(comm, rev);
    T result = exscan(x, func, rev);
    MPI_Comm_free(&rev);
    return result;
}

template <typename T>
T reverse_scan(T& x, MPI_Comm comm = MPI_COMM_WORLD) {
    MPI_Comm rev;
    rev_comm(comm, rev);
    T result = scan(x, rev);
    MPI_Comm_free(&rev);
    return result;
}

template <typename T, typename Func>
T reverse_scan(T& x, Func func, MPI_Comm comm = MPI_COMM_WORLD) {
    MPI_Comm rev;
    rev_comm(comm, rev);
    T result = scan(x, func, rev);
    MPI_Comm_free(&rev);
    return result;
}


/*********************
 *  Specialized ops  *
 *********************/

/************************
 *  Boolean reductions  *
 ************************/
// useful for testing global conditions, such as termination conditions

template<int dummy = 0>
bool test_all(bool x, MPI_Comm comm = MPI_COMM_WORLD) {
    int i = x ? 1 : 0;
    int result;
    MPI_Allreduce(&i, &result, 1, MPI_INT, MPI_LAND, comm);
    return result != 0;
}

template<int dummy = 0>
bool test_any(bool x, MPI_Comm comm = MPI_COMM_WORLD) {
    int i = x ? 1 : 0;
    int result;
    MPI_Allreduce(&i, &result, 1, MPI_INT, MPI_LOR, comm);
    return result != 0;
}

template<int dummy = 0>
bool test_none(bool x, MPI_Comm comm = MPI_COMM_WORLD) {
    int i = x ? 1 : 0;
    int result;
    MPI_Allreduce(&i, &result, 1, MPI_INT, MPI_LAND, comm);
    return result == 0;
}

} // namespace mxx

#endif // MXX_REDUCTION_HPP
