/**
 * @file    mpi_types.hpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   MPI Datatypes for C++ types.
 *
 * Copyright (c) TODO
 *
 * TODO add Licence
 */

#ifndef HPC_MPI_TYPES
#define HPC_MPI_TYPES

// MPI include
#include <mpi.h>

// C++ includes
#include <vector>
#include <numeric>


namespace mpi
{

/*
 * Mapping of C/C++ types to MPI datatypes.
 *
 * Possible ways of implementation
 * 1) templated function get_mpi_dt<template T>();
 * 2) overloaded functions with type deduction
 * 3) templated class with static method (-> allows partial specialization)
 * 4) templated class with member method (allows proper Type_free)
 */

template <typename T>
class datatype {};

#define DATATYPE_CLASS_MPI_BUILTIN(ctype, mpi_type)                         \
template <> class datatype<ctype> {                                         \
public:                                                                     \
    datatype() {}                                                           \
    MPI_Datatype type() const {return mpi_type;}                            \
    virtual ~datatype() {}                                                  \
};

// char
DATATYPE_CLASS_MPI_BUILTIN(char, MPI_CHAR);
DATATYPE_CLASS_MPI_BUILTIN(unsigned char, MPI_UNSIGNED_CHAR);
DATATYPE_CLASS_MPI_BUILTIN(signed char, MPI_SIGNED_CHAR);

// short
DATATYPE_CLASS_MPI_BUILTIN(unsigned short, MPI_UNSIGNED_SHORT);
DATATYPE_CLASS_MPI_BUILTIN(signed short, MPI_SHORT);

// int
DATATYPE_CLASS_MPI_BUILTIN(unsigned int, MPI_UNSIGNED);
DATATYPE_CLASS_MPI_BUILTIN(int, MPI_INT);

// long
DATATYPE_CLASS_MPI_BUILTIN(unsigned long, MPI_UNSIGNED_LONG);
DATATYPE_CLASS_MPI_BUILTIN(long, MPI_LONG);

// long long
DATATYPE_CLASS_MPI_BUILTIN(unsigned long long, MPI_UNSIGNED_LONG_LONG);
DATATYPE_CLASS_MPI_BUILTIN(long long, MPI_LONG_LONG);

// floats
DATATYPE_CLASS_MPI_BUILTIN(float, MPI_FLOAT);
DATATYPE_CLASS_MPI_BUILTIN(double, MPI_DOUBLE);
DATATYPE_CLASS_MPI_BUILTIN(long double, MPI_LONG_DOUBLE);

#undef DATATYPE_CLASS_MPI_BUILTIN

/**
 * @brief       MPI datatype mapping for std::array
 */
template <typename T, std::size_t size>
class datatype<std::array<T, size> > {
public:
    datatype() : _base_type() {
        MPI_Type_contiguous(size, _base_type.type(), &_type);
        MPI_Type_commit(&_type);
    }
    MPI_Datatype& type() const {
        return _type;
    }
    virtual ~datatype() {
        MPI_Type_free(&_type);
    }
private:
    MPI_Datatype _type;
    datatype<T> _base_type;
};

/**
 * @brief   MPI datatype mapping for std::pair
 */
template <typename T1, typename T2>
class datatype<std::pair<T1, T2> > {
public:
    datatype() : _base_type1(), _base_type2() {
        int blocklen[2] = {1, 1};
        MPI_Aint displs[2] = {0,0};
        // get actual displacement (in case of padding in the structure)
        std::pair<T1, T2> p;
        MPI_Aint p_adr, t1_adr, t2_adr;
        MPI_Get_address(&p, &p_adr);
        MPI_Get_address(&p.first(), &t1_adr);
        MPI_Get_address(&p.second(), &t2_adr);
        displs[0] = t1_adr - p_adr;
        displs[1] = t2_adr - p_adr;

        // create type
        MPI_Datatype types[2] = {_base_type1.type(), _base_type2.type()};
        MPI_Type_create_struct(2, blocklen, displs, types, &_type);
        MPI_Type_commit(&_type);
    }
    MPI_Datatype& type() const {
        return _type;
    }
    virtual ~datatype() {
        MPI_Type_free(&_type);
    }
private:
    MPI_Datatype _type;
    datatype<T1> _base_type1;
    datatype<T2> _base_type2;
};


template <std::size_t I, std::size_t N> struct fill_displacements;

template <class ...Types>
struct tuple_displacements
{
  typedef std::tuple<Types...> tuple_t;
  static constexpr std::size_t size = std::tuple_size<tuple_t>::value;
  public:
  std::array<MPI_Aint, size> displs;
  tuple_t tuple;

  void fill() {
    fill_displacements<size, size>::fill(*this);
  }


};

template <std::size_t N, std::size_t I>
struct fill_displacements
{
    template<class ...Types>
    static void fill(std::array<MPI_Aint, N>& displs, std::tuple<Types...>& tuple)
    {
        MPI_Aint t_adr, elem_adr;
        MPI_Get_address(&tuple, &t_adr);
        MPI_Get_address(&std::get<N-I>(tuple), &elem_adr);
        // byte offset from beginning of tuple
        displs[N-I] = elem_adr - t_adr;

        // TODO remove output
        std::cout << "displs " << N-I << " " << displs[N-I] << std::endl;
        // recursively (during compile time) call same function
        fill_displacements<N,I-1>::fill(displs, tuple);
    }
};

// Base case of meta-recursion
template <std::size_t N>
struct fill_displacements<N, 0>
{
    template<class ...Types>
    static void fill(std::array<MPI_Aint, N>&, std::tuple<Types...>&){
    }
};

// fill in MPI types
template <std::size_t N, std::size_t I>
struct fill_mpi_types
{
    template<class ...Types>
    static void fill(std::tuple<Types...>& datatypes,
                     std::array<MPI_Datatype, N>& mpi_datatypes)
    {
        // fill in type
        mpi_datatypes[N-I] = std::get<N-I>(datatypes).type();
        // recursively (during compile time) call same function
        fill_mpi_types<N,I-1>::fill(datatypes, mpi_datatypes);

    }
};

// Base case of meta-recursion
template <std::size_t N>
struct fill_mpi_types<N, 0>
{
    template<class ...Types>
    static void fill(std::tuple<Types...>&, std::array<MPI_Datatype, N>&) {
    }
};

/**
 * @brief   MPI datatype mapping for std::tuple
 */
template <class ...Types>
class datatype<std::tuple<Types...> > {
private:
  typedef std::tuple<Types...> tuple_t;
  typedef std::tuple<datatype<Types>...> datatypes_tuple_t;
  static constexpr std::size_t size = std::tuple_size<tuple_t>::value;
public:
    datatype() : _base_types() {
        int blocklen[size];
        // fill in the data for the tuple using meta-recursion
        std::array<MPI_Aint, size> displs;
        tuple_t t;
        fill_displacements<size, size>::fill(displs, t);
        std::array<MPI_Datatype, size> types;
        fill_mpi_types<size,size>::fill(_base_types, types);
        for (std::size_t i = 0; i < size; ++i)
        {
            blocklen[i] = 1;
        }

        // create type
        MPI_Type_create_struct(size, blocklen, &displs[0], &types[0], &_type);
        MPI_Type_commit(&_type);
    }

    const MPI_Datatype& type() const {
        return _type;
    }

    MPI_Datatype type() {
        return _type;
    }

    virtual ~datatype() {
        MPI_Type_free(&_type);
    }
private:
    MPI_Datatype _type;
    datatypes_tuple_t _base_types;
};

} // namespace mpi



#endif // HPC_MPI_TYPES
