/**
 * @file    mpi_file.hpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Block decompose and distribute file as string on MPI communicator.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 *
 * TODO add Licence
 */

#ifndef MPI_FILE_HPP
#define MPI_FILE_HPP

#include <mpi.h>

#include <string>
#include <fstream>
#include <streambuf>
#include <sstream>
#include <iostream>

#include "partition.hpp"

std::ifstream::pos_type get_filesize(const char* filename)
{
    std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
    return in.tellg();
}

class rangebuf: public std::streambuf {
public:
    rangebuf(std::streampos start,
                    size_t size,
                    std::streambuf* sbuf):
        size_(size), sbuf_(sbuf), buf_(new char[64])
    {
        sbuf->pubseekpos(start, std::ios_base::in);
    }
    int underflow() {
        size_t r(this->sbuf_->sgetn(this->buf_,
            std::min<size_t>(sizeof(this->buf_), this->size_)));
        this->size_ -= r;
        this->setg(this->buf_, this->buf_, this->buf_ + r);
        return this->gptr() == this->egptr()
            ? traits_type::eof()
            : traits_type::to_int_type(*this->gptr());
    }

    ~rangebuf()
    {
        delete [] this->buf_;
    }
protected:
    size_t size_;
    std::streambuf* sbuf_;
    char* buf_;
};

std::string file_block_decompose(const char* filename, MPI_Comm comm = MPI_COMM_WORLD, std::size_t max_local_size = 0)
{
    // TODO: handle error if file doesn't exist

    // get size of input file
    std::size_t file_size = get_filesize(filename);

    // get communication parameters
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);
    partition::block_decomposition<std::size_t> part(file_size, p, rank);

    // restrict max local size (assuming that it is the same parameter on each
    // processor)
    if (max_local_size > 0 && file_size / p > max_local_size)
        file_size = p*max_local_size;
    // block decompose
    std::size_t local_size = part.local_size();
    std::size_t offset = part.excl_prefix_size();

    // open file
    std::ifstream t(filename);
    // wrap in our custom range buffer (of type std::streambuf)
    rangebuf rb(offset, local_size, t.rdbuf());

    // read file (range) buffer into string stream
    std::stringstream ss;
    ss << &rb;

    std::string local_str = ss.str();

    return local_str;
}


#endif // MPI_FILE_HPP
