Parallel Suffix Array and LCP Construction
==========================================
[![Build Status](https://img.shields.io/travis/patflick/psac.svg)](https://travis-ci.org/patflick/psac)
[![Build Status](https://travis-ci.org/patflick/psac.svg?branch=master)](https://travis-ci.org/patflick/psac)
[![Test Coverage](https://img.shields.io/codecov/c/github/patflick/psac.svg)](http://codecov.io/github/patflick/psac?branch=master)
[![Apache 2.0 License](https://img.shields.io/badge/license-Apache%20v2.0-blue.svg)](LICENSE)

This library implements a distributed-memory parallel algorithm for the
construction of suffix and LCP arrays. The algorithm is implemented in `C++11`
and `MPI`.

## Authors

- Patrick Flick

## Code organization

- [`include/`](include/) contains the implementation of our algorithms in form
  of C++ template header files (a header-only library).
- [`src/`](src/) contains the sources for binaries, which make use of the
  implementations in [`include/`](include/).
- [`test`](test/) contains unit tests for the components of the library.
- [`ext/`](ext/) contains external, third-party dependencies/libraries. See the
  [README](ext/README.md) for details on the third-party libraries used.


### Dependencies

- `cmake` version >= 2.6
- C++11 compatible compiler (tested with gcc and clang)
- an `MPI` implementation supporting `MPI-2` or `MPI-3`.
- external (third-party) dependencies are included in the [`ext/`](`ext/`) directory

### Compiling

To compile the executables and tests via cmake run the following:

```sh
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=Release ../
make
```

## Running

After compiling, there will a multiple binaries available in the `build/bin`
folder. Running `--help` on them will give more detailed usage information.
Here's a short overview over the different binaries and their function:

- `psac` is our main executable. This will construct the suffix and LCP array of
  a given input file. Run with `mpirun` for parallel execution.
- `benchmark_sac` benchmarks multiple of our methods. Run with `mpirun`.
  usage as our other binaries. This is a sequential program. No *mpirun* needed.
- `dss` is a wrapper around `libdivsufsort` that follows the same command line
- `psac-vs-dss` runs both our suffix array construction and `libdivsufsort`,
  verifies the results against each other and outputs run-times of both.
- `test_*` various test executables, testing a variety of our internal methods.

## Licensing

Our code is licensed under the
**Apache License 2.0** (see [`LICENSE`](LICENSE)).
The licensing does not apply to the `ext` folder, which contains external
dependencies which are under their own licensing terms.
