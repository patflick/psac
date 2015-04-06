Parallel Suffix Array and LCP Construction
------------------------------------------

This library implements a distributed-memory parallel algorithm for the
construction of suffix and LCP arrays. The algorithm is implemented in `C++11`
and `MPI`.

## Authors

- Patrick Flick

## Code organization

- [`include/`](include/) contains the implementation of our algorithms in form
  of C++ template header files (a header-only library).
- [`include/mxx/`](include/mxx/) contains the `mxx` library, which is a header
  C++ template library with C++ bindings and algorithms for MPI. We intend to
  release `mxx` as a separate library eventually. Please see the
  [README](include/mxx/README.md) in that folder for further information on the
  `mxx` library.
- [`src/`](src/) contains the sources for binaries, which make use of the
  implementations in [`include/`](include/).
- [`src/tests`](src/tests/) contains tests for some of the components of the
  library.
- [`ext/`](ext/) contains external, third-party dependencies/libraries. See the
  [README](ext/README.md) for details on the third-party libraries used.


### Dependencies

- `cmake` version >= 2.6
- `gcc` version >= 4.7 (maybe 4.8)
- an `MPI` implementation supporting `MPI-2` or `MPI-3`.
- *(planned)* Google Test framework for unit tests
- and whatever comes along in the `ext` directory

### Compiling

Compiling all binaries and tests is easy thanks to `cmake`. Simply follow the
following steps:

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

TBD
