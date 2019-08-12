cmake_minimum_required(VERSION 2.6)

# project settings
project(psac)

#### Options
OPTION(PSAC_BUILD_EXES "Build all psac executables (command line tools & benchmark scripts)" ON)
OPTION(PSAC_BUILD_TESTS "Build unit tests" ON)
OPTION(PSAC_ENABLE_COVERAGE "Enable code coverage reporting" OFF)

##### General Compilation Settings

# Initialize CXXFLAGS.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wuninitialized --std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

### Test Coverage
if(PSAC_ENABLE_COVERAGE)
  # turn off stack protection for gcov coverage, because the stack protector shows
  # up as a never taken branch, and thus turns any last statement in a function
  # with a stack procetor into a partially covered statement.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -fno-stack-protector")
endif(PSAC_ENABLE_COVERAGE)

###### Executable and Libraries
# Save libs and executables in the same place
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

## psac header library
add_library(psaclib INTERFACE)
target_include_directories(psaclib INTERFACE include)

## psac dependencies

# load mxx and its gtest wrapper
set(MXX_BUILD_TESTS OFF CACHE BOOL "disable building mxx tests" FORCE)
add_subdirectory(ext/mxx)
target_link_libraries(psaclib INTERFACE mxx)

# include libdivsufsort (with 64bit support but without examples)
set(BUILD_DIVSUFSORT64 ON CACHE BOOL "enables divsufsort 64bit functions" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "enables divsufsort examples" FORCE)
add_subdirectory(ext/libdivsufsort)

# prettyprint
target_include_directories(psaclib INTERFACE ext/cxx-prettyprint)

# divsufsort (integrate into psac as `psac-dss-lib`)
add_library(psac-dss-lib INTERFACE)
target_link_libraries(psac-dss-lib INTERFACE psaclib)
target_include_directories(psac-dss-lib INTERFACE ${libdivsufsort_BINARY_DIR}/include)
target_link_libraries(psac-dss-lib INTERFACE divsufsort divsufsort64)


## build executables
if(PSAC_BUILD_EXES)
  add_subdirectory(src)
endif()

# build tests
if (PSAC_BUILD_TESTS)
  add_subdirectory(test)
endif()

