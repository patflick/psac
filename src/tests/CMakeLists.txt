cmake_minimum_required(VERSION 2.6)

# project settings
project(psac-tests)

# test sort: TODO: add this to MXX
add_executable(test_samplesort test_sort.cpp)
target_link_libraries(test_samplesort ${EXTRA_LIBS})
target_link_libraries(test_samplesort psac-timer) # for timer

# MPI File TODO: add this to MXX
add_executable(test_file test_file.cpp)
target_link_libraries(test_file ${EXTRA_LIBS})

# RMQ
add_executable(test_rmq test_rmq.cpp)
# Bitops
add_executable(test_bitops test_bitops.cpp)

# testing against libdivsufsort
add_executable(test_ldss test_libdss.cpp)
target_link_libraries(test_ldss divsufsort64)
target_link_libraries(test_ldss ${EXTRA_LIBS})
target_link_libraries(test_ldss psac-timer) # for timer
