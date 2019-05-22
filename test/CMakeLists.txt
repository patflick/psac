cmake_minimum_required(VERSION 2.6)

# project settings
project(psac-test)

# bitops TODO: don't use mxx for these sequential tests!
add_executable(test-bitops test_bitops.cpp)
target_link_libraries(test-bitops psaclib mxx-gtest-main)

add_executable(test-rmq test_rmq.cpp)
target_link_libraries(test-rmq psaclib mxx-gtest-main)

add_executable(test-samplelcp test_samplelcp.cpp)
target_link_libraries(test-samplelcp psac-dss-lib mxx-gtest-main)

# Parallel tests
add_executable(test-ansv test_ansv.cpp)
target_link_libraries(test-ansv psaclib mxx-gtest-main)

add_executable(test-ss test_stringset.cpp)
target_link_libraries(test-ss psaclib mxx-gtest-main)

add_executable(test-gsa test_gsa.cpp)
target_link_libraries(test-gsa psaclib mxx-gtest-main)

add_executable(test-suffixtree test_suffixtree.cpp)
target_link_libraries(test-suffixtree psaclib mxx-gtest-main)

add_executable(test-psac test_psac.cpp)
target_link_libraries(test-psac psac-dss-lib mxx-gtest-main)

add_executable(test-seq-query test_seq_query.cpp)
target_link_libraries(test-seq-query psac-dss-lib mxx-gtest-main)

add_executable(test-desa test_desa.cpp)
target_link_libraries(test-desa psac-dss-lib mxx-gtest-main)

# standalone tests
#add_executable(test-ss test_stringset.cpp)
#target_link_libraries(test-ss ${EXTRA_LIBS} rt)

# standalone tests
add_executable(test-lcp test_glcp.cpp)
target_link_libraries(test-lcp psaclib)
