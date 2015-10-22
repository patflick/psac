cmake_minimum_required(VERSION 2.6)

# project settings
project(psac-test)

# bitops TODO: don't use mxx for these sequential tests!
add_executable(test-bitops test_bitops.cpp)
target_link_libraries(test-bitops mxx-gtest-main)

add_executable(test-rmq test_rmq.cpp)
target_link_libraries(test-rmq mxx-gtest-main)

