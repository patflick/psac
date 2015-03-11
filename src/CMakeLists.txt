cmake_minimum_required(VERSION 2.6)

# project settings
project(psac-main)

# timer shared lib
add_library(psac-timer timer.cpp)
target_link_libraries(psac-timer rt)

add_executable(test_sac main.cpp)
target_link_libraries(test_sac ${EXTRA_LIBS})
target_link_libraries(test_sac psac-timer)

add_executable(benchmark_sac benchmark.cpp)
target_link_libraries(benchmark_sac ${EXTRA_LIBS})
target_link_libraries(benchmark_sac psac-timer)
