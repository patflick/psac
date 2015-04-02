cmake_minimum_required(VERSION 2.6)

# project settings
project(psac-main)

# timer shared lib
add_library(psac-timer timer.cpp)
target_link_libraries(psac-timer rt)

add_executable(test_sac main.cpp)
target_link_libraries(test_sac ${EXTRA_LIBS})
target_link_libraries(test_sac psac-timer)

# our main executable for suffix array construction
add_executable(psac psac.cpp)
target_link_libraries(psac ${EXTRA_LIBS})
target_link_libraries(psac psac-timer)

# benchmarks of our different internal methods
add_executable(benchmark_sac benchmark.cpp)
target_link_libraries(benchmark_sac ${EXTRA_LIBS})
target_link_libraries(benchmark_sac psac-timer)

# divsufsort executable (supporting 32 and 64 bits)
add_executable(dss dss.cpp)
target_link_libraries(dss psac-timer)
target_link_libraries(dss divsufsort)
target_link_libraries(dss divsufsort64)

# compare our algorithm against divsufsort and check correctness
add_executable(psac-vs-dss psac_vs_dss.cpp)
target_link_libraries(psac-vs-dss ${EXTRA_LIBS})
target_link_libraries(psac-vs-dss psac-timer)
target_link_libraries(psac-vs-dss divsufsort)
target_link_libraries(psac-vs-dss divsufsort64)

