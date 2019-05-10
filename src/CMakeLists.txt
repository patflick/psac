project(psac-src)

# for tclap command line parsing header library
include_directories(../ext/)


######################
#  main executables  #
######################

# our main executable for suffix array construction
add_executable(psac psac.cpp)
target_link_libraries(psac psac-dss-lib)

add_executable(gsac gsac.cpp)
target_link_libraries(gsac psac-dss-lib)

add_executable(desa-main desa_main.cpp)
target_link_libraries(desa-main psac-dss-lib)


################
#  benchmarks  #
################

# benchmarks of our different internal methods
add_executable(benchmark_sac benchmark.cpp)
target_link_libraries(benchmark_sac psaclib)

# benchmark with different values of `k` for initial k-mer sorting
add_executable(benchmark_k benchmark_k.cpp)
target_link_libraries(benchmark_k psaclib)

# benchmark ANSV
add_executable(benchmark-ansv benchmark_ansv.cpp)
target_link_libraries(benchmark-ansv psaclib)


################
# tools/utils  #
################

add_executable(print64 print64.cpp)

add_executable(mkpattern mkpattern.cpp)

add_executable(kmer-stats kmer_partition.cpp)
target_link_libraries(kmer-stats psac-dss-lib)


################
#  divsufsort  #
################

# divsufsort executable (supporting 32 and 64 bits)
add_executable(dss dss.cpp)
target_link_libraries(dss psac-dss-lib)

# compare our algorithm against divsufsort and check correctness
add_executable(psac-vs-dss psac_vs_dss.cpp)
target_link_libraries(psac-vs-dss psac-dss-lib)

