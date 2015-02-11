cmake_minimum_required(VERSION 2.6)

# project settings
project(psac)

##### General Compilation Settings

# Initialize CXXFLAGS.
#set(CMAKE_CXX_FLAGS                "-Wall -O0 -g --std=c++11 -D_GLIBCXX_DEBUG")
#set(CMAKE_CXX_FLAGS                "-Wall -Wuninitialized -O0 -g --std=c++11")
#set(CMAKE_CXX_FLAGS                "-Wall -O3 -funroll-loops -msse3 --std=c++11 -g -march=native")
#set(CMAKE_CXX_FLAGS                "-Wall -O3 -funroll-loops -msse3 -DNDEBUG --std=c++11 -march=native")
#set(CMAKE_CXX_FLAGS                "-Wall -O3 -DNDEBUG --std=c++11 -Wuninitialized -g")


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wuninitialized --std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -funroll-loops")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} -g")

# Add these standard paths to the search paths for FIND_LIBRARY
# to find libraries from these locations first
if(UNIX)
    set(CMAKE_LIBRARY_PATH "${CMAKE_LIBRARY_PATH} /lib /usr/lib")
endif()

# --------------------------------------------------------------
# Indicate CMake 2.7 and above that we don't want to mix relative
#  and absolute paths in linker lib lists.
# Run "cmake --help-policy CMP0003" for more information.
# --------------------------------------------------------------
if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif()


#### MPI
find_package(MPI REQUIRED)
if (MPI_FOUND)
    #set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MPI_COMPILE_FLAGS}")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS}")
    #set(CMAKE_LINK_FLAGS "${CMAKE_LINK_FLAGS} ${MPI_LINK_FLAGS}")
    set(EXTRA_LIBS ${EXTRA_LIBS} ${MPI_LIBRARIES})
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})
else (MPI_FOUND)
    message(SEND_ERROR "This application cannot compile without MPI")
endif (MPI_FOUND)




## add ScoreP support


option(ENABLE_PROFILING_SCOREP "Enable ScoreP profiling support." OFF)

#if(ENABLE_PROFILING_SCOREP)
#    message(STATUS "Profiling with ScoreP enabled.")
#
#    execute_process(COMMAND scorep-config --cxxflags --nocompiler
#                    OUTPUT_VARIABLE SCOREP_CXX_FLAGS)
#                ##RESULT_VARIABLE SCOREP_CONFIG_RETURN)
#                #if(NOT SCOREP_CONFIG_RETURN EQUAL 0)
#                #    message(FATAL_ERROR "Can NOT execute 'scorep-config' at $ENV{SCOREP_ROOT}/bin/scorep-config - check file permissions")
#                #endif()
#    execute_process(COMMAND scorep-config --mpp=mpi --ldflags
#                    OUTPUT_VARIABLE SCOREP_LD_FLAGS)
#    execute_process(COMMAND scorep-config --mpp=mpi --libs
#                    OUTPUT_VARIABLE SCOREP_LIBS)
#                string(STRIP "${SCOREP_LIBS}" SCOREP_LIBS)
#
#                #execute_process(COMMAND scorep-config --mpp=mpi --libs
#                #                OUTPUT_VARIABLE SCOREP_LIBFLAGS)
#                #string(STRIP "${SCOREP_LIBFLAGS}" SCOREP_LIBFLAGS)
#
#    # subsystem iniialization file
#    execute_process(COMMAND scorep-config --mpp=mpi --nocuda --noopencl --adapter-init
#                    OUTPUT_VARIABLE SCOREP_INIT_FILE)
#    file(WRITE ${CMAKE_BINARY_DIR}/scorep_init.c "${SCOREP_INIT_FILE}")
#
#    #if(SCOREP_ENABLE)
#        set(SCOREP_SRCFILES "${CMAKE_BINARY_DIR}/scorep_init.c")
#        #endif(SCOREP_ENABLE)
#
#
#    # TODO: possibly put the LIBS to target_link_libraries
#    set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${SCOREP_LD_FLAGS} ${SCOREP_LIBS}")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SCOREP_CXX_FLAGS}")
#    #set(EXTRA_LIBS ${EXTRA_LIBS} ${SCOREP_LIBS})
#endif(ENABLE_PROFILING_SCOREP)


if(ENABLE_PROFILING_SCOREP)
    message(STATUS "Profiling with ScoreP enabled.")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "scorep")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "scorep")
endif(ENABLE_PROFILING_SCOREP)



###### Executable and Libraries
# Save libs and executables in the same place
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib CACHE PATH "Output directory for libraries" )
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin CACHE PATH "Output directory for applications" )


# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
#include_directories("${PROJECT_SOURCE_DIR}/src")
#include_directories("${PROJECT_SOURCE_DIR}/include")


message(STATUS "scorep libs: ${SCOREP_LIBS}, MPI libs: ${EXTRA_LIBS}")

add_executable(test_sac main.cpp timer.cpp ${SCOREP_SRCFILES})
target_link_libraries(test_sac ${EXTRA_LIBS})
target_link_libraries(test_sac rt)
#target_link_libraries(test_sac ${SCOREP_LIBS})

add_executable(test_samplesort test_sort.cpp timer.cpp)
target_link_libraries(test_samplesort ${EXTRA_LIBS})
target_link_libraries(test_samplesort rt) # for timer

add_executable(test_file test_file.cpp)
message(STATUS "Extra libs: ${EXTRA_LIBS}")
target_link_libraries(test_file ${EXTRA_LIBS})

add_executable(test_rmq test_rmq.cpp)
add_executable(test_bitops test_bitops.cpp)

add_subdirectory(ext/libdivsufsort)
include_directories(${libdivsufsort_BINARY_DIR}/include)
include_directories(${libdivsufsort_SOURCE_DIR}/include)
add_executable(test_ldss test_libdss.cpp timer.cpp)
target_link_libraries(test_ldss divsufsort)
target_link_libraries(test_ldss ${EXTRA_LIBS})
target_link_libraries(test_ldss rt)

