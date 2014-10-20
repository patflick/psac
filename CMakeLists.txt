cmake_minimum_required(VERSION 2.6)

# project settings
project(psac)

##### General Compilation Settings

# Initialize CXXFLAGS.
#set(CMAKE_CXX_FLAGS                "-Wall -O0 -g --std=c++11")
set(CMAKE_CXX_FLAGS                "-Wall -O3 -funroll-loops -msse3 -DNDEBUG --std=c++11")



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
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MPI_COMPILE_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS}")
    set(CMAKE_LINK_FLAGS "${CMAKE_LINK_FLAGS} ${MPI_LINK_FLAGS}")
    set(EXTRA_LIBS ${EXTRA_LIBS} ${MPI_LIBRARIES})
    include_directories(${MPI_INCLUDE_PATH})
else (MPI_FOUND)
    message(SEND_ERROR "This application cannot compile without MPI")
endif (MPI_FOUND)

###### Executable and Libraries
# Save libs and executables in the same place
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib CACHE PATH "Output directory for libraries" )
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin CACHE PATH "Output directory for applications" )


# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
#include_directories("${PROJECT_SOURCE_DIR}/src")
#include_directories("${PROJECT_SOURCE_DIR}/include")



#add_executable(test_sac main.cpp)
#target_link_libraries(test_sac ${MPI_LIBRARIES})

add_executable(test_samplesort test_sort.cpp timer.cpp)
target_link_libraries(test_samplesort ${MPI_LIBRARIES})
target_link_libraries(test_samplesort rt) # for timer

add_executable(ref reference.cpp)
