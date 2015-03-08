cmake_minimum_required(VERSION 2.6)

# project settings
project(psac-main)

# timer shared lib
add_library(psac-timer timer.cpp)
target_link_libraries(psac-timer rt)

add_executable(test_sac main.cpp ${SCOREP_SRCFILES})
target_link_libraries(test_sac ${EXTRA_LIBS})
target_link_libraries(test_sac psac-timer) # TODO continuehere!
#target_link_libraries(test_sac ${SCOREP_LIBS})
