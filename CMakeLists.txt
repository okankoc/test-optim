cmake_minimum_required (VERSION 2.8)

set(WARNING_FLAGS "-Werror -Wall -Weffc++ -Wextra -pedantic -pedantic-errors")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${WARNING_FLAGS} -O3 -DNDEBUG -std=c++11 -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG ${WARNING_FLAGS} -std=c++11 -pthread")
MESSAGE(STATUS "${CMAKE_CXX_FLAGS_RELEASE}")

set(PROJECT_NAME unconstr_optim)
project(${PROJECT_NAME})

#MESSAGE(STATUS "cmake include path" ${CMAKE_INCLUDE_PATH})

# INCLUDE ARMADILLO AS REQUIRED LIBRARY
find_package(Armadillo REQUIRED)
#find_package(Eigen REQUIRED)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
include_directories(${ARMADILLO_INCLUDE_DIRS})
#set (BOOST_ROOT $ENV{HOME}/install)
#set (ARMADILLO_LIBRARY $ENV{HOME}/install/lib)
#set (ARMADILLO_INCLUDE_DIR $ENV{HOME}/install/include)
#set (NLOPT_LIBRARIES $ENV{HOME}/install/lib)
#set (NLOPT_INCLUDE_DIRS $ENV{HOME}/install/include)
#find_package (Boost COMPONENTS program_options unit_test_framework REQUIRED)# PATHS /home/robolab/software/lib) 
# second one is only for testing so in the future test cmake could enforce only
#find_package (Armadillo REQUIRED)# PATHS /home/robolab/software)
#MESSAGE(STATUS "armadillo path: " ${ARMADILLO_INCLUDE_DIRS})

# for loading strings it seems we must have a recent version of boost program options!
#MESSAGE(STATUS "boost path: "${BOOST_ROOT})

# ADD ADOLC INCLUDES AND LIBRARY
#set(ADOLC_PATH ${DIR_ONE_ABOVE}/adolc_base CACHE PATH "path to ADOLC")
#set(ADOLC_PATH /usr/local CACHE PATH "path to ADOLC")
#message(STATUS "Adolc path = " ${ADOLC_PATH})
#find_library(ADOLC NAMES adolc HINTS "${ADOLC_PATH}/lib64" "${ADOLC_PATH}/lib")

#find_package (ADOLC REQUIRED PATHS /usr/local/lib64)
#include_directories(${ADOLC_INCLUDE_DIRS})
#find_package (NLOPT REQUIRED)
#include_directories(${NLOPT_INCLUDE_DIRS})

add_subdirectory(src)
add_subdirectory(test)
# our own library is tested also
#find_package(${PROJECT_NAME} REQUIRED)

# MAKE SURE TO ADD USER LIBRARY TO LD_LIBRARY_PATH!
