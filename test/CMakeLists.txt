cmake_minimum_required (VERSION 2.8)
project(${PROJECT_NAME})

# CREATE SHARED LIBRARY
set(TEST_EXEC optim)
set(TEST_SRC
    test_optim.cpp
)
add_executable (${TEST_EXEC} ${TEST_SRC})

# INCLUDE HEADERS (top folder, as opposed to CMAKE_CURRENT_SOURCE_DIR)
target_include_directories (${TEST_EXEC} PRIVATE
    ${CMAKE_SOURCE_DIR}/include)

# INCLUDE "OUR" SHARED LIBRARY
target_link_libraries(${TEST_EXEC} 
    ${PROJECT_NAME}
    armadillo)

# INSTALL FOLDER
#find_library(${SHARED_OBJ} ${PROJECT_NAME} HINTS ${CMAKE_SOURCE_DIR}/lib)
#target_link_libraries(${TEST_EXEC} ${SHARED_OBJ})
install(TARGETS ${TEST_EXEC}
    DESTINATION ${CMAKE_SOURCE_DIR})

# ADD OPTIONS 
# HACK TO DISABLE EIGEN RELATED WARNINGS!
target_compile_options(${TEST_EXEC} PRIVATE -isystem/usr/local/include/eigen3)
