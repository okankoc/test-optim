cmake_minimum_required (VERSION 2.8)
project(${PROJECT_NAME})

# CREATE EXECUTABLE
set(SOURCES
    matrix.cpp
    optim.cpp
)
add_library(${PROJECT_NAME} SHARED ${SOURCES})

# INCLUDE HEADERS (top folder, as opposed to CMAKE_CURRENT_SOURCE_DIR)
target_include_directories (${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include)
#    ${ADOLC_PATH}/include)

# LINK EXTERNAL LIBRARIES HERE
target_link_libraries(${PROJECT_NAME} 
    armadillo)
#    ${ADOLC})

# INSTALL FOLDER
install(TARGETS ${PROJECT_NAME}
    DESTINATION ${CMAKE_SOURCE_DIR}/libs)
#    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# ADD OPTIONS 
# HACK TO DISABLE EIGEN RELATED WARNINGS!
target_compile_options(${PROJECT_NAME} PRIVATE -isystem/usr/local/include/eigen3)
