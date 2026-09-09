cmake_minimum_required(VERSION 3.16)
project(sipher_fhe VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# OpenFHE
find_package(OpenFHE REQUIRED CONFIG PATHS $ENV{HOME}/.local/lib/OpenFHE)

# Include
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories($ENV{HOME}/.local/include/openfhe)
include_directories($ENV{HOME}/.local/include/openfhe/third-party/include)
include_directories($ENV{HOME}/.local/include/openfhe/core)
include_directories($ENV{HOME}/.local/include/openfhe/pke)
include_directories($ENV{HOME}/.local/include/openfhe/binfhe)

# Link directories
link_directories($ENV{HOME}/.local/lib)

# Library
add_library(sipher_fhe_lib STATIC
    sipher_fhe.cpp
)
target_link_libraries(sipher_fhe_lib PUBLIC
    OPENFHEcore
    OPENFHEpke
    OPENFHEbinfhe
)

# Executable
add_executable(sipher_fhe main.cpp)
target_link_libraries(sipher_fhe PRIVATE sipher_fhe_lib)

# Install
install(TARGETS sipher_fhe DESTINATION bin)

message(STATUS "Sipher FHE Inference Engine")
message(STATUS "  OpenFHE: ${OpenFHE_LIBDIR}")
message(STATUS "  Build: ${CMAKE_BUILD_TYPE}")
