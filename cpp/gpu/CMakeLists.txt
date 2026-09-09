cmake_minimum_required(VERSION 3.25.2)

project(sipher_fhe_gpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# FIDESlib (includes patched OpenFHE + CUDA)
find_package(fideslib REQUIRED CONFIG)

if (fideslib_FOUND)
    message(STATUS "FOUND FIDESlib")
    message(STATUS "  Version:  ${fideslib_VERSION}")
    message(STATUS "  Headers:  ${FIDESLIB_INCLUDE_PATH}")
    message(STATUS "  Libs:     ${FIDESLIB_LIBRARIES_PATH}")
    message(STATUS "  Compiler: ${FIDESLIB_CXX_COMPILER}")
else ()
    message(FATAL_ERROR "FIDESlib NOT FOUND — install FIDESlib first")
endif ()

set(CMAKE_CXX_COMPILER ${FIDESLIB_CXX_COMPILER})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FIDESLIB_CXX_FLAGS} -O3")

add_executable(sipher_fhe_gpu
    main_gpu.cpp
    sipher_fhe_gpu.cpp
)

target_link_libraries(sipher_fhe_gpu PRIVATE fideslib::fideslib)
target_link_options(sipher_fhe_gpu PRIVATE -static-libstdc++ -static-libgcc)

set_target_properties(sipher_fhe_gpu PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
