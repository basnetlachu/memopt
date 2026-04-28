# Standalone build for the VMM probe.
#
# This CMakeLists does NOT depend on the rest of the memopt build. Copy this
# directory to the GPU box and build in isolation:
#
#     cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#     cmake --build build -j
#     ./build/vmm_probe [device_id]
#
# Only runtime requirement: CUDA driver (libcuda.so) and a working nvidia
# kernel module. No CUDA runtime (cudart), no nvcc at build time if you
# already have the driver-API headers (cuda.h) installed.
cmake_minimum_required(VERSION 3.18)
project(memopt_vmm_probe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# We only use the CUDA Driver API, not the runtime — so avoid find_package
# (CUDAToolkit) which insists on nvcc. Locate cuda.h and libcuda directly.
set(CUDA_SEARCH_HINTS
    "$ENV{CUDA_HOME}"
    "/usr/local/cuda"
    "/usr/local/cuda-13.0"
    "/usr/local/cuda-13"
    "/usr/local/cuda-12"
    "/opt/cuda")

find_path(CUDA_INCLUDE_DIR cuda.h
    HINTS ${CUDA_SEARCH_HINTS}
    PATH_SUFFIXES include)

# Prefer the real driver (installed with NVIDIA driver); fall back to the
# toolkit stub library so we can link even in non-GPU build environments.
find_library(CUDA_DRIVER_LIB cuda
    HINTS ${CUDA_SEARCH_HINTS}
    PATH_SUFFIXES lib64/stubs lib/stubs lib64 lib
          x86_64-linux-gnu /lib/x86_64-linux-gnu)

if(NOT CUDA_INCLUDE_DIR OR NOT CUDA_DRIVER_LIB)
    message(FATAL_ERROR
      "Could not locate cuda.h or libcuda. Install cuda-driver-dev-XX-Y "
      "(e.g. cuda-driver-dev-13-0) and ensure the NVIDIA driver is present.\n"
      "  CUDA_INCLUDE_DIR = ${CUDA_INCLUDE_DIR}\n"
      "  CUDA_DRIVER_LIB  = ${CUDA_DRIVER_LIB}")
endif()
message(STATUS "cuda.h    : ${CUDA_INCLUDE_DIR}/cuda.h")
message(STATUS "libcuda   : ${CUDA_DRIVER_LIB}")

add_executable(vmm_probe vmm_probe.cpp)
target_include_directories(vmm_probe PRIVATE ${CUDA_INCLUDE_DIR})
target_link_libraries(vmm_probe PRIVATE ${CUDA_DRIVER_LIB})
target_compile_options(vmm_probe PRIVATE -Wall -Wextra -Wno-unused-parameter)

add_executable(vmm_probe_concurrent vmm_probe_concurrent.cpp)
target_include_directories(vmm_probe_concurrent PRIVATE ${CUDA_INCLUDE_DIR})
target_link_libraries(vmm_probe_concurrent PRIVATE ${CUDA_DRIVER_LIB})
target_compile_options(vmm_probe_concurrent PRIVATE
    -Wall -Wextra -Wno-unused-parameter)
