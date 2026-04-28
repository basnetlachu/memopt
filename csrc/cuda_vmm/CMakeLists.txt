cmake_minimum_required(VERSION 3.18)
project(memopt_vmm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

# The core allocator (C++ shared lib)
add_library(memopt_vmm SHARED
    vmm_allocator.cpp
)

target_include_directories(memopt_vmm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(memopt_vmm PUBLIC
    CUDA::cuda_driver  # cuMemCreate / cuMemMap / cuMemUnmap / ...
    CUDA::cudart       # cudaMemcpy / cudaDeviceSynchronize / ...
    pthread
)

target_compile_options(memopt_vmm PRIVATE -Wall -Wextra)

# The test binary
add_executable(test_vmm_allocator test_vmm_allocator.cpp)
target_link_libraries(test_vmm_allocator PRIVATE
    memopt_vmm
    CUDA::cuda_driver
    CUDA::cudart
)
target_compile_options(test_vmm_allocator PRIVATE -Wall -Wextra)
