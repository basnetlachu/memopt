cmake_minimum_required(VERSION 3.18)
project(memopt_csrc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ═══════════════════════════════════════════════════════════════════════════
# Options — all optional features degrade gracefully
# ═══════════════════════════════════════════════════════════════════════════
option(MEMOPT_ENABLE_GDS       "Enable GPUDirect Storage (cuFile)"  OFF)
option(MEMOPT_ENABLE_IO_URING  "Enable async NVMe via io_uring"     OFF)
option(MEMOPT_ENABLE_RDMA      "Enable RDMA (libibverbs)"           OFF)
option(MEMOPT_ENABLE_AVX512    "Enable AVX-512 SIMD"                OFF)
option(MEMOPT_ENABLE_SANITIZERS "Enable ASan + TSan for testing"    OFF)
option(MEMOPT_ENABLE_TESTS     "Build C++ unit tests"               OFF)

# ═══════════════════════════════════════════════════════════════════════════
# Compiler flags
# ═══════════════════════════════════════════════════════════════════════════
include(cmake/CompilerFlags.cmake)

# ═══════════════════════════════════════════════════════════════════════════
# Find pybind11
# ═══════════════════════════════════════════════════════════════════════════
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    # Fall back to pip-installed pybind11
    execute_process(
        COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(pybind11_DIR)
        find_package(pybind11 CONFIG REQUIRED PATHS ${pybind11_DIR})
    else()
        # FetchContent as last resort
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG        v2.13.6
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
endif()

# ═══════════════════════════════════════════════════════════════════════════
# Optional: Find CUDA
# ═══════════════════════════════════════════════════════════════════════════
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(MEMOPT_HAS_CUDA TRUE)
    find_package(CUDAToolkit REQUIRED)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER_VERSION}")

    # Detect SM targets from environment or auto-detect
    if(DEFINED ENV{TORCH_CUDA_ARCH_LIST})
        # Use PyTorch's arch list if available
        set(CMAKE_CUDA_ARCHITECTURES $ENV{TORCH_CUDA_ARCH_LIST})
    else()
        # Default: Ampere + Hopper + Blackwell
        set(CMAKE_CUDA_ARCHITECTURES "86;90;100")
    endif()
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    set(MEMOPT_HAS_CUDA FALSE)
    message(STATUS "CUDA not found — GPU components disabled")
endif()

# ═══════════════════════════════════════════════════════════════════════════
# Phase 1a: _memopt_core (page_table + oracle)
# ═══════════════════════════════════════════════════════════════════════════
pybind11_add_module(_memopt_core
    core/page_table.cpp
    core/oracle.cpp
    core/block_directory.cpp
    core/bindings.cpp
)
target_include_directories(_memopt_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/core
)
target_compile_features(_memopt_core PRIVATE cxx_std_17)

# Install to the memopt package directory
install(TARGETS _memopt_core
    LIBRARY DESTINATION memopt
)

# ═══════════════════════════════════════════════════════════════════════════
# Phase 1b: _memopt_hooks (dispatch table + FNV-1a keys + atomic counters)
# ═══════════════════════════════════════════════════════════════════════════
pybind11_add_module(_memopt_hooks
    hooks/dispatch_table.cpp
    hooks/bindings.cpp
)
target_include_directories(_memopt_hooks PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/hooks
)
target_compile_features(_memopt_hooks PRIVATE cxx_std_17)

install(TARGETS _memopt_hooks
    LIBRARY DESTINATION memopt
)

# ═══════════════════════════════════════════════════════════════════════════
# Phase 1c: _memopt_paged (block manager + optional CUDA gather)
# ═══════════════════════════════════════════════════════════════════════════
if(MEMOPT_HAS_CUDA)
    pybind11_add_module(_memopt_paged
        paged/block_pool.cpp
        paged/paged_kv_cache.cpp
        paged/gather_kernel.cu
        paged/bindings.cpp
    )
    target_compile_definitions(_memopt_paged PRIVATE MEMOPT_CUDA_AVAILABLE)
    set_target_properties(_memopt_paged PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_compile_options(_memopt_paged PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    )
else()
    # CPU-only build — no gather kernel, link stub from .cu compiled as C++
    pybind11_add_module(_memopt_paged
        paged/block_pool.cpp
        paged/paged_kv_cache.cpp
        paged/bindings.cpp
    )
    # Compile the CPU stub from gather_kernel.cu as C++
    # The #ifndef __CUDACC__ path provides empty stubs
    target_sources(_memopt_paged PRIVATE paged/gather_kernel.cu)
    set_source_files_properties(paged/gather_kernel.cu PROPERTIES
        LANGUAGE CXX
    )
endif()

target_include_directories(_memopt_paged PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/paged
)
if(MEMOPT_HAS_CUDA)
    target_include_directories(_memopt_paged PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()
target_compile_features(_memopt_paged PRIVATE cxx_std_17)

# Enable 16-byte CAS (cmpxchg16b) for lock-free block pool on x86-64
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_compile_options(_memopt_paged PRIVATE -mcx16)
    target_link_libraries(_memopt_paged PRIVATE atomic)
endif()

install(TARGETS _memopt_paged
    LIBRARY DESTINATION memopt
)

# ═══════════════════════════════════════════════════════════════════════════
# Phase 2a: _memopt_cuda (stream pool + NVMe I/O + GDS)
# ═══════════════════════════════════════════════════════════════════════════

# Optional: cuFile GDS
if(MEMOPT_ENABLE_GDS)
    find_library(CUFILE_LIB cufile
        HINTS /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu)
    if(CUFILE_LIB)
        set(MEMOPT_GDS_AVAILABLE TRUE)
        message(STATUS "memopt: cuFile GDS enabled")
    else()
        message(STATUS "memopt: cuFile GDS requested but not found — disabled")
    endif()
endif()

pybind11_add_module(_memopt_cuda
    cuda_backend/stream_pool.cpp
    cuda_backend/nvme_io.cpp
    cuda_backend/gds.cpp
    cuda_backend/bindings.cpp
)
target_include_directories(_memopt_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_backend
)
if(MEMOPT_HAS_CUDA)
    target_include_directories(_memopt_cuda PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()
target_compile_features(_memopt_cuda PRIVATE cxx_std_17)

if(MEMOPT_HAS_CUDA)
    target_compile_definitions(_memopt_cuda PRIVATE MEMOPT_CUDA_AVAILABLE)
    target_link_libraries(_memopt_cuda PRIVATE
        CUDA::cudart
    )
endif()
if(MEMOPT_GDS_AVAILABLE)
    target_compile_definitions(_memopt_cuda PRIVATE MEMOPT_GDS_AVAILABLE)
    target_link_libraries(_memopt_cuda PRIVATE ${CUFILE_LIB})
endif()

# ── io_uring (async NVMe) ─────────────────────────────────────────────
if(MEMOPT_ENABLE_IO_URING)
    find_library(LIBURING_LIB
        NAMES uring liburing
        PATHS /usr/lib /usr/local/lib)

    find_path(LIBURING_INCLUDE
        NAMES liburing.h
        PATHS /usr/include /usr/local/include)

    if(LIBURING_LIB AND LIBURING_INCLUDE)
        message(STATUS "io_uring: found at ${LIBURING_LIB}")

        target_compile_definitions(_memopt_cuda PRIVATE
            MEMOPT_IO_URING_AVAILABLE=1)

        target_include_directories(_memopt_cuda PRIVATE
            ${LIBURING_INCLUDE})

        target_link_libraries(_memopt_cuda PRIVATE
            ${LIBURING_LIB})

        target_sources(_memopt_cuda PRIVATE
            cuda_backend/nvme_async.cpp)

        message(STATUS "io_uring: async NVMe enabled")
    else()
        message(WARNING
            "liburing not found. "
            "Async NVMe disabled. "
            "Install: apt install liburing-dev")
        set(MEMOPT_ENABLE_IO_URING OFF)
    endif()
else()
    message(STATUS
        "io_uring: sync fallback "
        "(use -DMEMOPT_ENABLE_IO_URING=ON to enable)")
endif()

# Always compile nvme_async.cpp — it degrades to synchronous pread()
# when MEMOPT_IO_URING_AVAILABLE is not defined.
if(NOT MEMOPT_ENABLE_IO_URING)
    target_sources(_memopt_cuda PRIVATE
        cuda_backend/nvme_async.cpp)
endif()

install(TARGETS _memopt_cuda
    LIBRARY DESTINATION memopt
)

# ═══════════════════════════════════════════════════════════════════════════
# Phase 2b: memopt-transport (standalone sidecar daemon)
# ═══════════════════════════════════════════════════════════════════════════

# Optional: RDMA / InfiniBand
if(MEMOPT_ENABLE_RDMA)
    find_library(IBVERBS_LIB ibverbs)
    find_library(RDMACM_LIB  rdmacm)
    if(IBVERBS_LIB AND RDMACM_LIB)
        set(MEMOPT_RDMA_AVAILABLE TRUE)
        message(STATUS "memopt: RDMA/InfiniBand enabled")
    else()
        message(STATUS "memopt: RDMA requested but libibverbs not found — TCP only")
    endif()
endif()

add_executable(memopt-transport
    transport/ring_buffer.cpp
    transport/rdma_engine.cpp
    transport/tcp_fallback.cpp
    transport/daemon.cpp
    transport/main.cpp
)
target_include_directories(memopt-transport PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/transport
)
target_compile_features(memopt-transport PRIVATE cxx_std_17)
target_sources(memopt-transport PRIVATE transport/env_overrides.cpp)
target_link_libraries(memopt-transport PRIVATE pthread)

if(MEMOPT_RDMA_AVAILABLE)
    target_compile_definitions(memopt-transport PRIVATE MEMOPT_RDMA_AVAILABLE)
    target_link_libraries(memopt-transport PRIVATE
        ${IBVERBS_LIB} ${RDMACM_LIB})
endif()

install(TARGETS memopt-transport RUNTIME DESTINATION bin)

# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: _memopt_simd (AVX-512 / AVX2 / scalar prefix matching)
# ═══════════════════════════════════════════════════════════════════════════
pybind11_add_module(_memopt_simd
    simd/prefix_match.cpp
    simd/bindings.cpp
)
target_include_directories(_memopt_simd PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/simd
)
target_compile_features(_memopt_simd PRIVATE cxx_std_17)

# SIMD flags — auto-detect via -march=native in Release, explicit otherwise
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(_memopt_simd PRIVATE -O3 -march=native)
elseif(MEMOPT_ENABLE_AVX512)
    target_compile_options(_memopt_simd PRIVATE -mavx512f -mavx512bw)
endif()

install(TARGETS _memopt_simd
    LIBRARY DESTINATION memopt
)

# ═══════════════════════════════════════════════════════════════════════════
# C++ Unit Tests (GoogleTest)
# ═══════════════════════════════════════════════════════════════════════════
if(MEMOPT_ENABLE_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.15.2
    )
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    # test_page_table
    add_executable(test_page_table
        ${CMAKE_CURRENT_SOURCE_DIR}/../tests/cpp/test_page_table.cpp
        core/page_table.cpp
    )
    target_include_directories(test_page_table PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core
    )
    target_link_libraries(test_page_table PRIVATE
        gtest_main
        pthread
    )
    target_compile_features(test_page_table PRIVATE cxx_std_17)
    add_test(NAME test_page_table COMMAND test_page_table)

    # test_oracle
    add_executable(test_oracle
        ${CMAKE_CURRENT_SOURCE_DIR}/../tests/cpp/test_oracle.cpp
        core/oracle.cpp
    )
    target_include_directories(test_oracle PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core
    )
    target_link_libraries(test_oracle PRIVATE
        gtest_main
        pthread
    )
    target_compile_features(test_oracle PRIVATE cxx_std_17)
    add_test(NAME test_oracle COMMAND test_oracle)

    # test_dispatch_table
    add_executable(test_dispatch_table
        ${CMAKE_CURRENT_SOURCE_DIR}/../tests/cpp/test_dispatch_table.cpp
        hooks/dispatch_table.cpp
    )
    target_include_directories(test_dispatch_table PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/hooks
    )
    target_link_libraries(test_dispatch_table PRIVATE
        gtest_main
        pthread
    )
    target_compile_features(test_dispatch_table PRIVATE cxx_std_17)
    add_test(NAME test_dispatch_table COMMAND test_dispatch_table)

    # test_block_pool
    add_executable(test_block_pool
        ${CMAKE_CURRENT_SOURCE_DIR}/../tests/cpp/test_block_pool.cpp
        paged/block_pool.cpp
    )
    target_include_directories(test_block_pool PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/paged
    )
    target_link_libraries(test_block_pool PRIVATE
        gtest_main
        pthread
    )
    target_compile_features(test_block_pool PRIVATE cxx_std_17)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        target_compile_options(test_block_pool PRIVATE -mcx16)
        target_link_libraries(test_block_pool PRIVATE atomic)
    endif()
    add_test(NAME test_block_pool COMMAND test_block_pool)

    # test_cuda_backend
    add_executable(test_cuda_backend
        ${CMAKE_CURRENT_SOURCE_DIR}/../tests/cpp/test_cuda_backend.cpp
        cuda_backend/stream_pool.cpp
        cuda_backend/nvme_io.cpp
        cuda_backend/gds.cpp
    )
    target_include_directories(test_cuda_backend PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda_backend
    )
    target_link_libraries(test_cuda_backend PRIVATE
        gtest_main
        pthread
    )
    target_compile_features(test_cuda_backend PRIVATE cxx_std_17)
    if(MEMOPT_HAS_CUDA)
        target_compile_definitions(test_cuda_backend PRIVATE MEMOPT_CUDA_AVAILABLE)
        target_link_libraries(test_cuda_backend PRIVATE CUDA::cudart)
    endif()
    if(MEMOPT_GDS_AVAILABLE)
        target_compile_definitions(test_cuda_backend PRIVATE MEMOPT_GDS_AVAILABLE)
        target_link_libraries(test_cuda_backend PRIVATE ${CUFILE_LIB})
    endif()
    add_test(NAME test_cuda_backend COMMAND test_cuda_backend)

    # test_transport
    add_executable(test_transport
        ${CMAKE_CURRENT_SOURCE_DIR}/../tests/cpp/test_transport.cpp
        transport/ring_buffer.cpp
        transport/rdma_engine.cpp
        transport/tcp_fallback.cpp
        transport/daemon.cpp
        transport/env_overrides.cpp
    )
    target_include_directories(test_transport PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/transport
    )
    target_link_libraries(test_transport PRIVATE
        gtest_main
        pthread
    )
    target_compile_features(test_transport PRIVATE cxx_std_17)
    if(MEMOPT_RDMA_AVAILABLE)
        target_compile_definitions(test_transport PRIVATE MEMOPT_RDMA_AVAILABLE)
        target_link_libraries(test_transport PRIVATE
            ${IBVERBS_LIB} ${RDMACM_LIB})
    endif()
    add_test(NAME test_transport COMMAND test_transport)

    # test_prefix_match
    add_executable(test_prefix_match
        ${CMAKE_CURRENT_SOURCE_DIR}/../tests/cpp/test_prefix_match.cpp
        simd/prefix_match.cpp
    )
    target_include_directories(test_prefix_match PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/simd
    )
    target_link_libraries(test_prefix_match PRIVATE
        gtest_main
        pthread
    )
    target_compile_features(test_prefix_match PRIVATE cxx_std_17)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(test_prefix_match PRIVATE -O3 -march=native)
    elseif(MEMOPT_ENABLE_AVX512)
        target_compile_options(test_prefix_match PRIVATE -mavx512f -mavx512bw)
    endif()
    add_test(NAME test_prefix_match COMMAND test_prefix_match)
endif()


# ═══════════════════════════════════════════════════════════════════════════
# AMD ROCm backend
# ═══════════════════════════════════════════════════════════════════════════
#
# Two compile modes:
#   MEMOPT_ENABLE_ROCM=ON   → require HIP + ROCm; build real bindings
#   MEMOPT_ENABLE_ROCM=OFF  → build stub module so Python imports
#                             always succeed on CUDA-only hosts
#
# The stub emits `_memopt_rocm.ROCM_AVAILABLE = False`; the real module
# emits `ROCM_AVAILABLE = True`. Python shim (_rocm_backend_py.py)
# checks this attribute.

option(MEMOPT_ENABLE_ROCM "Build AMD ROCm backend" OFF)

if(MEMOPT_ENABLE_ROCM)
    set(ROCM_PATH $ENV{ROCM_PATH}
        CACHE PATH "Path to ROCm installation")
    if(NOT ROCM_PATH)
        set(ROCM_PATH "/opt/rocm")
    endif()

    find_package(hip QUIET
        PATHS ${ROCM_PATH}/cmake
              ${ROCM_PATH}/lib/cmake/hip)

    if(NOT hip_FOUND)
        message(WARNING
            "HIP not found at ${ROCM_PATH}. "
            "ROCm backend disabled — building stub. "
            "Set ROCM_PATH or install ROCm.")
        set(MEMOPT_ENABLE_ROCM OFF)
    else()
        message(STATUS "Found HIP: ${hip_VERSION}")

        pybind11_add_module(_memopt_rocm
            rocm/rocm_backend.cpp)
        target_compile_definitions(_memopt_rocm PRIVATE
            MEMOPT_ROCM_AVAILABLE=1)
        target_link_libraries(_memopt_rocm PRIVATE hip::host)

        # gfx90a  = MI210, MI250X
        # gfx940  = MI300A
        # gfx941  = MI300X
        # gfx942  = MI300X (newer revision)
        set_target_properties(_memopt_rocm PROPERTIES
            HIP_ARCHITECTURES "gfx90a;gfx940;gfx941;gfx942")

        install(TARGETS _memopt_rocm LIBRARY DESTINATION memopt)
        message(STATUS "ROCm backend: enabled")
    endif()
endif()

if(NOT MEMOPT_ENABLE_ROCM)
    # Stub build — no MEMOPT_ROCM_AVAILABLE define, so the cpp emits
    # the stub module. Safe to compile on NVIDIA / macOS / CPU-only.
    pybind11_add_module(_memopt_rocm
        rocm/rocm_backend.cpp)
    install(TARGETS _memopt_rocm LIBRARY DESTINATION memopt)
    message(STATUS "ROCm backend: stub (disabled)")
endif()
