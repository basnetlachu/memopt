# CompilerFlags.cmake — compiler flags for memopt C++ components

# Release flags
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR NOT CMAKE_BUILD_TYPE)
    add_compile_options(-O3)

    # LTO for release builds
    include(CheckIPOSupported)
    check_ipo_supported(RESULT IPO_SUPPORTED)
    if(IPO_SUPPORTED)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
        message(STATUS "LTO enabled")
    endif()
endif()

# Warnings
add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Wno-unused-parameter
)

# AVX-512
if(MEMOPT_ENABLE_AVX512)
    add_compile_options(-mavx512f -mavx512bw)
    add_compile_definitions(MEMOPT_AVX512=1)
    message(STATUS "AVX-512 enabled")
endif()

# Sanitizers
if(MEMOPT_ENABLE_SANITIZERS)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
    message(STATUS "AddressSanitizer + UBSan enabled")
endif()
