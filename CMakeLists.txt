# 1. Require CMake 3.28+ for modern standard support and compatibility
cmake_minimum_required(VERSION 3.28)

# 2. Define project name, version, and language
project(CrescendoEngine 
    VERSION 1.0.0
    DESCRIPTION "Bare-Metal C++26 AI Music Generator & Stem Separator"
    LANGUAGES CXX
)

# 3. Cross-Platform C++26 Configuration, MSVC uses /std:c++latest for C++26; GCC/Clang use cxx_std_26.
set(CMAKE_CXX_EXTENSIONS OFF) # Enforce strict ISO C++ over vendor extensions

function(target_enable_cxx26 TARGET_NAME)
    if(MSVC)
        # /std:c++latest enables C++26 draft features
        # /Zc:__cplusplus forces MSVC to report the REAL C++ standard value instead of 199711
        target_compile_options(${TARGET_NAME} PRIVATE /std:c++latest /Zc:__cplusplus)
        # Fall back to cxx_std_23 in CMake's table so MSVC doesn't reject the build
        target_compile_features(${TARGET_NAME} PRIVATE cxx_std_23)
    else()
        target_compile_features(${TARGET_NAME} PRIVATE cxx_std_26)
    endif()
endfunction()

# 4. Global Compiler Warnings & Optimization Flags
add_library(project_warnings INTERFACE)
if(MSVC)
    target_compile_options(project_warnings INTERFACE /W4 /WX /fp:fast)
else()
    target_compile_options(project_warnings INTERFACE -Wall -Wextra -Wpedantic -Werror -ffast-math)
endif()

# 5. Hardware SIMD Detection & Flags
option(USE_AVX2 "Force enable AVX2 SIMD optimizations" OFF)
option(USE_NEON "Force enable ARM NEON SIMD optimizations" OFF)

add_library(simd_optimizations INTERFACE)
if(USE_AVX2 OR CMAKE_SYSTEM_PROCESSOR MATCHES "amd64|x86_64|x86")
    message(STATUS "Configuring SIMD target: x86_64 / AVX2 + FMA")
    if(MSVC)
        target_compile_options(simd_optimizations INTERFACE /arch:AVX2)
    else()
        target_compile_options(simd_optimizations INTERFACE -mavx2 -mfma)
    endif()
    target_compile_definitions(simd_optimizations INTERFACE CRESCENDO_USE_AVX2=1)
elseif(USE_NEON OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    message(STATUS "Configuring SIMD target: ARM64 / NEON")
    if(NOT MSVC)
        target_compile_options(simd_optimizations INTERFACE -mcpu=native)
    endif()
    target_compile_definitions(simd_optimizations INTERFACE CRESCENDO_USE_NEON=1)
endif()

# 6. Global Include Directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 7. Core Modular Libraries (Built from scratch, zero dependencies)

# Create the Tensor library containing SIMD, Tensor, Autograd, and AdamW modules
add_library(crescendo_tensor)
target_sources(crescendo_tensor
    PUBLIC
        FILE_SET CXX_MODULES
        FILES 
            src/tensor/tensor.cppm
            src/tensor/autograd.cppm
            src/tensor/simd_ops.cppm
            src/tensor/optimizer.cppm
)

target_link_libraries(crescendo_tensor PRIVATE project_warnings simd_optimizations)
target_enable_cxx26(crescendo_tensor)

# Digital Signal Processing (DSP) Library
add_library(crescendo_dsp STATIC
    src/dsp/griffin_lim.cpp
)

# Register the C++26 FFT Module file set directly to the DSP library
target_sources(crescendo_dsp
    PUBLIC
        FILE_SET CXX_MODULES
        FILES 
            src/dsp/fast_fourier_transform.cppm
            src/dsp/short_time_fourier_transform.cppm
            src/dsp/mel_filterbank.cppm
)
target_link_libraries(crescendo_dsp PUBLIC crescendo_tensor PRIVATE project_warnings simd_optimizations)
target_enable_cxx26(crescendo_dsp)

# Neural Network Models Library (U-Net, Attention, Diffusion, Separation)
add_library(crescendo_models STATIC
    src/models/diffusion.cpp
    src/models/separator.cpp
)

target_sources(crescendo_models
    PUBLIC
        FILE_SET CXX_MODULES
        FILES 
            src/models/convolution2d.cppm
            src/models/groupnorm.cppm
            src/models/attention.cppm
            src/models/unet.cppm
)
target_link_libraries(crescendo_models PUBLIC crescendo_dsp crescendo_tensor PRIVATE project_warnings)
target_enable_cxx26(crescendo_models)

# Audio I/O & Interactive Editor Engine
add_library(crescendo_audio STATIC
    src/audio/editor.cpp
    src/audio/wav_parser.cpp
    src/audio/m4a_codec.cpp
)
target_link_libraries(crescendo_audio PUBLIC crescendo_tensor PRIVATE project_warnings)
target_enable_cxx26(crescendo_audio)


# 8. Main CLI Executable
add_executable(crescendo
    src/main.cpp
)

# Attach libraries and enforce C++26 on the executable
target_link_libraries(crescendo PRIVATE 
    crescendo_models 
    crescendo_audio 
    crescendo_dsp 
    crescendo_tensor
    project_warnings
)
target_enable_cxx26(crescendo) # Applied MSVC-safe C++26

# 9. Test Suite Setup
enable_testing()

macro(add_crescendo_test TEST_NAME TEST_FILE)
    add_executable(${TEST_NAME} ${TEST_FILE})
    target_link_libraries(${TEST_NAME} PRIVATE crescendo_models crescendo_audio crescendo_dsp project_warnings)
    target_enable_cxx26(${TEST_NAME}) # Applied MSVC-safe C++26
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endmacro()

# Define your unit tests
add_crescendo_test(test_fft tests/test_fast_fourier_transform.cpp)
add_crescendo_test(test_autograd tests/test_autograd.cpp)
add_crescendo_test(test_audio_io tests/test_audio_io.cpp)