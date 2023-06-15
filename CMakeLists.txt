cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

option(DEBUG_MODE_ENC "debug enc computation result" OFF)
option(DEBUG_MODE_DEC "debug dec computation result" OFF)
option(DEBUG_RESULT_ATTENTION "debug enc attention computation result" ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)

# Require C++14 and disable compiler-specific extensions
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(DEFAULT_BUILD_TYPE "Release")

#-------------------------------------------------------------------------------
# Project name and version
#-------------------------------------------------------------------------------
project(test_gemm_i8 LANGUAGES C CXX CUDA)
set(${PROJECT_NAME}_VERSION 0.0.0)

# Prohibit in-source build
if (${PROJECT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR})
    message(FATAL_ERROR
            "test_gemm_i8 does not support in-source CMake builds at this time.")
endif (${PROJECT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR})


# Enable languages.
enable_language(CXX C)

# Use ${CMAKE_CXX_COMPILER} as the cuda host compiler.
if (NOT CMAKE_CUDA_HOST_COMPILER)
	set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
endif()
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

### Find Required Libs ###
find_package(CUDA REQUIRED)
find_package(CUDAToolkit)

# Find CUDA ARCH VERSION
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
SET( ${CUDA_ARCH_LIST})
list (GET CUDA_ARCH_LIST 0 ARCH_NUMBER)
SET(CUDA_ARCH sm_${ARCH_NUMBER})
SET(CUDA_COMPUTE compute_${ARCH_NUMBER})


message(--------------------------------------------)
message(CUDA_ARCH = ${CUDA_ARCH})
message(CUDA_COMPUTE = ${CUDA_COMPUTE})
message(--------------------------------------------)



set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)

set(CUDA_HEADER_DIRS ${CUDA_PATH}/include)
set(CUDA_LIB_DIRS ${CUDA_PATH}/lib64)


set(CMAKE_CUDA_RUNTIME_LIBRARY "Static")

set(CUDA_FLAGS "--expt-extended-lambda --expt-relaxed-constexpr --keep --verbose --compiler-options -fPIC -g -v -G -Xcompiler -Wall")
set(CUDA_FLAGS "-gencode code=${CUDA_ARCH},arch=${CUDA_COMPUTE} ${CUDA_FLAGS}")
string(REGEX REPLACE "^sm_" "" ARCH_NUMBER "${CUDA_ARCH}")

if ("${CUDA_ARCH}" STREQUAL "sm_${ARCH_NUMBER}")
	set(CMAKE_CUDA_ARCHITECTURES "${ARCH_NUMBER}")
else()
	message(FATAL_ERROR "Unknown CUDA_ARCH: ${CUDA_ARCH}")
endif()

message(STATUS "Using CUDA architecture: ${CUDA_ARCH}")

set(CUDA_FLAGS "-ccbin=${CMAKE_CXX_COMPILER} ${CUDA_FLAGS}")
set(CMAKE_CUDA_HOST_LINK_LAUNCHER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_FLAGS "${CUDA_FLAGS}" CACHE STRING "CUDA flags set for test_gemm_i8" FORCE)

#-------------------------------------------------------------------------------
# Process configuration options
#-------------------------------------------------------------------------------
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -no-pie -lpthread")


message(STATUS "test_gemm_i8 build type: CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
message(STATUS "test_gemm_i8 version: v${test_gemm_i8_VERSION_STRING}")

set(MAIN_FILE ${PROJECT_SOURCE_DIR}/test_gemm_i8.cu)
# Declaring the library
set(ALL_SRC ${MAIN_FILE})
foreach(FNAME IN LISTS ALL_SRC)
    message(${FNAME})
endforeach()


add_executable(test_gemm_i8 ${ALL_SRC})
target_include_directories(test_gemm_i8 PRIVATE ${CUDA_HEADER_DIRS} SYSTEM ${CUDA_INCLUDE_DIRS} ${FMHA_HDR_FILES_DIR} ${HDR_FILES_DIR})
target_link_directories(test_gemm_i8 PRIVATE ${CUDA_LIB_DIRS})
target_link_libraries (test_gemm_i8 PRIVATE ${CUDA_LIBRARIES} )
set_target_properties(test_gemm_i8 PROPERTIES POSITION_INDEPENDENT_CODE ON)