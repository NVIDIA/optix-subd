#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#


# Include developer's "private.cmake" (if any) for ad hoc customization.
include( "private.cmake" OPTIONAL )

# Minimum version of cmake to support OptiXIR compilation
cmake_minimum_required(VERSION 3.27.7 FATAL_ERROR)

# Temporarily quiet some warnings related to FetchContent usage
if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake")

include( optix-toolkit/CMake/Policies.cmake )

# Set the default build to Release.  Note this doesn't do anything for VS where 
# you configure the build type after generation
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif()


# default CUDA arch.  This needs to be declared before any project(... CUDA) lines
set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "")

project(OptixSubd LANGUAGES C CXX CUDA VERSION 1.13.0)

message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

# C++ standards

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CMakeDependentOption)


# Optionally download asset files.  Could be large, so off by default

option( ENABLE_ASSET_DOWNLOAD "Download scene files automatically during configure" OFF)
include("${CMAKE_SOURCE_DIR}/CMake/FetchAssets.cmake")


# Put all the runtime stuff in the same directory.  By default, CMake puts each targets'
# output into their own directory.  We want all the targets to be put in the same
# directory, and we can do this by setting these variables.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

include(GNUInstallDirs)

# Warning and error definitions

if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/we4099>)  # mixing struct/class forward decl
else()
    # Ignore warnings from some third party headers by including them with -isystem
    include_directories(SYSTEM
      ${CMAKE_CURRENT_LIST_DIR}/extern/osd_lite
      ${CMAKE_CURRENT_LIST_DIR}/extern/osd_lite/opensubdiv
      ${CMAKE_CURRENT_LIST_DIR}/extern/osd_lite/opensubdiv/tmr
      ${CMAKE_CURRENT_LIST_DIR}/texture
      ${CMAKE_BINARY_DIR}/_deps/implot-src
      ${CMAKE_BINARY_DIR}/_deps/imgui-src
    )

    set(CLUSTER_BENCH_COMPILE_WARNINGS
      $<$<COMPILE_LANGUAGE:CXX>:-Werror=mismatched-tags>  # mixing struct/class forward decl
      $<$<COMPILE_LANGUAGE:CXX>:-Werror=conversion>
      $<$<COMPILE_LANGUAGE:CXX>:-Werror=narrowing>
    )

    # STL bounds detection in GCC
    add_compile_options(
      $<$<CONFIG:Debug>:-D_GLIBCXX_DEBUG>
      $<$<CONFIG:Debug>:-D_GLIBCXX_DEBUG_BACKTRACE>
    )
endif()

# CUDA

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)

# Full kernel debug info (-G) has a big performance cost, keep it off for now.  It can also be 
# enabled for single targets in the CMakeLists files in subdirs.

#add_compile_options($<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>)
add_compile_options($<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo>)

add_compile_options($<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo>)


find_package( CUDAToolkit 12.0 REQUIRED )


# TBB (optional) needed for C++17 parallel algorithms on gcc
if ( NOT MSVC )
    find_package( TBB QUIET )
endif()

# OTK, including OptiX, glfw, etc

if(MSVC)
    add_definitions(-DNOMINMAX)
endif()

# workaround: use locally generated glad headers to avoid connection problems
add_library( glad SHARED
  extern/KHR/khrplatform.h
  extern/glad/glad.c
  extern/glad/glad.h
)
target_compile_definitions( glad
  PRIVATE GLAD_GLAPI_EXPORT_BUILD
  PUBLIC  GLAD_GLAPI_EXPORT )
target_include_directories( glad PUBLIC extern )
add_library(glad::glad ALIAS glad)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/optix-toolkit EXCLUDE_FROM_ALL)


# OSD
add_definitions(-D_USE_MATH_DEFINES)
set(OSD_LITE_FOLDER_NAME "external/osd_lite")
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/extern/osd_lite EXCLUDE_FROM_ALL)


# Build configuration


# Local targets

include(CMake/embed_cuda.cmake)

option(ENABLE_OPTIXIR "Enable OptiXIR compilation" ON)
if(ENABLE_OPTIXIR)
  set(CUDA_COMPILATION_OUTPUT_TYPE "OPTIXIR")
else()
  set(CUDA_COMPILATION_OUTPUT_TYPE "PTX")
endif()

set( _optix_src_files deviceCode.cu closestHit.cu )
set( _optix_hdr_files embeddedDeviceCode.h )
set( _optix_libraries OptiXToolkit::ShaderUtil OptiXToolkit::Util )
set( _optix_embedded_names embeddedDeviceCode embeddedDeviceCodeCH )


embed_cuda(
  RELOCATABLE
  OUTPUT_TARGET
    EmbeddedDeviceCode
  ${CUDA_COMPILATION_OUTPUT_TYPE}
  LIBRARIES
    ${_optix_libraries}
  INCLUDES
    "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES
    ${_optix_src_files}
  HEADER
    ${_optix_hdr_files}
  EMBEDDED_SYMBOL_NAMES
    ${_optix_embedded_names}
)

if (MSVC)
    target_compile_definitions(EmbeddedDeviceCode PRIVATE _USE_MATH_DEFINES)
    target_compile_definitions(EmbeddedDeviceCodeCuda PRIVATE _USE_MATH_DEFINES)
endif()

add_subdirectory(cluster_builder)
add_subdirectory(material)
add_subdirectory(motionvec)
add_subdirectory(profiler)
add_subdirectory(scene)
add_subdirectory(subdivision)
add_subdirectory(texture)
add_subdirectory(wireframe)

set( optixSubd_SOURCES )
list( APPEND optixSubd_SOURCES
    args.cpp
    args.h
    denoiserstub.h
    denoiserstub.cu
    GBuffer.cu
    GBuffer.h
    GBuffer.cuh
    optixSubd.cpp
    optixSubdApp.cpp
    optixSubdApp.h
    optixSubdGUI.cpp
    optixSubdGUI.h
    optixRenderer.cpp
    optixRenderer.h
    pipeline.cpp
    pipeline.h
    shadingTypes.h
    statistics.cpp
    statistics.h 
    utils.cuh
    utils.h
    version.cpp
)


add_executable( optixSubd
    ${optixSubd_SOURCES}
    $<TARGET_OBJECTS:cluster_builder>
    $<TARGET_OBJECTS:material>
    $<TARGET_OBJECTS:motionvec>
    $<TARGET_OBJECTS:profiler>
    $<TARGET_OBJECTS:scene>
    $<TARGET_OBJECTS:texture>
    $<TARGET_OBJECTS:wireframe>
)

target_include_directories( optixSubd PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_BINARY_DIR}/EmbeddedDeviceCode
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries( optixSubd
  EmbeddedDeviceCode
  OptiXToolkit::Gui
  OptiXToolkit::Util
  OptiXToolkit::ShaderUtil
  glfw
  glad
  imgui
  implot
  jsoncpp_static
  subdivision
  ${CUDA_LIBS}
  ${CMAKE_DL_LIBS}  
)


# Optionally link tbb for parallel C++17 algorithms on gcc
if ( TBB_FOUND )
    target_link_libraries( optixSubd TBB::tbb )
endif()


set_target_properties( optixSubd PROPERTIES CUDA_SEPARABLE_COMPILATION ON )

set_target_properties( optixSubd PROPERTIES INSTALL_RPATH ${OptiXToolkit_DIR}/../../OptiXToolkit )

if( WIN32 )

  # Set optixSubd as the startup project in MSVC
  set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT optixSubd)

  # Copy shared libraries that the built executable depends on.
  add_custom_command( TARGET optixSubd POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:optixSubd> $<TARGET_FILE_DIR:optixSubd>
    COMMAND_EXPAND_LISTS )
 
endif()

# Add individual per-target compile options because many third party
# dependencies are not compatible with more strict warnings.
foreach(TARGET
  #cluster_builder
  material
  motionvec
  profiler
  scene
  wireframe
  # texture
  optixSubd
  )
  target_compile_options(${TARGET} PRIVATE ${CLUSTER_BENCH_COMPILE_WARNINGS})
endforeach()


set_target_properties(glfw                PROPERTIES FOLDER "external/GLFW3")
set_target_properties(glad                PROPERTIES FOLDER "external/glad")
set_target_properties(imgui               PROPERTIES FOLDER "external/imgui")
set_target_properties(implot              PROPERTIES FOLDER "external/implot")
set_target_properties(jsoncpp_static      PROPERTIES FOLDER "external/jsoncpp")
set_target_properties(Gui                 PROPERTIES FOLDER "otk")
set_target_properties(ShaderUtil          PROPERTIES FOLDER "otk")
set_target_properties(Util                PROPERTIES FOLDER "otk")


#
# generate build stamp in ${CMAKE_CURRENT_BINARY_DIR}/version.h
#

add_custom_target(update_buildstamp
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE=${CMAKE_SOURCE_DIR}/version.h.in
        -DTARGET=${CMAKE_CURRENT_BINARY_DIR}/ocb_version.h
        -DPROJECT_CONFIG=$<CONFIG>
        -P "${CMAKE_SOURCE_DIR}/CMake/build_stamp.cmake"
    VERBATIM
)

add_dependencies(optixSubd update_buildstamp)


