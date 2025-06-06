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

# Common CMake files are included from a top-level directory, but otherwise
# this directory can be built as a standalone project.
set(OTK_SOURCE_DIR ".." CACHE PATH "OptiX Toolkit source directory")

# Using the latest CMake is highly recommended, to ensure up-to-date CUDA language support.
cmake_minimum_required(VERSION 3.23 FATAL_ERROR)
include(${OTK_SOURCE_DIR}/CMake/Policies.cmake)

project(Util LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)
include(${OTK_SOURCE_DIR}/CMake/BuildConfig.cmake)

find_package( CUDAToolkit 10.0 REQUIRED )

# Note: 
# There are two ways to get OptiX headers below.
# FetchOptiX downloads the headers from github into the build dir
# FindOptiX lets you point to existing headers elsewhere on your system

set( ENABLE_OPTIX ON )
include("${OTK_SOURCE_DIR}/CMake/FetchOptiX.cmake")

# FindOptiX method
# Set OptiX_INSTALL_DIR to the root of the OptiX SDK when configuring CMake.
## set(OptiX_INSTALL_DIR "OptiX_INSTALL_DIR-NOTFOUND" CACHE PATH "Path to OptiX installed location.")
## list(APPEND CMAKE_MODULE_PATH ${OTK_SOURCE_DIR}/CMake)
## find_package(OptiX REQUIRED)

option( BUILD_SHARED_LIBS "Build using shared libraries" ON )

add_library(Util
  src/Exception.cpp
  )

if(NOT PROJECT_IS_TOP_LEVEL)
  add_library(OptiXToolkit::Util ALIAS Util)
endif()

target_sources(Util 
  PUBLIC 
  FILE_SET HEADERS 
  BASE_DIRS include
  FILES
  include/OptiXToolkit/Util/Exception.h
)

target_include_directories(Util PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>  # for Config.h
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )

target_link_libraries(Util 

    # expose OptiX header include path to our clients
    PUBLIC
    OptiX::OptiX

    PRIVATE
    ${CUDA_LIBS}
  )

set_target_properties(Util PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

install(TARGETS Util
  EXPORT UtilTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/OptiXToolkit
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/OptiXToolkit
  FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

install(EXPORT UtilTargets
  FILE UtilTargets.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OptiXToolkit
  NAMESPACE OptiXToolkit::
  )

configure_file( src/Config.h.in include/OptiXToolkit/Util/Config.h @ONLY )
install( FILES ${CMAKE_CURRENT_BINARY_DIR}/include/OptiXToolkit/Util/Config.h 
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/OptiXToolkit/Util
  )
