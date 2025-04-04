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
set(OTK_SOURCE_DIR "../.." CACHE PATH "OptiX Toolkit source directory")

# Using the latest CMake is highly recommended, to ensure up-to-date CUDA language support.
cmake_minimum_required(VERSION 3.23 FATAL_ERROR)
include("${OTK_SOURCE_DIR}/CMake/Policies.cmake")

project(Gui LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

option( ENABLE_IMGUI "Enable support for Imgui renderer" ON)
option( ENABLE_IMPLOT "Supplement Imgui widgets with Implot real-time graphing library" ON)
option( BUILD_SHARED_LIBS "Build using shared libraries" ON )

# Temporarily quiet some warnings related to FetchContent usage
if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()

include(GNUInstallDirs)
include("${OTK_SOURCE_DIR}/CMake/BuildConfig.cmake")
include("${OTK_SOURCE_DIR}/CMake/FetchGlfw.cmake")
include("${OTK_SOURCE_DIR}/CMake/FetchGlad.cmake")
include("${OTK_SOURCE_DIR}/CMake/FetchImgui.cmake")
include("${OTK_SOURCE_DIR}/CMake/FetchImplot.cmake")

if(PROJECT_IS_TOP_LEVEL)
  find_package(OptiXToolkit REQUIRED)
endif()

set(src_files
  src/Camera.cpp
  src/GLCheck.cpp
  src/GLDisplay.cpp
  src/Trackball.cpp
  src/Window.cpp
  src/stb_image_write.h
)

set(public_link_libraries OptiXToolkit::ShaderUtil OptiXToolkit::Util)

set(private_link_libraries glfw glad::glad imgui)

if (ENABLE_IMGUI)
  list(APPEND src_files src/ImguiRenderer.cpp include/OptiXToolkit/Gui/ImguiRenderer.h)
  list(APPEND public_link_libraries imgui)
endif()

if (ENABLE_IMPLOT)
  list(APPEND public_link_libraries implot)
endif()

add_library(Gui ${src_files})

if(NOT PROJECT_IS_TOP_LEVEL)
  add_library(OptiXToolkit::Gui ALIAS Gui)
endif()

target_sources(Gui 
  PUBLIC 
  FILE_SET HEADERS 
  BASE_DIRS include
  FILES
  include/OptiXToolkit/Gui/CUDAOutputBuffer.h
  include/OptiXToolkit/Gui/Camera.h
  include/OptiXToolkit/Gui/GLCheck.h
  include/OptiXToolkit/Gui/GLDisplay.h
  include/OptiXToolkit/Gui/Trackball.h
  include/OptiXToolkit/Gui/Window.h
)

target_include_directories(Gui PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

target_link_libraries(Gui PUBLIC ${public_link_libraries} PRIVATE ${private_link_libraries})

set_target_properties(Gui PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

