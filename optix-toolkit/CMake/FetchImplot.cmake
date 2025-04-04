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

if( TARGET implot )
    return()
endif()

if (ENABLE_IMPLOT)

    if (NOT TARGET imgui)
        message(FATAL_ERROR "Implot requires imgui")
    endif()

    include(FetchContent)
    FetchContent_Declare(
        implot
        GIT_REPOSITORY https://github.com/epezent/implot.git
        GIT_TAG v0.14
        )
    FetchContent_MakeAvailable(implot)

    # Override Imgui build - we want a lean static library

    set(implot_srcs
        ${CMAKE_BINARY_DIR}/_deps/implot-src/implot.cpp
        ${CMAKE_BINARY_DIR}/_deps/implot-src/implot.h
        ${CMAKE_BINARY_DIR}/_deps/implot-src/implot_internal.h
        ${CMAKE_BINARY_DIR}/_deps/implot-src/implot_items.cpp
    )

    add_library(implot STATIC ${implot_srcs})
    set_target_properties(implot PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_compile_definitions(implot PRIVATE IMGUI_DEFINE_MATH_OPERATORS)
    target_include_directories(implot PUBLIC "${CMAKE_BINARY_DIR}/_deps/implot-src/")
    target_link_libraries(implot imgui)
endif()
