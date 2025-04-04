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

if( TARGET imgui )
    return()
endif()

if (ENABLE_IMGUI)

    include(FetchContent)
    FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG v1.89.5
        )
    FetchContent_MakeAvailable(imgui)

    # Override Imgui build - we want a lean static library

    set(imgui_srcs
        ${CMAKE_BINARY_DIR}/_deps/imgui-src/imconfig.h
        ${CMAKE_BINARY_DIR}/_deps/imgui-src/imgui_draw.cpp
        ${CMAKE_BINARY_DIR}/_deps/imgui-src/imgui_tables.cpp
        ${CMAKE_BINARY_DIR}/_deps/imgui-src/imgui_widgets.cpp
        ${CMAKE_BINARY_DIR}/_deps/imgui-src/imgui_internal.h
        ${CMAKE_BINARY_DIR}/_deps/imgui-src/imgui.cpp
        ${CMAKE_BINARY_DIR}/_deps/imgui-src/imgui.h
        ${CMAKE_BINARY_DIR}/_deps/imgui-src/imstb_rectpack.h
        ${CMAKE_BINARY_DIR}/_deps/imgui-src/imstb_textedit.h
        ${CMAKE_BINARY_DIR}/_deps/imgui-src/imstb_truetype.h
        ${CMAKE_BINARY_DIR}/_deps/imgui-src/backends/imgui_impl_glfw.cpp
        ${CMAKE_BINARY_DIR}/_deps/imgui-src/backends/imgui_impl_glfw.h
        ${CMAKE_BINARY_DIR}/_deps/imgui-src/backends/imgui_impl_opengl3.cpp
        ${CMAKE_BINARY_DIR}/_deps/imgui-src/backends/imgui_impl_opengl3.h
    )

    add_library(imgui STATIC ${imgui_srcs})
    set_target_properties(imgui PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(imgui PUBLIC "${CMAKE_BINARY_DIR}/_deps/imgui-src/")
    target_link_libraries(imgui glfw)

endif()
