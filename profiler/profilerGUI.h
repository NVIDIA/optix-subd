//
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// 
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// 
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

// clang-format off

#pragma once

#include <profiler/profiler.h>
#include <imgui.h>

struct ImPlotContext;

// clang-format on

namespace otk {
class ImGuiRenderer;
}

class ProfilerGUI
{
  public:

    // if fps >= 0 displays value in profiler controller window
    int fps = -1;

    // if ntris > 0 displays value in profiler controller window 
    size_t ntris = 0;

    uint32_t streaming = 0;

    struct ControllerWindow
    {
        ImVec2    pos   = ImVec2( 0, 0 );
        ImGuiCond cond  = ImGuiCond( 0 );
        ImVec2    pivot = ImVec2( 0, 0 );
        ImVec2    size  = ImVec2( 115, 0 );
    } controllerWindow;

    struct ProfilingWindow
    {
        ImVec2    pos   = ImVec2( 0, 0 );
        ImGuiCond cond  = ImGuiCond_FirstUseEver;
        ImVec2    pivot = ImVec2( 1, 0 );
        ImVec2    size  = ImVec2( 800, 250 );
    } profilingWindow;

  public:

    template <typename... SamplerGroup>
    void buildUI( otk::ImGuiRenderer& renderer, SamplerGroup const&... groups );

    bool m_displayGraphWindow = true;

    static int humanFormatter( double value, char* buff, int bufsize, void* = nullptr );
    static int metricFormatter( double value, char* buff, int bufsize, void* data );
    static int megabytesFormatter( double value, char* buff, int bufsize, void* = nullptr );
    static int memoryFormatter( double value, char* buff, int bufsize, void* = nullptr );

  private:

    void buildControllerUI( otk::ImGuiRenderer& renderer );

    void buildFrequencySelectorUI();
};

template <typename... SamplerGroup>
inline void ProfilerGUI::buildUI( otk::ImGuiRenderer& renderer, SamplerGroup const&... groups )
{
    // Synchronize with device streams so timers can be polled safely by UI elements
    Profiler::get().frameSync();

    buildControllerUI( renderer );

    if( m_displayGraphWindow )
    {
        ImGui::SetNextWindowPos( profilingWindow.pos, profilingWindow.cond, profilingWindow.pivot );
        ImGui::SetNextWindowSize( profilingWindow.size, profilingWindow.cond );
        ImGui::SetNextWindowBgAlpha( .65f );

        if( ImGui::Begin( "Profiler", &m_displayGraphWindow, ImGuiWindowFlags_None ) )
        {
            buildFrequencySelectorUI();

            if( ImGui::BeginTabBar( "MyTabBar", ImGuiTabBarFlags_Reorderable ) )
            {
                ImVec2 tabSize = profilingWindow.size;

                (
                    [&] {
                        if( ImGui::BeginTabItem( groups.name.c_str() ) )
                        {
                            groups.buildUI( renderer );

                            ImGui::EndTabItem();
                        }
                    }(),
                    ... );
                ImGui::EndTabBar();
            }
        }
        ImGui::End();
    }
}
