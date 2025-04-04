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

#include "./profilerGUI.h"

#include <OptiXToolkit/Gui/ImguiRenderer.h>

#include <imgui_internal.h>
#include <implot.h>
#include <implot_internal.h>

#include <cassert>
#include <cmath>
#include <type_traits>
#include <variant>

// clang-format on

int ProfilerGUI::humanFormatter( double value, char* buff, int bufsize, void* )
{
    static char const* scale[] = { "", "K", "M", "B", "T", "Q", };

    int ndigits = static_cast<int>( value == 0 ? 0 : 1 + std::floor( log10l( std::abs( value ) ) ) );

    int exp = ndigits <= 4 ? 0 : 3 * ((ndigits - 1) / 3);

    if( ( exp / 3 ) >= std::size( scale ) )
        return false;

    double n = static_cast<double>( value / powl( 10, exp ) );

    bool decimals = value - n == 0. ? false : true;

    return std::snprintf( buff, bufsize, decimals ? "%5.1f%s" : "%5.0f%s", n, scale[exp / 3] );
}

int ProfilerGUI::metricFormatter( double value, char* buff, int bufsize, void* data )
{
    const char* unit = (const char*)data;
    static double v[] = { 1000000000,1000000,1000,1,0.001,0.000001,0.000000001 };
    static const char* p[] = { "G","M","k","","m","u","n" };
    if (value == 0) {
        return snprintf(buff, bufsize, "0 %s", unit);
    }
    for (int i = 0; i < 7; ++i) {
        if (fabs(value) >= v[i]) {
            return snprintf(buff, bufsize, "%g %s%s", value / v[i], p[i], unit);
        }
    }
    return snprintf(buff, bufsize, "%g %s%s", value / v[6], p[6], unit);
}

int ProfilerGUI::megabytesFormatter( double value, char* buff, int bufsize, void* ) 
{

    double mbsize = value / (1024 * 1024);
    return snprintf(buff, bufsize, "%8.1f MB", mbsize);
}

int ProfilerGUI::memoryFormatter( double value, char* buff, int bufsize, void* )
{
    static char const* suffixes[] = { "B", "KB",  "MB",  "GB", "TB" };

    uint8_t s = 0;
    for( ; value >= 1024; ++s )
        value /= 1024;

    assert( s < std::size( suffixes ) );

    if( value - std::floor(value) == 0. )
        snprintf( buff, bufsize, "% 9d %s", (int)value, suffixes[s] );
    else
        snprintf( buff, bufsize, "%8.1f %s", value, suffixes[s] );
    return 0;
}

// expects a [0, 1] normalized value 
static ImVec4 heatmapColor( float value )
{
    static float3 colors[] = { {0.f, 1.f, 0.f}, { 1., 1.f, 0.f}, { 1.f, 0.f, 0.f } };

    uint8_t i0 = 0;
    uint8_t i1 = 0;
    float f = 0.f;

    if( value <= 0.f )
        return ImVec4( .5f, .5f, .5f, 1.f );
    else if( value >= 1.f )
        i0 = i1 = std::size( colors ) - 1;
    else
    {
        f = value * ( std::size( colors ) - 1 );
        i0 = (uint8_t)std::floor( f );
        i1 = i0 + 1;
        f = f - float( i0 );
    }

    float3 c = colors[i0] + f * ( colors[i1] - colors[i0] );
    return ImVec4( c.x, c.y, c.z, 1.f );
}

void ProfilerGUI::buildControllerUI( otk::ImGuiRenderer& renderer )
{
    ImGui::SetNextWindowPos( controllerWindow.pos, controllerWindow.cond, controllerWindow.pivot );
    ImGui::SetNextWindowSize( controllerWindow.size );
    ImGui::SetNextWindowBgAlpha( .65f );

    ImGui::Begin( "ProfilerController", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar );

    ImVec2 itemSize = ImGui::GetItemRectSize();

    {
        ImVec4 color = heatmapColor( float( streaming ) / 1000.f );
        ImGui::PushStyleColor( ImGuiCol_Text, color );
        ImGui::PushFont( renderer.getIconicFont() );
        ImGui::Text( "%s", (char const*)(u8"\ue094") );
        ImGui::PopFont();
        ImGui::PopStyleColor();
        if( ImGui::IsItemHovered() )
            ImGui::SetTooltip( "%d pages", streaming );
    }

    ImGui::SameLine( 32.f );
    
    if( ntris > 0 )
    {
        static char buf[50];
        if( humanFormatter( static_cast<double>( ntris ), buf, sizeof( buf ) ) )
            ImGui::Text( "Tris %s", buf);
        else
            ImGui::Text("Too many !");
    }

    ImGui::PushFont( renderer.getIconicFont() );

    bool buttonState = m_displayGraphWindow;
    if( buttonState )
        ImGui::PushStyleColor( ImGuiCol_Button, ImVec4( 1.f, 0.f, 0.f, 1.f ) );
    if( ImGui::Button( (char const*)( u8"\ue0ae" "## controller button" ), {0.f, itemSize.y} ) )
    {
        m_displayGraphWindow = !buttonState;
    }
    if( buttonState )
        ImGui::PopStyleColor();

    ImGui::PopFont();

    ImGui::SameLine( 32.f );

    if( fps >= 0 )
        ImGui::Text( "FPS   % 5d", fps );
    else
        ImGui::Text( "FPS    ----" );

    controllerWindow.size = ImGui::GetWindowSize();

    ImGui::End();

    ImPlot::SetCurrentContext( renderer.getImPlotContext() );
}

void ProfilerGUI::buildFrequencySelectorUI()
{
    Profiler& profiler = Profiler::get();

    int rate = 0;

    if (profiler.recordingFrequency >= 120)
        rate = 5;
    else if (profiler.recordingFrequency >= 60)
        rate = 4;
    else if (profiler.recordingFrequency >= 30)
        rate = 3;
    else if (profiler.recordingFrequency >= 10)
        rate = 2;
    else if (profiler.recordingFrequency >= 1)
        rate = 1;

    ImVec2 size = ImGui::GetWindowSize();

    ImGui::SameLine(size[0] - (64 + 10));
    ImGui::PushItemWidth(64);
    if (ImGui::Combo("##SamplingFrequency", &rate, "---Hz\0001Hz\00010Hz\00030Hz\00060Hz\000120Hz\0"))
    {
        switch (rate)
        {
            case 0: profiler.recordingFrequency = -1; break;
            case 1: profiler.recordingFrequency = 1; break;
            case 2: profiler.recordingFrequency = 10; break;
            case 3: profiler.recordingFrequency = 30; break;
            case 4: profiler.recordingFrequency = 60; break;
            case 5: profiler.recordingFrequency = 120; break;
        }
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Profiling rate: frequency (in Hertz) at which samples are recorded each second\n"
            "or unconstrained records every frame.");
}

