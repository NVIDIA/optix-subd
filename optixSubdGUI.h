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

#include "shadingTypes.h"

#include <profiler/profilerGUI.h>

#include <OptiXToolkit/Gui/ImguiRenderer.h>

#include <filesystem>
#include <map>
#include <set>

// clang-format on

class OptixSubdApp;
struct GLFWindow;


struct MediaAsset {

    enum class Type : uint8_t {
        OBJ_FILE = 0,
        OBJ_SEQUENCE
    } type = Type::OBJ_FILE;
    
    char const* name = nullptr;

    std::string sequenceName;   // decorated sequence name to display in GUI
    std::string sequenceFormat; // format string to generate paths to individual files
    int padding = 0;            // number of digits in sequence numbers (or 0 if no padding detected)


    int2 frameRange = { std::numeric_limits<int>::max(), std::numeric_limits<int>::min() };
    float frameRate = 24.f;

    bool isSequence() const 
    { 
        return type == Type::OBJ_SEQUENCE; 
    }

    char const* getName() const
    {
        return isSequence() ? sequenceName.c_str() : name;
    }

    void growFrameRange(int frame)
    {
        frameRange.x = std::min(frame, frameRange.x);
        frameRange.y = std::max(frame, frameRange.y);
    };
};

typedef std::map<std::string, MediaAsset> MediaAssetsMap;

struct UIData
{

    bool vsync = false;

    bool showUI = true;
    bool showOverlay = false;

    // path to imgui.ini settings file (auto-saved by imgui)
    std::string iniFilepath;

    MediaAssetsMap mediaAssets;
    
    MediaAsset const* currentAsset = nullptr;

    void selectCurrentAsset( const std::string& name )
    {
        currentAsset = nullptr;
        if( name.empty() )
            return;
        if( auto it = mediaAssets.find( name ); it != mediaAssets.end() )
            currentAsset = &it->second;
    }

    std::array<char const*, 3> formatFilters() const;

    bool  showTimeLineEditor       = true;
    bool  showRecommendationWindow = false;
    bool  showHelpWindow           = true;

    otk::ImGuiRenderer::TimeLineEditorState timeLineEditorState;

    void togglePlayPause();

};

class OptixSubdGUI : public otk::ImGuiRenderer
{
public:
    OptixSubdGUI( OptixSubdApp& app, UIData& uidata );

    ~OptixSubdGUI();

    void animate( float elapsedTimeSeconds );

    virtual void buildUI() override;

    OptixSubdApp& getApp() { return m_app; }

    void setAnimationRange( int2 frameRange, float frameRate );

    void init( GLFWwindow* window );

    UIData& getUIData() { return m_ui; }

    ProfilerGUI& getProfilerGUI() { return m_profiler; }

    static constexpr ImVec4 const nvidia_green = ImVec4( (118.f / 255.f), (185.f / 255.f), 0.f, 1.f );

private:

    size_t computeTriangleCount() const;

    void buildUI_cameraSettings();
    void buildUI_help( int2 window_size, bool* display_help_window );
    void buildUI_timeline( int2 window_size, float timeline_width );
    void buildUI_overlay( int2 window_size );

    void buildUI_internal( int2 window_size );

    void loadAsset( MediaAsset const& asset, std::string const& name );

    UIData& m_ui;

    OptixSubdApp& m_app;

    ProfilerGUI m_profiler;

};
