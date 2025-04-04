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

#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

// clang-format on

struct ImFont;
struct ImGuiContext;
struct ImPlotContext;
struct GLFWwindow;

namespace otk
{

class ImGuiRenderer
{
public:

    ~ImGuiRenderer();

    void init(GLFWwindow* window);

    bool keyboardUpdate(int key, int scancode, int action, int mods);
    bool keyboardCharInput(unsigned int unicode, int mods);
    bool mousePosUpdate(double xpos, double ypos);
    bool mouseScrollUpdate(double xoffset, double yoffset);
    bool mouseButtonUpdate(int button, int action, int mods);

    void animate(float elapsedTimeSeconds);
    void render();

    // ImGui & ImPlot declare contexts as global variables in their headers:
    // as expected, this does not work well across compiled modules. These
    // accessors can be used to force/reset the contexts in a module.
    ImGuiContext* getImGuiContext() const { return m_imgui; }
    ImPlotContext* getImPlotContext() const { return m_implot; }

    ImFont* getNVRgFont() { return m_nvidiaRgFont; }
    ImFont* getNVBldFont() { return m_nvidiaBldFont; }
    ImFont* getIconicFont() { return m_iconicFont; }

public:

    struct TimeLineEditorState
    {
        template <typename T> constexpr T clamp(T value, T lower, T upper) { return std::min(std::max(value, lower), upper); }

        enum class Playback : uint8_t { Pause = 0, Play } mode = Playback::Pause;
        bool loop = true;
        int2 frameRange = { 0, 0 };
        float frameRate = 30.f;
        float startTime = 0.f;
        float endTime = 0.f;
        float currentTime = 0.f;

        std::function<void(TimeLineEditorState const&)> playCallback;
        std::function<void(TimeLineEditorState const&)> pauseCallback;
        std::function<void(TimeLineEditorState const&)> setTimeCallback;

        void update(float elapsedTime);
        float animationTime() const { return currentTime - startTime; }

        // programmatic manipulation
        inline bool isPlaying() const { return mode == Playback::Play; }
        inline bool isPaused() const { return mode == Playback::Pause; }

        inline void setFrame(float time) { currentTime = clamp(time / frameRate, startTime, endTime); }
        inline void stepForward() { currentTime = clamp(currentTime + 1.f / frameRate, startTime, endTime); }
        inline void stepBackward() { currentTime = clamp(currentTime - 1.f / frameRate, startTime, endTime); }
        inline void rewind() { currentTime = startTime; }
        inline void fastForward() { currentTime = endTime; }
    };

protected:

    // misc UI helpers
    bool buildAzimuthElevationSliders(float3& direction, bool negative);

    bool buildTimeLineEditor(TimeLineEditorState& state, float2 size);

protected:

    // TTF font management and open-source 'iconic' font to make buttons
    ImFont* loadFontconst(char const* fontFile, float fontSize, uint16_t const* range);
    ImFont* addFontFromMemoryCompressedBase85TTF(const char* data, float fontSize, const uint16_t* range);
    void buildGlyphsTable(ImFont const* font);

    // rudimentary cross-platform browsers
    static bool folderDialog(std::string& filepath);
    static bool fileDialog(bool bOpen, char const* filters, std::string& filepath);

protected:
    
    // virtual interface

    virtual void buildUI() = 0;

    GLFWwindow* m_window = nullptr;
    ImGuiContext* m_imgui = nullptr;
    ImPlotContext* m_implot = nullptr;

    ImFont* m_nvidiaRgFont = nullptr;
    ImFont* m_nvidiaBldFont = nullptr;
    ImFont* m_iconicFont = nullptr;

private:

    static char const* getNVSansFontRgCompressedBase85TTF();
    static char const* getNVSansFontBoldCompressedBase85TTF();

    // 'iconic' open-source TTF font (lots of standard icons to make buttons with)
    static constexpr float const iconicFontSize = 18.f;
    static uint16_t const* getOpenIconicFontGlyphRange();
    static char const* getOpenIconicFontCompressedBase85TTF();

private:

    static constexpr int key_count = 348 + 1;

    std::array<bool, 3> m_mouseDown = { false };
    std::array<bool, key_count> m_keyDown = { false };
};

} // end namespace otk
