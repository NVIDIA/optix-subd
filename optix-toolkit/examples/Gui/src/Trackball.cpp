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

#include <OptiXToolkit/Gui/Camera.h>
#include <OptiXToolkit/Gui/Trackball.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <GLFW/glfw3.h>

#include <array>
#include <cassert>
#include <cmath>
#include <algorithm>

namespace otk
{

namespace
{
    inline float radians(float degrees) { return degrees * M_PIf / 180.0f; }
    inline float degrees(float radians) { return radians * M_1_PIf * 180.0f; }

    // keyboard

    enum class KeyboardControls : uint8_t
    {
        MoveUp = 0,
        MoveDown,
        MoveLeft,
        MoveRight,
        MoveForward,
        MoveBackward,
        RollLeft,
        RollRight,
        SpeedUp,
        SlowDown,
        OrbitMode,
        Count
    };

    static std::array<bool, int(KeyboardControls::Count)> keyboardState = { false };

    // mouse

    enum class MouseButtons : uint8_t
    {
        Left = 0,
        Middle,
        Right,
        Count
    };

    static std::array<bool, int(MouseButtons::Count)> mouseButtonsState = { false };

    bool userInput = false;
} // namespace


void Trackball::keyboardUpdate(int key, int code, int action, int mods)
{
    static const std::array<std::pair<int, int>, 18> keyboardMap = {{
        { GLFW_KEY_Q,             int(KeyboardControls::MoveDown) },
        { GLFW_KEY_E,             int(KeyboardControls::MoveUp) },
        { GLFW_KEY_A,             int(KeyboardControls::MoveLeft) },
        { GLFW_KEY_D,             int(KeyboardControls::MoveRight) },
//        { GLFW_KEY_DOWN,          int(KeyboardControls::MoveDown) },
//        { GLFW_KEY_UP,            int(KeyboardControls::MoveUp) },
//        { GLFW_KEY_LEFT,          int(KeyboardControls::MoveLeft) },
//        { GLFW_KEY_RIGHT,         int(KeyboardControls::MoveRight) },
        { GLFW_KEY_W,             int(KeyboardControls::MoveForward) },
        { GLFW_KEY_S,             int(KeyboardControls::MoveBackward) },
        { GLFW_KEY_Z,             int(KeyboardControls::RollLeft) },
        { GLFW_KEY_X,             int(KeyboardControls::RollRight) },
        { GLFW_KEY_LEFT_SHIFT,    int(KeyboardControls::SpeedUp) },
        { GLFW_KEY_RIGHT_SHIFT,   int(KeyboardControls::SpeedUp) },
        { GLFW_KEY_LEFT_CONTROL,  int(KeyboardControls::SlowDown) },
        { GLFW_KEY_RIGHT_CONTROL, int(KeyboardControls::SlowDown) },
        { GLFW_KEY_LEFT_ALT,      int(KeyboardControls::OrbitMode) },
        { GLFW_KEY_RIGHT_ALT,     int(KeyboardControls::OrbitMode) },
    }};

    auto it = std::find_if(keyboardMap.begin(), keyboardMap.end(), 
        [&key](std::pair<int, int> control) { return control.first == key; });

    if (it != keyboardMap.end())
        keyboardState[it->second] = (action == GLFW_PRESS || action == GLFW_REPEAT);
}

void Trackball::mouseButtonUpdate(int button, int action, int)
{
    static const std::array<std::pair<int, int>, 3> mouseButtonsdMap = {{
        { GLFW_MOUSE_BUTTON_LEFT,    int(MouseButtons::Left) },
        { GLFW_MOUSE_BUTTON_MIDDLE,  int(MouseButtons::Middle) },
        { GLFW_MOUSE_BUTTON_RIGHT,   int(MouseButtons::Right) },
    }};

    assert(action == GLFW_PRESS || action == GLFW_RELEASE);

    auto it = std::find_if(mouseButtonsdMap.begin(), mouseButtonsdMap.end(),
        [&button](std::pair<int, int> control) { return control.first == button; });

    if (it != mouseButtonsdMap.end())
    {
        mouseButtonsState[it->second] = action == GLFW_PRESS;
        if (action == GLFW_RELEASE)
            m_performTracking = false;
    }
}

bool otk::Trackball::animate(float deltaT)
{

    // Update far plane to cover scene + camera
    {
        otk::Aabb aabb = m_sceneAabb;
        aabb.include( m_camera->getEye() );
        m_camera->setFar( 1.1f * length( aabb.extent() ) );
    }

    bool result = userInput;

    if (!m_roamMode)
    {
        userInput = false;
        return result;
    }

    float moveSpeed = deltaT * m_moveSpeed * m_moveSpeedMultiplier;
    float rollSpeed = deltaT * m_rollSpeed;

    if (keyboardState[int(KeyboardControls::SpeedUp)])
    {
        moveSpeed *= 3.f;
        rollSpeed *= 3.f;
    }
 
    if (keyboardState[int(KeyboardControls::SlowDown)])
    {
        moveSpeed *= .1f;
        rollSpeed *= .1f;
    }

    auto const& basis = m_camera->getBasis();

    if (keyboardState[int(KeyboardControls::MoveForward)])
    {   m_camera->translate(m_camera->getDirection() * moveSpeed); result |= true; }
    if (keyboardState[int(KeyboardControls::MoveBackward)])
    {   m_camera->translate(-m_camera->getDirection() * moveSpeed); result |= true; }

    if (keyboardState[int(KeyboardControls::MoveLeft)])
    {   m_camera->translate(-normalize(basis[0]) * moveSpeed); result |= true; }
    if (keyboardState[int(KeyboardControls::MoveRight)])
    {   m_camera->translate(normalize(basis[0]) * moveSpeed); result |= true; }

    if (keyboardState[int(KeyboardControls::MoveUp)])
    {   m_camera->translate(normalize(basis[1]) * moveSpeed); result |= true; }
    if (keyboardState[int(KeyboardControls::MoveDown)])
    {   m_camera->translate(-normalize(basis[1]) * moveSpeed); result |= true; }

    if (keyboardState[int(KeyboardControls::RollLeft)])
    {   m_camera->roll(rollSpeed); result |= true; }
    if (keyboardState[int(KeyboardControls::RollRight)])
    {   m_camera->roll(-rollSpeed); result |= true; }

    userInput = false;
    return result;
}

void Trackball::mouseTrackingUpdate(int2 pos, int2 canvasSize)
{
    if (!m_performTracking)
    {
        reinitOrientationFromCamera();
        m_performTracking = true;
        return;
    }

    m_delta = pos - m_prevPos;
    m_prevPos = pos;
    m_pos = pos;
    
    updateCamera(pos, canvasSize);
}

bool Trackball::mouseWheelUpdate(int dir)
{
    zoom(dir);
    return true;
}

float3 Trackball::getCameraDirection() const
{
    // use lat/long for view definition
    float3 localDir;
    localDir.x = cos(m_latitude) * sin(m_longitude);
    localDir.y = cos(m_latitude) * cos(m_longitude);
    localDir.z = sin(m_latitude);

    return  m_u * localDir.x + m_v * localDir.y + m_w * localDir.z;
}

void Trackball::applyGimbalLock()
{
    if (!m_gimbalLock) {
        reinitOrientationFromCamera();
        if (m_camera->hasChanged())
            m_camera->setUp(m_w);
    }
}

void Trackball::updateCamera(int2 pos, int2 canvas)
{
    if (m_roamMode == false || keyboardState[int(KeyboardControls::OrbitMode)])
    {
        if (mouseButtonsState[int(MouseButtons::Left)])
        {
            m_latitude = radians(std::min(89.0f, std::max(-89.0f, degrees(m_latitude) + 0.5f * m_delta.y)));
            m_longitude = radians(fmod(degrees(m_longitude) - 0.5f * m_delta.x, 360.0f));

            float3 dirWS = getCameraDirection();

            m_camera->setEye(m_camera->getLookat() + dirWS * m_cameraEyeLookatDistance);

            applyGimbalLock();
            userInput |= true;
        }
        else if (mouseButtonsState[int(MouseButtons::Middle)])
        {
            float2 delta = { float(m_delta.x)/float(canvas.x), float(m_delta.y)/float(canvas.y) };
            m_camera->pan(delta);
            userInput |= true;
        }
        else if (mouseButtonsState[int(MouseButtons::Right)])
        {
            float factor = float(m_delta.x) / float(canvas.x)
                         + float(m_delta.y) / float(canvas.y);
            constexpr float const dollySpeed = 2.f;
            m_camera->dolly(1.f - dollySpeed * factor);
            userInput |= true;
        }
    }
    else if (m_roamMode == true)
    {
        if (mouseButtonsState[int(MouseButtons::Left)])
        {
            m_latitude = radians(std::min(89.0f, std::max(-89.0f, degrees(m_latitude) + 0.5f * m_delta.y)));
            m_longitude = radians(fmod(degrees(m_longitude) - 0.5f * m_delta.x, 360.0f));

            float3 dirWS = getCameraDirection();

            m_camera->setLookat(m_camera->getEye() - dirWS * m_cameraEyeLookatDistance);

            applyGimbalLock();
            userInput |= true;
        }
    }
}

void Trackball::setReferenceFrame(const float3& u, const float3& v, const float3& w)
{
    m_u = u;
    m_v = v;
    m_w = w;
    float3 dirWS = -m_camera->getDirection();
    float3 dirLocal;
    dirLocal.x = dot(dirWS, u);
    dirLocal.y = dot(dirWS, v);
    dirLocal.z = dot(dirWS, w);
    m_longitude = atan2(dirLocal.x, dirLocal.y);
    m_latitude = asin(dirLocal.z);
}

void Trackball::zoom(int direction)
{
    float zoom = (direction > 0) ? 1 / m_zoomMultiplier : m_zoomMultiplier;
    m_cameraEyeLookatDistance *= zoom;
    const float3& lookat = m_camera->getLookat();
    const float3& eye = m_camera->getEye();
    m_camera->setEye(lookat + (eye - lookat) * zoom);
}

void Trackball::reinitOrientationFromCamera()
{
    auto const& basis = m_camera->getBasis();
    m_u =  normalize(basis[0]);
    m_v =  normalize(basis[1]);
    m_w = -normalize(basis[2]);
    std::swap(m_v, m_w);
    m_latitude = 0.0f;
    m_longitude = 0.0f;
    m_cameraEyeLookatDistance = length(m_camera->getLookat() - m_camera->getEye());
}

} // namespace otk
