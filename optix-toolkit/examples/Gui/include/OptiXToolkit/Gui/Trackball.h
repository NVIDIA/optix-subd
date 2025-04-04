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

#pragma once

#include <array>

#include <OptiXToolkit/ShaderUtil/Aabb.h>

struct GLFWgamepadstate;

namespace otk
{

class Camera;


// The Trackball polls input devices to remote-control a Camera
// 
// Only a single of each keyboard/mice/gamepad can be controlled,
// so this should be refactored as a singleton (or extended to
// support multiple controllers)
class Trackball
{
public:

    // Input device polling functions
    void keyboardUpdate(int key, int code, int action, int mods);
    void mouseButtonUpdate(int button, int action, int mods);
    void mouseTrackingUpdate(int2 pos, int2 canvasSize);
    bool mouseWheelUpdate(int dir);

    // Set the camera that will be controlled by the trackball.
    // Warning, this also initializes the reference frame of the trackball from the camera.
    // The reference frame defines the orbit's singularity.
    inline void setCamera(Camera* camera) { m_camera = camera; reinitOrientationFromCamera(); }
    inline const Camera* currentCamera() const { return m_camera; }

    // Apply user inputs to the controlled camera
    // Returns true if the user has interacted with any of the input 
    // devices since the previous call to the function.
    bool animate(float deltaT);

    // Input preferences

    void zoom(int direction);
    
    float moveSpeed() const { return m_moveSpeed; }
    void setMoveSpeed(float speed) { m_moveSpeed = speed; }

    float moveSpeedMultiplier() const { return m_moveSpeedMultiplier; }
    void setMoveSpeedMultiplier(float mult) { m_moveSpeedMultiplier = mult; }

    float rollSpeed() const { return m_rollSpeed; }
    void setRollSpeed(float speed) { m_rollSpeed = speed; }

    // Setting the gimbal lock to 'on' will fix the reference frame (i.e., the singularity of the trackball).
    // In most cases this is preferred.
    // For free scene exploration the gimbal lock can be turned off, which causes the trackball's reference frame
    // to be update on every camera update (adopted from the camera).
    bool gimbalLock() const { return m_gimbalLock; }
    void setGimbalLock(bool val) { m_gimbalLock = val; }

    // Adopts the reference frame from the camera.
    // Note that the reference frame of the camera usually has a different 'up' than the 'up' of the camera.
    // Though, typically, it is desired that the trackball's reference frame aligns with the actual up of the camera.
    void reinitOrientationFromCamera();

    // Specify the frame of the orbit that the camera is orbiting around.
    // The important bit is the 'up' of that frame as this is defines the singularity.
    // Here, 'up' is the 'w' component.
    // Typically you want the up of the reference frame to align with the up of the camera.
    // However, to be able to really freely move around, you can also constantly update
    // the reference frame of the trackball. This can be done by calling reinitOrientationFromCamera().
    // In most cases it is not required though (set the frame/up once, leave it as is).
    void setReferenceFrame(const float3& u, const float3& v, const float3& w);

    // In 'roam' mode, the mouse moves the camera in first-person mode and the 'alt'
    // key must be held to move in third-person mode (orbit). 'roam' mode off disables
    // first person movement (keyboard & mouse), but orbiting no longer requires holding
    // the 'alt' key.
    void setRoamMode(bool roam) { m_roamMode = roam; }
    bool getRoamMode() const { return m_roamMode; }

    void setSceneAabb( const otk::Aabb& aabb ) { m_sceneAabb = aabb; }

private:

    float3 getCameraDirection() const;

    void applyGimbalLock();

    void updateCamera(int2 pos, int2 canvasSize);

private:
    bool         m_roamMode                 = false;
    bool         m_gimbalLock               = true;
    Camera*      m_camera                   = nullptr;
    float        m_cameraEyeLookatDistance  = 0.0f;
    float        m_zoomMultiplier           = 1.1f;
    float        m_moveSpeed                = 1.0f;
    float        m_moveSpeedMultiplier      = 1.0f;
    float        m_rollSpeed                = 180.f / 5.f;

    // for updating camera zFar 
    otk::Aabb    m_sceneAabb;

    float        m_latitude                 = 0.0f;   // in radians
    float        m_longitude                = 0.0f;   // in radians

    // mouse tracking
    bool         m_performTracking             = false;
    int2         m_pos                      = { 0, 0 };
    int2         m_prevPos                  = { 0, 0 };
    int2         m_delta                    = { 0, 0 };

    // trackball computes camera orientation (eye, lookat) using
    // latitude/longitude with respect to this frame local frame for trackball
    float3       m_u                        = { 0.0f, 0.0f, 0.0f };
    float3       m_v                        = { 0.0f, 0.0f, 0.0f };
    float3       m_w                        = { 0.0f, 0.0f, 0.0f };
};

} // namespace otk
