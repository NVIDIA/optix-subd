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

#include <OptiXToolkit/ShaderUtil/Aabb.h>
#include <OptiXToolkit/ShaderUtil/Affine.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>
#include <OptiXToolkit/ShaderUtil/Matrix.h>
#include <OptiXToolkit/ShaderUtil/Quaternion.h>

#include <array>
#include <string>

namespace otk {

// implementing a perspective camera
class Camera {

public:

    bool hasChanged() const { return m_changed; }

    float3 getDirection() const { return normalize(m_lookat - m_eye); }
    void setDirection(const float3& dir) { m_lookat = m_eye + length(m_lookat - m_eye) * dir; }

    void translate(float3 const& v);
    void rotate(float yaw, float pitch, float roll);
    void roll(float speed);

    void dolly(float factor);
    void pan(float2 speed);
    void zoom(const float factor);

    void frame(Aabb const& aabb);

    // UVW forms an orthogonal, but not orthonormal basis!
    std::array<float3, 3> const& getBasis();

    std::string getCliArgs() const;
    std::string getPositionRotation() const;

public:

    float3 getEye() const { return m_eye; }
    float3 getLookat() const { return m_lookat; }
    float3 getUp() const { return m_up; }

    float getFovY() const { return m_fovY; }
    float getAspectRatio() const { return m_aspectRatio; }
    
    float getZNear() const { return m_zNear; };
    float getZFar() const { return m_zFar; };

    quat getRotation() const;
    Affine getWorldToview() const;
    Affine getTranslatedWorldToview() const;

    Matrix4x4 getViewMatrix() const; 
    Matrix4x4 getProjectionMatrix() const;
    Matrix4x4 getViewProjectionMatrix() const;

  public:

    void setEye( float3 eye );
    void setLookat( float3 lookat );
    void setUp( float3 up );
    void setRotation( otk::quat rotation );

    void setFovY( float fovy );
    void setAspectRatio( float ar );
    void setNear( float near );
    void setFar( float far );

    void set(std::string const& camc_string);

private:

    void computeBasis(float3& u, float3& v, float3& w) const;

    std::array<float3, 3> m_basis = {};

    float3 m_eye = make_float3(1.f);
    float3 m_lookat = make_float3(0.f);
    float3 m_up = make_float3(0.f, 1.f, 0.f);

    float m_fovY        = 35.f;
    float m_aspectRatio = 1.f;
    float m_zNear       = 0.01f;
    float m_zFar        = 1000.f;

    bool  m_changed = true;
};

} // namespace otk
