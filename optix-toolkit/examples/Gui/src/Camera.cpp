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
#include <OptiXToolkit/ShaderUtil/Matrix.h>

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <sstream>

static inline float radians(float degrees) { return degrees * M_PIf / 180.0f; }

namespace otk {

void Camera::translate(float3 const& v) 
{
    m_eye += v;
    m_lookat += v;
    m_changed = true;
}

void Camera::rotate(float yaw, float pitch, float roll)
{
    float3 w = m_lookat - m_eye;
    float wlen = length(w);
    float3 u = normalize(cross(w, m_up));

    Matrix4x4 rotation = Matrix4x4::identity();
    if (yaw != 0.f)
        rotation = Matrix4x4::rotate(-yaw, {0.f, 1.f, 0.f}) * rotation;
    if (pitch != 0.f)
        rotation = Matrix4x4::rotate(pitch, u) * rotation;

    if (roll != 0.f)
    {
        // simplified roll: fine for first-person movement,
        // but incorrect for a flight simulator
        rotation = Matrix4x4::rotate(roll, w) * rotation;
    }

    float4 dir = normalize(float4{ w.x, w.y, w.z, 0.f } * rotation);
    float4 up = normalize(float4{ m_up.x, m_up.y, m_up.z, 0.f } *rotation);
    
    m_lookat = m_eye + float3{ dir.x, dir.y, dir.z } * wlen;
    m_up = { up.x, up.y, up.z };

    m_changed = true;
}

void Camera::roll(float speed)
{
    auto const& basis = getBasis();
    float3 u = normalize(basis[0]);
    float3 v = normalize(basis[1]);
    m_up = u * cos(radians(90.0f + speed)) + v * sin(radians(90.0f + speed));
    m_changed = true;
}

void Camera::pan(float2 speed)
{
    auto const& basis = getBasis();
    float3 u = basis[0] * (-2.f * speed.x);
    float3 v = basis[1] * ( 2.f * speed.y);
    translate(u + v);
}

void Camera::dolly(float factor)
{
    // move closer by factor
    float3 oldEyeOffset = m_eye - m_lookat;
    m_eye = m_lookat + (oldEyeOffset * factor);
    m_changed = true;
}

void Camera::zoom(const float factor)
{
    // increase/decrease field-of-view angle by factor
    m_fovY = fminf(150.f, m_fovY * factor);
    m_changed = true;
}

void Camera::frame(Aabb const& aabb)
{
    setFovY(35.0f);
    setLookat(aabb.center());
    setEye(aabb.center() + 1.2f * make_float3(aabb.maxExtent()));
    setUp({ 0.f, 1.f, 0.f });
    m_changed = true;
}

std::array<float3, 3> const& Camera::getBasis()
{
    if (hasChanged())
    {
        computeBasis(m_basis[0], m_basis[1], m_basis[2]);
        m_changed = false;
    }
    return m_basis;
}

void Camera::computeBasis(float3& U, float3& V, float3& W) const
{
    float wlen = 0.f;

    W = m_lookat - m_eye; // Do not normalize W -- it implies focal length
    wlen = length(W);
    U = normalize(cross(W, m_up));
    V = normalize(cross(U, W));

    float vlen = wlen * tanf(0.5f * m_fovY * M_PIf / 180.0f);
    V *= vlen;
    float ulen = vlen * m_aspectRatio;
    U *= ulen;
}

void Camera::setEye(float3 eye) 
{ 
    m_eye = eye;
    m_changed = true; 
}
void Camera::setLookat(float3 lookat) 
{
    m_lookat = lookat;
    m_changed = true; 
}
void Camera::setUp(float3 up) 
{
    m_up = up;
    m_changed = true; 
}
void Camera::setFovY(float fovy) 
{
    m_fovY = fovy;
    m_changed = true;
}
void Camera::setAspectRatio(float ar) 
{ 
    m_aspectRatio = ar; 
    m_changed = true; 
}
void Camera::setNear( float near )
{
    m_zNear   = near;
    m_changed = true;
}
void Camera::setFar( float far )
{
    m_zFar    = far;
    m_changed = true;
}

static inline std::istream& operator>>(std::istream& is, float3& v)
{
    char st;
    is >> st >> v.x >> st >> v.y >> st >> v.z >> st;
    return is;
}

void Camera::set(std::string const& camc_string)
{
    std::istringstream istr(camc_string);
    istr >> m_eye;
    istr >> m_lookat;
    istr >> m_up;
    istr >> m_fovY;
    m_changed = true;
}

std::string Camera::getCliArgs() const
{
    char buf[255];
    snprintf(buf, std::size(buf), "[%g,%g,%g][%g,%g,%g][%g,%g,%g]%g",
        m_eye.x, m_eye.y, m_eye.z,
        m_lookat.x, m_lookat.y, m_lookat.z,
        m_up.x, m_up.y, m_up.z, 
        m_fovY);
    return buf;
}

std::string Camera::getPositionRotation() const
{
    float3 t = getEye();
    otk::quat r = getRotation();
    std::stringstream text;
    text << "\"position\": [" << t.x << ", " << t.y << ", " << t.z << "], ";
    text << "\"rotation\": [" << r.w << ", " << r.x << ", " << r.y << ", " << r.z << "],";
    return text.str();
}

quat Camera::getRotation() const
{
    Affine twtv = getTranslatedWorldToview();
    quat rotation;
    otk::decomposeAffine<float, float3>( twtv, (float3*)nullptr, &rotation, nullptr );
    return rotation;
}

void Camera::setRotation( otk::quat rotation )
{
    m_lookat = m_eye + otk::applyQuat( rotation, float3( 0.f, 0.f, -1.f ) );
    m_up = normalize( otk::applyQuat( rotation, float3( 0.f, 1.f, 0.f ) ) );
    m_changed = true;
}

Affine Camera::getTranslatedWorldToview() const
{
    float3 W = normalize( m_lookat - m_eye );
    float3 U = normalize( cross( W, m_up ) );
    float3 V = normalize( cross( U, W ) );

    // Set inverse=transpose of camera matrix directly
    Affine m;
    m.linear.setRow( 0, U );
    m.linear.setRow( 1, V );
    m.linear.setRow( 2, -W );
    return m;
}

Affine Camera::getWorldToview() const
{
    Affine twtv = getTranslatedWorldToview();
    return Affine::translate( -m_eye ) * twtv;
}

Matrix4x4 Camera::getViewMatrix() const
{
    float3 W = normalize( m_lookat - m_eye );
    float3 U = normalize( cross( W, m_up ) );
    float3 V = normalize( cross( U, W ) );

    // Set inverse=transpose of camera matrix directly
    float m[16];
    m[0] = U.x;
    m[1] = U.y;
    m[2] = U.z;
    m[3] = -dot( U, m_eye );

    m[4] = V.x;
    m[5] = V.y;
    m[6] = V.z;
    m[7] = -dot( V, m_eye );

    m[8]  = -W.x;
    m[9]  = -W.y;
    m[10] = -W.z;
    m[11] = dot( W, m_eye );

    m[12] = 0.f;
    m[13] = 0.f;
    m[14] = 0.f;
    m[15] = 1.0f;

    return Matrix<4, 4>( m );
}

Matrix4x4 Camera::getProjectionMatrix() const
{
    const float fov_rad = m_fovY * M_PIf / 180.f;
    const float tan_fov = tan( fov_rad / 2.f );

    // Column vectors in row-major memory layout
    // X-basis is in [0, 4, 8, 12]
    float m[16];
    m[0] = 1.f / ( tan_fov * m_aspectRatio );
    m[1] = 0.f;
    m[2] = 0.f;
    m[3] = 0.f;

    m[4] = 0.f;
    m[5] = 1.f / tan_fov;
    m[6] = 0.f;
    m[7] = 0.f;

    m[8]  = 0.f;
    m[9]  = 0.f;
    m[10] = ( m_zFar + m_zNear ) / ( m_zNear - m_zFar );
    m[11] = 2.f * m_zFar * m_zNear / ( m_zNear - m_zFar );

    m[12] = 0.f;
    m[13] = 0.f;
    m[14] = -1.f;
    m[15] = 0.f;

    return Matrix<4, 4>( m );
}

Matrix4x4 Camera::getViewProjectionMatrix() const
{
    return getProjectionMatrix() * getViewMatrix();
}

} // namespace otk
