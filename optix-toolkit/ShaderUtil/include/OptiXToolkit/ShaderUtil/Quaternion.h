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

#include <OptiXToolkit/ShaderUtil/Affine.h>
#include <OptiXToolkit/ShaderUtil/Matrix.h>

#include <algorithm>
#include <vector_functions.h>
#include <vector_types.h>

#if !defined(__CUDACC_RTC__)
#include <cmath>
#include <cstdlib>
#endif

namespace otk
{
    template <typename T> struct quaternion
    {
        T w = T(1), x = T(0), y = T(0), z = T(0);
    };

    template <typename T> quaternion<T> operator - (quaternion<T> const& a) { return quaternion<T>{ -a.w, -a.x, -a.y, -a.z }; }

    template <typename T> quaternion<T> operator - (quaternion<T> const& a, quaternion<T> const& b) { return quaternion<T>{ a.w - b.w, a.x - b.x, a.y - b.y, a.z - b.z }; }
    template <typename T> quaternion<T> operator + (quaternion<T> const& a, quaternion<T> const& b) { return quaternion<T>{ a.w + b.w, a.x + b.x, a.y + b.y, a.z + b.z }; }

    template <typename T> quaternion<T> operator * (T a, quaternion<T> const& b) { return quaternion<T>{a* b.w, a* b.x, a* b.y, a* b.z}; }
    template <typename T> quaternion<T> operator / (T a, quaternion<T> const& b) { return quaternion<T>{a / b.w, a / b.x, a / b.y, a / b.z}; }
    template <typename T> quaternion<T> operator * (quaternion<T> const& a, T b) { return quaternion<T>{a.w* b, a.x* b, a.y* b, a.z* b}; }
    template <typename T> quaternion<T> operator / (quaternion<T> const& a, T b) { return quaternion<T>{a.w / b, a.x / b, a.y / b, a.z / b}; }

    template <typename T> bool operator == (quaternion<T> const& a, quaternion<T> const& b) { return a.w == b.w && a.x == b.x && a.y == b.y && a.z == b.z; }
    template <typename T> bool operator != (quaternion<T> const& a, quaternion<T> const& b) { return a.w != b.w || a.x != b.x || a.y != b.y || a.z != b.z; }

    template <typename T> quaternion<T> operator += (quaternion<T> const& a, quaternion<T> const& b) { return a.w += b.w; a.x += b.x; a.y += b.y; a.z += b.z; };
    template <typename T> quaternion<T> operator -= (quaternion<T> const& a, quaternion<T> const& b) { return a.w -= b.w; a.x -= b.x; a.y -= b.y; a.z -= b.z; };

    template <typename T> quaternion<T> operator += (quaternion<T> const& a, T b) { a.w += b; a.x += b; a.y += b; a.z += b; };
    template <typename T> quaternion<T> operator -= (quaternion<T> const& a, T b) { a.w -= b; a.x -= b; a.y -= b; a.z -= b; };
    template <typename T> quaternion<T> operator *= (quaternion<T> const& a, T b) { a.w *= b; a.x *= b; a.y *= b; a.z *= b; };
    template <typename T> quaternion<T> operator /= (quaternion<T> const& a, T b) { a.w /= b; a.x /= b; a.y /= b; a.z /= b; };

    template<typename T> quaternion<T> operator * (quaternion<T> const& a, quaternion<T> const& b)
    {
        return quaternion<T>{
            a.w* b.w - a.x * b.x - a.y * b.y - a.z * b.z,
                a.w* b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                a.w* b.y + a.y * b.w + a.z * b.x - a.x * b.z,
                a.w* b.z + a.z * b.w + a.x * b.y - a.y * b.x };
    }

    template<typename T> OTK_HOSTDEVICE quaternion<T>& operator *= (quaternion<T>& a, quaternion<T> const& b)
    {
        a = a * b;
        return a;
    }


    template<typename T> OTK_HOSTDEVICE T dot(quaternion<T> const& a, quaternion<T> const& b)
    {
        return a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
    }

    template<typename T> OTK_HOSTDEVICE T lengthSquared(quaternion<T> const& a)
    {
        return dot(a, a);
    }

    template<typename T> OTK_HOSTDEVICE T length(quaternion<T> const& a)
    {
        return T(sqrt(lengthSquared(a)));
    }

    template<typename T> OTK_HOSTDEVICE quaternion<T> normalize(quaternion<T> const& a)
    {
        return a / length(a);
    }

    template<typename T> OTK_HOSTDEVICE quaternion<T> conjugate(quaternion<T> const& a)
    {
        return quaternion<T>(a.w, -a.x, -a.y, -a.z);
    }

    template<typename T> OTK_HOSTDEVICE quaternion<T> pow(quaternion<T> const& a, int b)
    {
        if (b <= 0)
            return {};
        if (b == 1)
            return a;
        quaternion<T> odd = {};
        quaternion<T> even = a;
        while (b > 1)
        {
            if (b % 2 == 1)
                odd *= even;
            even *= even;
            b /= 2;
        }
        return odd * even;
    }

    template<typename T> OTK_HOSTDEVICE quaternion<T> inverse(quaternion<T> const& a)
    {
        return conjugate(a) / lengthSquared(a);
    }

    template <typename T, typename V> OTK_HOSTDEVICE V applyQuat(quaternion<T> const& a, V const& b)
    {
        quaternion<T> q = { 0, b.x, b.y, b.z };
        quaternion<T> resultQ = a * q * conjugate(a);
        V result = { resultQ.x, resultQ.y, resultQ.z };
        return result;
    }

    template<typename T, typename V> OTK_HOSTDEVICE quaternion<T> rotation(V const& axis, T radians)
    {
        // Note: assumes axis is normalized
        T sinHalfTheta = sin(T(0.5) * radians);
        T cosHalfTheta = cos(T(0.5) * radians);

        return quaternion<T>{ cosHalfTheta, axis.x * sinHalfTheta, axis.y * sinHalfTheta, axis.z * sinHalfTheta };
    }

    template<typename T, typename V> OTK_HOSTDEVICE quaternion<T> rotationEuler(V const& euler)
    {
        T sinHalfX = (T)sin(T(0.5) * euler.x);
        T cosHalfX = (T)cos(T(0.5) * euler.x);
        T sinHalfY = (T)sin(T(0.5) * euler.y);
        T cosHalfY = (T)cos(T(0.5) * euler.y);
        T sinHalfZ = (T)sin(T(0.5) * euler.z);
        T cosHalfZ = (T)cos(T(0.5) * euler.z);

        quaternion<T> quatX = quaternion<T>(cosHalfX, sinHalfX, 0, 0);
        quaternion<T> quatY = quaternion<T>(cosHalfY, 0, sinHalfY, 0);
        quaternion<T> quatZ = quaternion<T>(cosHalfZ, 0, 0, sinHalfZ);

        // Note: multiplication order for quats is like column-vector convention
        return quatZ * quatY * quatX;
    }

    template<typename T> OTK_HOSTDEVICE quaternion<T> lerp(quaternion<T> const& a, quaternion<T> const& b, T u)
    {
        if (dot(a, b) < 0.f)
            return a - u * (b + a);
        else
            return a + u * (b - a);
    }
    
    template<typename T> OTK_HOSTDEVICE quaternion<T> nlerp(quaternion<T> const& a, quaternion<T> const& b, T u)
    {
        return normalize(lerp(a, b, u));
    }

    template<typename T> OTK_HOSTDEVICE quaternion<T> slerp(quaternion<T> const& a, quaternion<T> const& b, T u)
    {
        if (T d = dot(a, b); (T(-1) <= d) && (d <= T(1)))
        {
            T theta = T(acos(d));
            if (theta <= T(0))
                return a;
            return (a * T(sin((T(1) - u)) * theta) + b * T(sin(u * theta))) / T(sin(theta));
        }
        return a;
    }

    template<typename T> OTK_HOSTDEVICE Matrix3x3 toMatrix(quaternion<T> const& q)
    {
        return {
            T(1 - 2 * (q.y * q.y + q.z * q.z)), T(    2 * (q.x * q.y - q.z * q.w)), T(    2 * (q.x * q.z + q.y * q.w)),
            T(    2 * (q.x * q.y + q.z * q.w)), T(1 - 2 * (q.x * q.x + q.z * q.z)), T(    2 * (q.y * q.z - q.x * q.w)),
            T(    2 * (q.x * q.z - q.y * q.w)), T(    2 * (q.y * q.z + q.x * q.w)), T(1 - 2 * (q.x * q.x + q.y * q.y)),
        };
    }

    template<typename T, typename V> OTK_HOSTDEVICE Affine toAffine(quaternion<T> const& q)
    {
        return { toMatrix( q ), V( 0 ) };
    }

    template<typename T, typename V> void decomposeAffine(Affine const& transform,
        V* pTranslation, quaternion<T>* pRotation, V* pScaling)
    {
        if (pTranslation)
            *pTranslation = transform.translation;

        V col0 = transform.linear.getCol(0);
        V col1 = transform.linear.getCol(1);
        V col2 = transform.linear.getCol(2);

        V scaling;
        scaling.x = length(col0);
        scaling.y = length(col1);
        scaling.z = length(col2);
        if (scaling.x > 0.f) col0 /= scaling.x;
        if (scaling.y > 0.f) col1 /= scaling.y;
        if (scaling.z > 0.f) col2 /= scaling.z;

        V zAxis = cross(col0, col1);
        if (dot(zAxis, col2) < T(0))
        {
            scaling.x = -scaling.x;
            col0 = -col0;
        }

        if (pScaling)
            *pScaling = scaling;

        if (pRotation)
        {
            // https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
            quaternion<T> rotation;
            rotation.w = std::sqrt(std::max(T(0), T(1) + col0.x + col1.y + col2.z)) * T(0.5);
            rotation.x = std::sqrt(std::max(T(0), T(1) + col0.x - col1.y - col2.z)) * T(0.5);
            rotation.y = std::sqrt(std::max(T(0), T(1) - col0.x + col1.y - col2.z)) * T(0.5);
            rotation.z = std::sqrt(std::max(T(0), T(1) - col0.x - col1.y + col2.z)) * T(0.5);
            rotation.x = std::copysign(rotation.x, col2.y - col1.z);
            rotation.y = std::copysign(rotation.y, col0.z - col2.x);
            rotation.z = std::copysign(rotation.z, col1.x - col0.y);
            *pRotation = rotation;
        }
    }

    using quat = quaternion<float>;
    using dquat = quaternion<double>;

} // end namespace otk
