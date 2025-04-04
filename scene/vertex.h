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

#include <OptiXToolkit/ShaderUtil/vec_math.h>
#include <cstdint>
#include <cstdio>

struct Vertex {
    float3 point = { 0 };

    __host__ __device__ void Clear(void* = 0) {
        point = { 0 };
    }

    __host__ __device__ void AddWithWeight(Vertex const& src, float weight) {
        point += weight * src.point;
    }

    __host__ __device__ void Print() const {
        printf("%f %f %f", point.x, point.y, point.z);
    }

    __host__ __device__ float& operator[](uint32_t i)
    {
        return reinterpret_cast<float*>(&point.x)[i];
    }

    __host__ __device__ float operator[](uint32_t i) const
    {
        return reinterpret_cast<const float*>(&point.x)[i];
    }

    __host__ __device__ Vertex operator*(float s) const { return Vertex{s * this->point}; };

    __host__ __device__ Vertex& operator+=(const Vertex& v) { this->point = v.point + this->point; return *this; };
    __host__ __device__ Vertex operator+(const Vertex& v) const { return Vertex{v.point + this->point}; };
};

__host__ __device__ inline Vertex operator*(float s, const Vertex& v) { return Vertex{s * v.point}; };


struct LimitFrame
{
    float3 point = { 0 };
    float3 deriv1 = { 0 };
    float3 deriv2 = { 0 };

    __host__ __device__ void Clear(void* = 0) {
        point = { 0 };
        deriv1 = { 0 };
        deriv2 = { 0 };
    }

    __host__ __device__ void AddWithWeight(Vertex const& src,
        float weight, float d1Weight, float d2Weight) {

        point += weight * src.point;
        deriv1 += d1Weight * src.point;
        deriv2 += d2Weight * src.point;
    }

    // Get linearly index float.
    // The indexed array consists of:
    //   [point.x, point.y, point.z, deriv1.x, ...]
    __host__ __device__ float& operator[](uint32_t i)
    {
        return reinterpret_cast<float*>(&point.x)[i];
    }
};

struct TexCoord {

    __host__ __device__ 
    void Clear() 
    { 
        uv = { 0 };
    }

    __host__ __device__ 
    void Set(TexCoord const& other, float w) {
        uv.x = w * other.uv.x;
        uv.y = w * other.uv.y;
    }

    __host__ __device__ 
    void AddWithWeight(TexCoord const& src, float w) {
        uv.x += w * src.uv.x;
        uv.y += w * src.uv.y;
    }

    __host__ __device__ TexCoord operator*(float s) const { return TexCoord{s * this->uv}; };
    __host__ __device__ TexCoord& operator+=(const TexCoord& rhs) { this->uv = rhs.uv + this->uv; return *this; };
    __host__ __device__ TexCoord operator+(const TexCoord& rhs) const { return TexCoord{rhs.uv + this->uv}; };


    float2 uv = { 0 };
};

// Texture coordinate with partial derivs w.r.t the parametric U and V directions of the surface
struct TexCoordLimitFrame {

    __host__ __device__ 
    void Clear() 
    { 
        uv = { 0 };
        deriv1 = { 0 };
        deriv2 = { 0 };
    }

    __host__ __device__ 
    void AddWithWeight(TexCoord const& src, float weight, float du_weight, float dv_weight ) {
        uv += weight * src.uv;
        deriv1 += du_weight * src.uv;
        deriv2 += dv_weight * src.uv;
    }

    float2 uv = {0}; 
    float2 deriv1 = {0};  // (dST/du)
    float2 deriv2 = {0};  // (dST/du)
};
