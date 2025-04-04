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

#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include <tuple>


template <typename F>
struct ReadWriteResourceCuArray
{
    uint2 m_size = {0, 0};
    cudaSurfaceObject_t m_surfaceObj = 0;
    cudaTextureObject_t m_pointSampleTexObj = 0;
    cudaArray_t m_array = 0;

    inline bool isValid() const { return m_array; }
};

struct cudaGraphicsResource;
template <typename F>
struct ReadWriteResourceInterop
{
    uint2 m_size = {0, 0};
    cudaSurfaceObject_t   m_surfaceObj         = 0;
    cudaTextureObject_t   m_pointSampleTexObj  = 0;
    cudaArray_t m_array = 0;
    cudaGraphicsResource* m_cudaGraphicsResource = nullptr;
    uint32_t              m_glTexId = 0;
    bool                  m_mapped = false;

    inline bool isValid() const { return m_glTexId; }
};



typedef ReadWriteResourceCuArray<float>  RwFloat;
typedef ReadWriteResourceCuArray<float2> RwFloat2;
typedef ReadWriteResourceCuArray<float4> RwFloat4;

typedef ReadWriteResourceInterop<float>  RwFloatInterop;


struct GBuffer
{
    enum class Channel : uint8_t
    {
        ALBEDO = 0,
        NORMALS,
        MOTIONVECS,
        COLOR,
        DEPTH,
        DEPTH_HIRES,
        DENOISED,
        COUNT
    };

    RwFloat4 m_albedo;
    RwFloat4 m_normals;
    RwFloat2 m_motionvecs;
    RwFloat4 m_color;
    RwFloat  m_depth;
    RwFloatInterop m_depthHires;
    RwFloat4 m_denoised;

    // convenience for compile-time iteration over channels
    auto channels() {
        return std::tie( m_albedo, m_normals, m_motionvecs, m_color, m_depth, m_depthHires, m_denoised );
    }

    const uint2 m_rendersize = { 0, 0 };
    const uint2 m_targetsize = { 0, 0 };

    GBuffer() = delete;

    // Allocates resources
    GBuffer( uint2 rendersize, uint2 targetsize );

    // Destroys resources
    ~GBuffer();
    
    void map();
    void unmap();

    // Launches a blit kernel to copy a gbuffer channel to a uchar4* output buffer.
    void blit( Channel channel, uchar4* output, uint2 outputSize, CUstream stream = 0 );

    // Launches a kernel to return a single value in device mem - use judiciously
    void pickdepth( uint2 pixel, float * d_out );
};



