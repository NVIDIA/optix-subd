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

// clang-format off
#include <cluster_builder/cluster.h>
#include <scene/sceneTypes.h>
#include <texture/textureCache.h>

#include <OptiXToolkit/ShaderUtil/Matrix.h>
#include <GBuffer.h>

#include <cstdint>
#include <cstring>

#include <cuda_runtime.h>
#include <optix_stubs.h>

// clang-format on


enum class ColorMode : uint8_t
{
    BASE_COLOR = 0,
    COLOR_BY_TRIANGLE,
    COLOR_BY_NORMAL,
    COLOR_BY_TEXCOORD,
    COLOR_BY_MATERIAL,
    COLOR_BY_CLUSTER_ID,
    COLOR_BY_CLUSTER_UV,      
    COLOR_BY_MICROTRI_AREA,
    COUNT
};


struct BoundValues
{
    bool enableWireframe = false;
    bool enableSurfaceWireframe = false;

    ColorMode colorMode = ColorMode::COLOR_BY_NORMAL;

};

// Things needed to compute motion vectors
struct HitResult
{
    uint32_t instanceId   = ~uint32_t( 0 );
    uint32_t surfaceIndex = ~uint32_t( 0 );
    float    u            = 0.0f;
    float    v            = 0.0f;
    float2   texcoord     = { 0 };
};

struct Params
{
    unsigned int              frame_index = 0;
    unsigned int              subframe_index = 0;
    HitResult*                hit_buffer = nullptr;
    uchar4*                   frame_buffer = nullptr;
    const ClusterShadingData* cluster_shading_data = nullptr;
    ClusterPattern            cluster_pattern {};
    const float3*             clusterVertexPositions = nullptr;
    const uint32_t*           packedClusterVertexNormals = nullptr;

    RwFloat4 aovAlbedo;
    RwFloat4 aovNormals;
    RwFloat4 aovColor;
    RwFloat aovDepth;
    RwFloatInterop aovDepthHires;

    float3 eye {};
    float3 U {};
    float3 V {};
    float3 W {};
    float2 jitter {};

    int  aoSamples = 9;

    float3 missColor = { 0.1f, 0.1f, 0.1f };

    OptixTraversableHandle handle = 0;

    BoundValues bound;

    // for COLOR_BY_MICROTRI_AREA colorMode.
    otk::Matrix4x4 tessViewProjectionMatrix;

    // for wireframe
    otk::Matrix4x4 viewProjectionMatrix;
    otk::Matrix4x4 viewMatrix;
    otk::Matrix4x4 projectionMatrix;

};

struct RayGenData
{
};

struct MissData
{
};


struct MaterialCuda;
struct HitGroupData
{
    MaterialCuda const* material;
};


enum RayType
{
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_OCCLUSION,
    RAY_TYPE_COUNT
};

