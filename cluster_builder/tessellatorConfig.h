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

#include "cluster.h"

struct MaterialCuda;
class TextureCuda;

namespace otk{
    class Camera;
}

struct TessellatorConfig
{
    static constexpr float DEFAULT_FINE_TESSELLATION_RATE   = 1.0f;
    static constexpr float DEFAULT_COARSE_TESSELLATION_RATE = 1.0f / 15.0f;
    static constexpr float DEFAULT_DISPLACEMENT_FILTER_SCALE = 0.0f;
    static constexpr float DEFAULT_DISPLACEMENT_FILTER_MIP_BIAS = 0.0f;

    bool  enableVertexNormals      = false;
    float fineTessellationRate     = DEFAULT_FINE_TESSELLATION_RATE;
    float coarseTessellationRate   = DEFAULT_COARSE_TESSELLATION_RATE;
    bool  enableFrustumVisibility  = true;
    bool  enableBackfaceVisibility = true;

    uint2 viewport_size        = { 0u, 0u };

    uint32_t         maxClusterEdgeSegments = 11;
    unsigned char    quantNBits      = 0;
    ClusterPattern   cluster_pattern = ClusterPattern::SLANTED;

    float               displacement_scale      = 1.0f;
    float               displacement_bias       = 0.0f;
    float               displacement_filter_scale = DEFAULT_DISPLACEMENT_FILTER_SCALE;
    float               displacement_filter_mip_bias = DEFAULT_DISPLACEMENT_FILTER_MIP_BIAS;

    const otk::Camera*   camera     = nullptr;
    const MaterialCuda*  materials  = nullptr;
};
