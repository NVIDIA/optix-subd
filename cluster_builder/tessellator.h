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


#include <cluster_builder/cluster.h>
#include <cluster_builder/tessellatorConfig.h>
#include <subdivision/SubdivisionSurface.h>

#include <span>


struct TessellationCounters 
{
    uint32_t clusters = 0;
    uint32_t vertices = 0;
    uint32_t triangles = 0;
};

namespace tess 
{

void computeClusterTiling( const TessellatorConfig&  tessConfig,
                           const SubdivisionSurface& subdivisionSurface,
                           const otk::Matrix3x4&     localToWorld,
                           std::span<GridSampler>    gridSamplers,
                           std::span<Cluster>        clusters,
                           TessellationCounters*     d_counters );
                 

struct FillClusterOutput
{
    std::span<float3>             vertexPositions;
    std::span<uint32_t>           packedVertexNormals;
    std::span<ClusterShadingData> clusterShadingData;

    const std::span<CUdeviceptr>                        templateAddresses;
    std::span<OptixClusterAccelBuildInputTemplatesArgs> indirectArgData;
};


void fillClusters( const TessellatorConfig&           tessConfig,
                   const SubdivisionSurface&          subdSurface,
                   const std::span<const GridSampler> gridSamplers,
                   const std::span<const Cluster>     clusters,
                   uint32_t                           clusterOffset,
                   FillClusterOutput&                 out );

};  // namespace
