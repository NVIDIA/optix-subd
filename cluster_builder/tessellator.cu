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

#include "tessellator.h"

#include "clusterGatherers.cuh"
#include "computeClusterTiling.cuh"
#include "fillClusters.cuh"

namespace tess {
  

void computeClusterTiling( const TessellatorConfig&  tessConfig,
                           const SubdivisionSurface& subdivisionSurface,
                           const otk::Matrix3x4&     localToWorld,
                           std::span<GridSampler>    gridSamplers,
                           std::span<Cluster>        clusters,
                           TessellationCounters*     d_counters )
{
    
    {
        OTK_ASSERT( tessConfig.viewport_size.y >= 0.f );
        PatchGatherer out{ d_counters, gridSamplers, clusters, tessConfig.maxClusterEdgeSegments };
        
        // Versions with/without displacement maps

        if( subdivisionSurface.m_hasDisplacement && tessConfig.displacement_scale > 0.f )
        {
            auto subd = DisplacedSubdCUDA<Vertex, Vertex>( subdivisionSurface, tessConfig.displacement_scale,
                    tessConfig.displacement_bias, tessConfig.displacement_filter_scale,
                    tessConfig.displacement_filter_mip_bias, tessConfig.materials );
            computeClusterTiling( tessConfig, subd, localToWorld, out );
        }
        else
        {
            auto subd = SubdCUDA<Vertex, Vertex>( subdivisionSurface );
            computeClusterTiling( tessConfig, subd, localToWorld, out );
        }
    }
}


void fillClusters( const TessellatorConfig&           tessConfig,
                   const SubdivisionSurface&          subd,
                   const std::span<const GridSampler> gridSamplers,
                   const std::span<const Cluster>     clusters,
                   uint32_t                           clusterOffset,
                   FillClusterOutput&                 out )
{

    // Versions with/without vertex normals

    if( out.packedVertexNormals.size() == 0 )
    {
        ClusterGatherer gatherer = { out.vertexPositions,
                                     out.clusterShadingData,
                                     tessConfig.quantNBits,
                                     out.templateAddresses,
                                     out.indirectArgData,
                                     clusterOffset,
                                     tessConfig.maxClusterEdgeSegments };

        fillClusters( tessConfig, subd, gridSamplers, clusters, gatherer );
    }
    else
    {
        ClusterGatherer_N gatherer = { out.vertexPositions,
                                       out.clusterShadingData,
                                       tessConfig.quantNBits,
                                       out.templateAddresses,
                                       out.indirectArgData,
                                       clusterOffset,
                                       tessConfig.maxClusterEdgeSegments,
                                       out.packedVertexNormals };

        fillClusters( tessConfig, subd, gridSamplers, clusters, gatherer );
    }
}
};  // namespace tess
