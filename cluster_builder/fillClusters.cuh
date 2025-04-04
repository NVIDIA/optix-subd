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

#include <cluster_builder/tilings.h>
#include <subdivision/SubdivisionSurfaceCUDA.h>


template <typename SUBD_T, typename CLUSTER_GATHERER, ClusterPattern P>
__device__ void writeClusterPoints_warp( const Cluster& rCluster, const GridSampler& rSampler, const SUBD_T& subd, uint32_t iSurface, CLUSTER_GATHERER& gatherer )
{
    // warp wide loop
    for( uint32_t point_index = threadIdx.x; point_index < rCluster.verticesPerCluster(); point_index += 32 )
    {
        float2 uv = rSampler.uv<P>( rCluster.linear2idx2d( point_index ) + rCluster.offset );
        float du = rSampler.du( uv );
        float dv = rSampler.dv( uv );

        const LimitFrame limit = subd.evaluate( iSurface, uv, du, dv );

        gatherer.writeLimit( limit, rCluster, point_index );
    }
}

template <typename SUBD, typename CLUSTER_GATHERER, ClusterPattern PATTERN>
__global__ void fillClusterKernel( const std::span<const GridSampler> gridSamplers,
                                   const std::span<const Cluster>     clusters,
                                   const SUBD                         subd,
                                   CLUSTER_GATHERER                   gatherer )
{
    const uint32_t i_cluster = blockIdx.x * blockDim.y + threadIdx.y;
    if( i_cluster >= clusters.size() )
        return; 

    const auto         cluster   = clusters[i_cluster];
    const uint32_t     i_surface = cluster.iSurface;
    const GridSampler& sampler   = gridSamplers[i_surface];

    writeClusterPoints_warp<SUBD, CLUSTER_GATHERER, PATTERN>( cluster, sampler, subd, i_surface, gatherer );
    __syncwarp();
    gatherer.writeCluster( cluster, i_cluster, sampler, subd.materialId( i_surface ) );

    // Extra shading data: texcoords on corners of surfaces (quad patches)
    {
        const float2 surface_uvs[4] = { { 0, 0 }, { 1, 0 }, { 1, 1 }, { 0, 1 } };

        if( threadIdx.x < 4 )
        {
            TexCoordLimitFrame texcoord;
            subd.texcoord_subd.evaluate( texcoord, surface_uvs[threadIdx.x], i_surface, subd.texcoord_subd.patch_points[i_surface] );
            gatherer.writeTexCoord( texcoord, i_cluster, threadIdx.x );
        }
    }
}


template <ClusterPattern PATTERN, typename CLUSTER_GATHERER>
void fillClusters( const TessellatorConfig&           tessConfig,
                   const SubdivisionSurface&          subdivision_surface,
                   const std::span<const GridSampler> gridSamplers,
                   const std::span<const Cluster>     clusters,
                   CLUSTER_GATHERER&                  gatherer )
{

    constexpr uint32_t WarpsPerBlock = 4;
    const dim3 grid_shape = { std::max( div_up( static_cast<uint32_t>( clusters.size() ), WarpsPerBlock ), 1u ), 1, 1 };
    const dim3 block_shape = { 32, WarpsPerBlock };

    if ( subdivision_surface.m_hasDisplacement && tessConfig.displacement_scale > 0.f )
    {
        typedef DisplacedSubdCUDA<Vertex, Vertex> SubdCuda;
        auto subd = SubdCuda( subdivision_surface, tessConfig.displacement_scale, tessConfig.displacement_bias,
                tessConfig.displacement_filter_scale, tessConfig.displacement_filter_mip_bias, tessConfig.materials );
        fillClusterKernel<SubdCuda, CLUSTER_GATHERER, PATTERN><<<grid_shape, block_shape>>>( gridSamplers, clusters, subd, gatherer );
    }
    else
    {
        typedef SubdCUDA<Vertex, Vertex> SubdCuda;
        auto subd = SubdCuda( subdivision_surface );
        fillClusterKernel<SubdCuda, CLUSTER_GATHERER, PATTERN><<<grid_shape, block_shape>>>( gridSamplers, clusters, subd, gatherer );
    }
    CUDA_SYNC_CHECK();
}

template <typename CLUSTER_GATHERER>
void fillClusters( const TessellatorConfig&           tessConfig,
                   const SubdivisionSurface&          subdSurface,
                   const std::span<const GridSampler> gridSamplers,
                   const std::span<const Cluster>     clusters,
                   CLUSTER_GATHERER&                  gatherer )
{
    if( tessConfig.cluster_pattern == ClusterPattern::SLANTED )
        fillClusters<ClusterPattern::SLANTED, CLUSTER_GATHERER>( tessConfig, subdSurface, gridSamplers, clusters, gatherer );
    else
        fillClusters<ClusterPattern::REGULAR, CLUSTER_GATHERER>( tessConfig, subdSurface, gridSamplers, clusters, gatherer );
}
