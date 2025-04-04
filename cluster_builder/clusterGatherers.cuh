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

#include <cluster_builder/tessellator.h>
#include <cluster_builder/tilings.h>
#include <scene/vertex.h>
#include <utils.cuh>

#include <span>


struct PatchGatherer
{
    TessellationCounters*  counters;
    std::span<GridSampler> gridSamplers;
    std::span<Cluster>     clusters;
    uint32_t               maxClusterEdgeSegments;

    template <typename SUBD_T>
    __device__ void writeSurfaceWarp( const SUBD_T& subd, uint32_t i_surface, uint32_t edge_segments, std::span<LimitFrame> samples )
    {
        auto&      gridSampler = gridSamplers[i_surface];
        const auto i_lane      = threadIdx.x;
        if( i_lane < 4 )
            ( &gridSampler.edge_segments.x )[i_lane] = edge_segments;
        tileSurfaceWithClusters( subd, i_surface );
    }

  protected:
    template <typename SUBD_T>
    __device__ void tileSurface_warp( const SUBD_T&        subd,
                                      const SurfaceTiling& surfaceTiling,
                                      const uint32_t       i_surface,
                                      const uint32_t       surfaceClusterOffset,
                                      const uint32_t       surfaceVertexOffset )
    {
        uint32_t tilingClusterOffset = surfaceClusterOffset;
        uint32_t tilingVertexOffset  = surfaceVertexOffset;
        for( auto i_tiling = 0; i_tiling < surfaceTiling.N_SUB_TILINGS; ++i_tiling )
        {
            const auto clusterTiling = surfaceTiling.subTilings[i_tiling];
            if( surfaceClusterOffset < clusters.size() )
            {
                auto tilingClusterIds = clusters.subspan( tilingClusterOffset );

                const auto i_lane = threadIdx.x;
                // make clusters with tilingSize
                for( uint32_t i_cluster = i_lane;
                     i_cluster < std::min( clusterTiling.clusterCount(), static_cast<uint32_t>( tilingClusterIds.size() ) );
                     i_cluster += 32 )
                {
                    tilingClusterIds[i_cluster] =
                        Cluster( i_surface, tilingVertexOffset + clusterTiling.clusterVertexCount() * i_cluster,
                                 surfaceTiling.clusterOffset( i_tiling, i_cluster ), clusterTiling.clusterSize );
                }
            }
            tilingClusterOffset += clusterTiling.clusterCount();
            tilingVertexOffset += clusterTiling.vertexCount();
        }
    }

    template <typename SUBD_T>
    __device__ void tileSurfaceWithClusters( const SUBD_T& subd, const uint32_t i_surface )
    {
        auto& gridSampler    = gridSamplers[i_surface];
        auto  surface_size   = gridSampler.gridSize();
        auto  surfaceTiling  = SurfaceTiling( surface_size, maxClusterEdgeSegments );
        // compute cluster and vertex offsets into linear storage using global counters
        const auto i_lane = threadIdx.x;
        uint32_t   surfaceClusterOffset, surfaceVertexOffset;
        if( i_lane == 0 )
        {
            surfaceClusterOffset = atomicAdd( &counters->clusters, surfaceTiling.clusterCount() );
            surfaceVertexOffset  = atomicAdd( &counters->vertices, surfaceTiling.vertexCount() );

            const uint32_t n_cluster_tris = 2 * surface_size.x * surface_size.y;
            atomicAdd( &counters->triangles, n_cluster_tris );
        }

        surfaceClusterOffset = __shfl_sync( 0xFFFFFFFF, surfaceClusterOffset, 0 );
        surfaceVertexOffset  = __shfl_sync( 0xFFFFFFFF, surfaceVertexOffset, 0 );

        tileSurface_warp( subd, surfaceTiling, i_surface, surfaceClusterOffset, surfaceVertexOffset );
    }
};

__device__ inline float quantize( float a, uint32_t nbits )
{
    nbits = 32 - static_cast<int>( nbits );
    int mask = (1<<(32-nbits))-1;
    return __int_as_float( __float_as_int(a) & ~mask );
}

__device__ inline float3 quantize( float3 v, uint32_t nbits )
{
    return float3{ quantize(v.x, nbits), quantize(v.y, nbits), quantize(v.z, nbits) };
}


struct ClusterGatherer
{
    std::span<float3>             vertexPositions;
    std::span<ClusterShadingData> clusterShadingData;
    uint32_t                      quantizationBits = 0;

    const std::span<CUdeviceptr>                        templateAddresses;
    std::span<OptixClusterAccelBuildInputTemplatesArgs> indirectArgData;
    uint32_t                                            clusterOffset;
    uint32_t                                            maxClusterEdgeSegments;

    __device__ void writeLimit( const LimitFrame& vertexLimit, const Cluster& cluster, uint32_t vertexIndex )
    {
        vertexPositions[cluster.nVertexOffset + vertexIndex] = quantize( vertexLimit.point, quantizationBits );
    }

    __device__ void writeTexCoord( const TexCoordLimitFrame& texLimit, uint32_t clusterIndex, uint32_t cornerIndex )
    {
        clusterShadingData[clusterIndex].m_surface_texcoords[threadIdx.x] = texLimit.uv;
    }

    __device__ void writeCluster( const Cluster& cluster, uint32_t clusterIndex, const GridSampler& gridSampler, uint16_t materialId )
    {
        if( threadIdx.x == 0 )
        {
            const CUdeviceptr vertexBufferAddress = reinterpret_cast<CUdeviceptr>( &vertexPositions[cluster.nVertexOffset] );

            const auto clusterResolution = cluster.size;
            const uint32_t templateIndex = ( clusterResolution.x - 1 ) * maxClusterEdgeSegments + ( clusterResolution.y - 1 );
            indirectArgData[clusterIndex] =
                OptixClusterAccelBuildInputTemplatesArgs{ .clusterIdOffset     = clusterIndex + clusterOffset,
                                                          .sbtIndexOffset      = materialId,
                                                          .clusterTemplate     = templateAddresses[templateIndex],
                                                          .vertexBuffer        = vertexBufferAddress,
                                                          .vertexStrideInBytes = sizeof( float3 ) };

            assert( clusterIndex < clusterShadingData.size() );
            clusterShadingData[clusterIndex] = {
                .m_surface_edge_segments = gridSampler.edge_segments,
                .m_surface_id = cluster.iSurface,
                .m_cluster_vertex_offset = cluster.nVertexOffset,
                .m_cluster_offset = cluster.offset,
                .m_cluster_size = cluster.size,
            };
        }
    }
};


struct ClusterGatherer_N : ClusterGatherer
{
    std::span<uint32_t> packedVertexNormals;


    __device__ void writeLimit( const LimitFrame& vertexLimit, const Cluster& cluster, uint32_t vertexIndex )
    {
        ClusterGatherer::writeLimit( vertexLimit, cluster, vertexIndex );

        float3 normal = normalize( cross( vertexLimit.deriv1, vertexLimit.deriv2 ) );
        packedVertexNormals[vertexIndex + cluster.nVertexOffset] = packNormalizedVector( normal );
    }
};
