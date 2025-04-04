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

#include <stdio.h>

__device__ inline ushort2 operator*(const ushort2 a, const ushort2 b)
{
    return {static_cast<uint16_t>(a.x * b.x), static_cast<uint16_t>(a.y * b.y)};
}

__device__ inline ushort2 operator+(const ushort2 a, const ushort2 b)
{
    return {static_cast<uint16_t>(a.x + b.x), static_cast<uint16_t>(a.y + b.y)};
}

__device__ inline bool operator==(const ushort2 a, const ushort2 b)
{
    return (a.x == b.x) && (a.y == b.y);
}

__device__ inline bool operator==(const uchar2 a, const uchar2 b)
{
    return (a.x == b.x) && (a.y == b.y);
}

struct ClusterTiling
{
    ushort2 tilingSize;   // number of tiles in x and y direction
    uchar2  clusterSize;  // number of quads in x and y direction inside tile

    __device__ inline uint32_t clusterCount() const { return tilingSize.x * tilingSize.y; }
    __device__ inline uint32_t clusterVertexCount() const { return ( clusterSize.x + 1 ) * ( clusterSize.y + 1 ); }
    __device__ inline uint32_t vertexCount() const { return clusterVertexCount() * clusterCount(); }

    __device__ inline ushort2 clusterIndex2D( uint32_t rowMajorIndex ) const
    {
        return { static_cast<uint16_t>(rowMajorIndex % tilingSize.x), static_cast<uint16_t>(rowMajorIndex / tilingSize.x) };
    }

    __device__ inline ushort2 quadOffset2D( uint32_t rowMajorIndex ) const
    {
        return clusterIndex2D( rowMajorIndex ) * ushort2{ clusterSize.x, clusterSize.y };
    }

    __device__ inline uint2 vertexIndex2D( uint32_t rowMajorIndex ) const
    {
        const auto vertices_u = clusterSize.x + 1;
        return { rowMajorIndex % vertices_u, rowMajorIndex / vertices_u };
    }
};

struct SurfaceTiling
{
        enum
    {
        REGULAR,
        RIGHT,
        TOP,
        CORNER,
        N_SUB_TILINGS
    };
    ClusterTiling subTilings[N_SUB_TILINGS];
    ushort2       quadOffsets[N_SUB_TILINGS];  // quad offset of the tiling in x and y direction

    __device__ inline SurfaceTiling( const ushort2 surfaceSize, uint32_t maxEdgeSegments )
    {
        constexpr uint8_t target_edge_segments{ 8u };

        ushort2 regular_grid_size;
        uchar2  mod_cluster;
        {
            auto div_clusters = ushort2{ static_cast<uint16_t>( surfaceSize.x / target_edge_segments ),
                                         static_cast<uint16_t>( surfaceSize.y / target_edge_segments ) };
            mod_cluster       = uchar2{ static_cast<uint8_t>( surfaceSize.x % target_edge_segments ),
                                        static_cast<uint8_t>( surfaceSize.y % target_edge_segments ) };

            if( div_clusters.x > 0 && mod_cluster.x + target_edge_segments <= maxEdgeSegments )
            {
                div_clusters.x -= 1;
                mod_cluster.x += target_edge_segments;
            }
            if( div_clusters.y > 0 && mod_cluster.y + target_edge_segments <= maxEdgeSegments )
            {
                div_clusters.y -= 1;
                mod_cluster.y += target_edge_segments;
            }
            regular_grid_size = div_clusters;
        }
        assert( regular_grid_size.x * target_edge_segments + mod_cluster.x == surfaceSize.x );

        subTilings[REGULAR]  = ClusterTiling{ regular_grid_size, { target_edge_segments, target_edge_segments } };
        quadOffsets[REGULAR] = { 0u, 0u };

        subTilings[RIGHT]  = ClusterTiling{ { 1u, regular_grid_size.y }, { mod_cluster.x, target_edge_segments } };
        quadOffsets[RIGHT] = { static_cast<uint16_t>(regular_grid_size.x * target_edge_segments), 0u };

        subTilings[TOP]  = ClusterTiling{ { regular_grid_size.x, 1u }, { target_edge_segments, mod_cluster.y } };
        quadOffsets[TOP] = { 0u, static_cast<uint16_t>( regular_grid_size.y * target_edge_segments ) };

        subTilings[CORNER]  = ClusterTiling{ { 1u, 1u }, mod_cluster };
        quadOffsets[CORNER] = { static_cast<uint16_t>( regular_grid_size.x * target_edge_segments ),
                                static_cast<uint16_t>( regular_grid_size.y * target_edge_segments ) };
    }

    __device__ uint32_t inline clusterCount() const
    {
        uint32_t sum = 0;
        for( auto i_tiling = 0; i_tiling < N_SUB_TILINGS; ++i_tiling )
            sum += subTilings[i_tiling].clusterCount();
        return sum;
    }

    __device__ uint32_t inline vertexCount() const
    {
        uint32_t sum = 0;
        for( auto i_tiling = 0; i_tiling < N_SUB_TILINGS; ++i_tiling )
            sum += subTilings[i_tiling].vertexCount();
        return sum;
    }

    __device__ ushort2 inline clusterOffset( const uint8_t iTiling, const uint32_t iCluster ) const
    {
        return quadOffsets[iTiling] + subTilings[iTiling].quadOffset2D( iCluster );
    }
};

