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
#include <algorithm>
#include <assert.h>
#include <cstdint>
#include <stdio.h>

#ifndef _WIN32
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#endif


enum class ClusterPattern
{
    REGULAR,
    SLANTED
};

// Class for computing uv sample locations on a rectangular grid.
//
// Each edge may have a different number of segments (distinct UV locations),
// and the UVs are interpolated across the interior in either a regular 
// or slanted pattern.
//
struct GridSampler
{
    ushort4 edge_segments;

    inline __host__ __device__ uint16_t gridSizeX() const { return std::max( edge_segments.x, edge_segments.z ); }
    inline __host__ __device__ uint16_t gridSizeY() const { return std::max( edge_segments.y, edge_segments.w ); }
    inline __host__ __device__ ushort2 gridSize() const { return { gridSizeX(), gridSizeY() }; }

    inline __host__ __device__ float u_v0( uint16_t u_index ) const
    {
        return  round( edge_segments.x / static_cast<float>(gridSizeX()) * u_index ) / edge_segments.x;
    }

    inline __host__ __device__ float u_v1( uint16_t u_index ) const
    {
        return round( edge_segments.z / static_cast<float>(gridSizeX()) * u_index ) / edge_segments.z;
    }

    inline __host__ __device__ float u1_v( uint16_t v_index ) const
    {
        return round( edge_segments.y / static_cast<float>(gridSizeY()) * v_index ) / edge_segments.y;
    }

    inline __host__ __device__ float u0_v( uint16_t v_index ) const
    {
        return  round( edge_segments.w / static_cast<float>(gridSizeY()) * v_index ) / edge_segments.w;
    }

    inline __host__ __device__ float2 regular_interior_uv( uint16_t i, uint16_t j ) const
    {
        assert( 0 < i && i < gridSizeX() );
        assert( 0 < j && j < gridSizeY() );
        float cluster_u = i / static_cast<float>( gridSizeX() );
        float cluster_v = j / static_cast<float>( gridSizeY() );
        return make_float2( cluster_u, cluster_v );
    }

    inline __host__ __device__ float2 slanted_interior_uv( uint16_t i, uint16_t j ) const
    {
        assert( 0 < i && i < gridSizeX() );
        assert( 0 < j && j < gridSizeY() );
        float Du = u_v1( i ) - u_v0( i );
        float Dv = u1_v( j ) - u0_v( j );
        float cluster_u = ( u_v0( i ) + u0_v( j ) * Du ) / ( 1.0f - Du * Dv );
        float cluster_v = ( u0_v( j ) + u_v0( i ) * Dv ) / ( 1.0f - Du * Dv );
        return make_float2( cluster_u, cluster_v );
    }

    inline __host__ __device__ float2 boundary_uv( uint16_t i, uint16_t j ) const
    {
        if( j == 0 )
            return make_float2( u_v0( i ), 0.0f );
        if( j == gridSizeY() )
            return make_float2( u_v1( i ), 1.0f );

        if( i == 0 )
            return make_float2( 0.0f, u0_v( j ) );
        if( i == gridSizeX() )
            return make_float2( 1.0f, u1_v( j ) );
        assert( false );

        return make_float2(0.0f, 0.0f);
    }

    template<ClusterPattern P>
    inline __host__ __device__ float2 uv( ushort2 uv_index ) const
    {
        auto i = uv_index.x, j = uv_index.y;
        // interior uv locations
        if( 0 < i && i < gridSizeX() && 0 < j && j < gridSizeY() )
        {
            if( ClusterPattern::SLANTED == P )
                return slanted_interior_uv( i, j );
            else
                return regular_interior_uv( i, j );
        }
        else
            return boundary_uv( i, j );

        return make_float2( 0.0f );
    }

    // The functions below estimate the parametric edge lengths du and dv around any point on
    // a surface, i.e., if you wanted to tessellate at this point, what should be the spacing.
    // They are needed for displacement texture filtering on the surface.
    //
    // TODO: define something that is C0 continuous across clusters.  The lerp is not quite C0.
    inline __host__ __device__ float du( float2 uv ) const
    {
        float v = uv.y;
        return 1 / ( (1.0f - v) * float(edge_segments.x) + v * float(edge_segments.z) );
    }

    inline __host__ __device__ float dv( float2 uv ) const
    {
        float u = uv.x;
        return 1 / ( (1.0f - u) * float(edge_segments.w) + u * float(edge_segments.y) );
    }

    inline __host__ __device__ bool isEmpty() const
    {
        return gridSizeX() == 0 && gridSizeY() == 0;
    }
};


struct Cluster
{
    uint32_t iSurface      = 0u;  // index of the surface (patch) generating this cluster
    uint32_t nVertexOffset = 0u;  // vertex array index of this cluster's [0, 0]-corner
    ushort2  offset{ 0u, 0u };    // cluster's offset inside sample grid
    uchar2   size{ 0u, 0u };      // cluster's size

    __host__ __device__ inline Cluster( const uint32_t iSurface     = 0u,
                                        const uint32_t vertexOffset = 0u,
                                        const ushort2  offset       = { 0u, 0u },
                                        const uchar2   size         = { 0u, 0u } )
        : iSurface( iSurface )
        , nVertexOffset( vertexOffset )
        , offset( offset )
        , size( size )
    {
        assert( size.x < ( 1 << 4 ) );
        assert( size.y < ( 1 << 4 ) );
    }

    __device__ inline uchar2 clusterSize() const { return size; }

    __host__ __device__ inline uint32_t verticesPerCluster() const
    {
        return ( this->size.x + 1 ) * ( this->size.y + 1 );
    }

    __host__ __device__ inline uint32_t quadsPerCluster() const
    {
        return this->size.x * this->size.y;
    }

    __host__ __device__ inline uint32_t trianglesPerCluster() const { return 2 * quadsPerCluster(); }

    __host__ __device__ ushort2 linear2idx2d( uint16_t index_linear ) const
    {
        const uint16_t vertices_u = this->size.x + 1;
        return { static_cast<uint16_t>(index_linear % vertices_u),
            static_cast<uint16_t>(index_linear / vertices_u) };
    }
};


struct ClusterShadingData
{
    ushort4  m_surface_edge_segments;
    float2   m_surface_texcoords[4];
    uint32_t m_surface_id;

    uint32_t m_cluster_vertex_offset;
    ushort2  m_cluster_offset;  // within surface
    uchar2   m_cluster_size;
};

#ifndef _WIN32
#pragma GCC diagnostic pop
#endif
