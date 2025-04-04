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

#include <cluster_builder/clusterGatherers.cuh>
#include <cluster_builder/tilings.h>

#include <subdivision/SubdivisionSurfaceCUDA.h>

#include "../statistics.h"
#include "../utils.h"

#include <OptiXToolkit/Gui/Camera.h>

#include <span>


// functor to map the edge length to edge segment
struct SphereMetric
{
    float3   camera_pos = {0};
    uint32_t window_height = 0;
    float    tessellationRate = 1.f;
    otk::Matrix3x4 localToWorld = otk::Matrix3x4::affineIdentity();

    // Segments per unit length at point p.
    __host__ __device__ float segment_rate( LimitFrame limit_frame ) const
    {
        const float3 poi       = localToWorld * make_float4(limit_frame.point, 1.0f);
        const float  distance  = std::max( length( poi - camera_pos ), 0.01f );
        float        edge_rate = float( window_height ) * tessellationRate / distance;

        return edge_rate;
    };
};

struct EdgeVisibility
{
    otk::Matrix4x4 m_viewProjectMatrix = otk::Matrix4x4::identity();
    otk::Matrix3x4 m_localToWorld      = otk::Matrix3x4::affineIdentity();

    float3 m_cameraPos    = { 0.f, 0.f, 0.f };
    float2 m_viewportSize = { 0.f, 0.f };
    
    bool m_enableFrustumVisibility = true;
    bool m_enableBackfaceVisibility = true;

    __device__ void setSurface_warp( uint32_t i_surface ) { }

    __device__ float operator()( std::span<LimitFrame> limit_frames ) const
    {
        // 
        // expected limit_frames layout:
        //
        //          e2
        //
        //     p6---p5---p4
        //     |          |
        // e3  p7        p3  e1
        //     |          |
        //     p0---p1---p2
        //
        //          e0
        //

        const auto i_lane = threadIdx.x;

        // each lane is assigned to one of the 4 surface edges
        if( i_lane > 3 )
            return 1.f;
 
        uint8_t n_limit_frames = (uint8_t)limit_frames.size();

        float visibility = 1.f;

        // compute a normalized visibility factor for the 3 limit samples locations
        // along the edge against the frustum, hi-z and back-facing criteria.
        float3 pworld[3];
        float4 pclip[3];
 
        float frustum_factor = 0.f;
        for( uint8_t i = 0; i < 3; ++i )
        {
            pworld[i] = limit_frames[( i_lane * 2 + i ) % n_limit_frames].point;
            pclip[i] = m_viewProjectMatrix * make_float4( pworld[i], 1.f );

            float dist = ( 1.f / pclip[i].w ) * sqrtf( pclip[i].x * pclip[i].x + pclip[i].y * pclip[i].y );
            frustum_factor = max( frustum_factor, pclip[i].w < 0.f ? 1.f : smoothstep( 1.3f, 2.5f, dist ) );
        }

        if( m_enableFrustumVisibility )
            visibility *= ( 1.f - frustum_factor );

        if( visibility == 0.f )
            return visibility;

        if( m_enableBackfaceVisibility )
        {
            for( uint8_t i = 0; i < 3; ++i )
            {

                float3 t0 = limit_frames[( i_lane * 2 + i ) % limit_frames.size()].deriv1;
                float3 t1 = limit_frames[( i_lane * 2 + i ) % limit_frames.size()].deriv2;
                float3 nobj = cross( t0, t1 );
                float3 nworld = normalize( m_localToWorld * make_float4( nobj, 0.f ) );

                float cos_theta = dot( normalize( pworld[i] - m_cameraPos ), nworld );

                float backface_factor = smoothstep( .6f, 1.f, cos_theta );

                visibility *= ( 1.f - backface_factor );
            }
        }
        return visibility;
    }
};

struct EdgeSegments
{
    SphereMetric edgeRates;
    EdgeVisibility visibility;

    otk::Matrix3x4 localToWorld = otk::Matrix3x4::affineIdentity();
    float m_visibilityRateMultiplier = 1.0f;

    __device__ uint32_t operator()( std::span<LimitFrame> limit_frames ) const
    {
        const auto i_lane = threadIdx.x;

        float segment_rate = i_lane < 4 ? this->edgeRates.segment_rate( limit_frames[2 * i_lane + 1] ) : .0f;

        if( i_lane < limit_frames.size() )
            limit_frames[i_lane].point = localToWorld * make_float4( limit_frames[i_lane].point, 1.0f );

        float edge_length = length( limit_frames[( i_lane + 1 ) % limit_frames.size()].point
                                    - limit_frames[i_lane % limit_frames.size()].point );
        if( i_lane < 8 )
        {
            edge_length += __shfl_down_sync( 0xFF, edge_length, 1, limit_frames.size() );

            // get the edge lengths starting at the corner vertices (in 2*i_lane threads).
            edge_length = __shfl_sync( 0xFF, edge_length, 2 * i_lane, limit_frames.size() );
        }


        float edge_visibility = visibility( limit_frames );

        segment_rate *= ( m_visibilityRateMultiplier + edge_visibility * ( 1.f - m_visibilityRateMultiplier ) );

        return i_lane < 4 ? std::max(1u, static_cast<uint32_t>( lroundf( edge_length * segment_rate ) ) ) : 0;
    }
};



// A kernel for computing number of edge segments for a given surface (quad patch).
//
// clang-format off
template <typename SUBD_T>
#if NDEBUG  // necessary because DEBUG mode isn't able to fit this kernel to the bounds
__launch_bounds__( 128, 7 )
#endif
__global__ void surfaceTilingKernel( TessellatorConfig tessConfig, SUBD_T subd, EdgeSegments metric, PatchGatherer out )
// clang-format on
{
    const uint32_t i_warp = threadIdx.y;
    // one warp per surface
    const uint32_t i_surface = WARPS_PER_BLOCK * blockIdx.x + i_warp;

    if( !( i_surface < subd.number_of_surfaces() ) )
        return;

    if( !subd.hasLimit( i_surface ) )
        return;

    // Frustum "culling"
    metric.visibility.setSurface_warp( i_surface );

    // -------------------------------------------------------------------------
    // Evaluate corner and mid points for surface quad
    // -------------------------------------------------------------------------
    //
    // sample locations:
    //
    //          e2
    //
    //     p6---p5---p4
    //     |          |
    // e3  p7        p3  e1
    //     |          |
    //     p0---p1---p2
    //
    //          e0
    //
    const static float2   aUVs[]       = { { 0, 0 }, { 0.5, 0 }, { 1, 0 }, { 1, 0.5 },
                                           { 1, 1 }, { 0.5, 1 }, { 0, 1 }, { 0, 0.5 } };
    constexpr uint32_t n_uv_samples = std::size(aUVs);
    __shared__ LimitFrame samples[WARPS_PER_BLOCK * n_uv_samples];

    auto warp_samples = std::span(&samples[n_uv_samples * i_warp], n_uv_samples);

    subd.warpEvaluateBSplinePatch8( warp_samples, i_surface, std::span( aUVs ) );
    const uint32_t edge_segments = metric( warp_samples );  // i-th edge-segment (0..3) in i-th lane of current warp
    if( threadIdx.x < 4 )
        assert( edge_segments >= 1 );
    out.writeSurfaceWarp( subd, i_surface, edge_segments, warp_samples );
}

template <typename SUBD_T>
__global__ void patchPointsKernel( SUBD_T subd )
{
    const uint32_t i_warp = threadIdx.y;
    // one warp per surface
    const uint32_t i_surface = WARPS_PER_BLOCK * blockIdx.x + i_warp;

    if( !( i_surface < subd.number_of_surfaces() ) )
        return;

    if( !subd.hasLimit( i_surface ) )
        return;

    // Evaluate texcoords patch points
    {
        auto* patch_points = subd.texcoord_subd.patch_points[i_surface];
        subd.texcoord_subd.warpEvaluatePatchPoints( patch_points, i_surface );
    }

    // Evaluate subd patch points
    {
        auto* patch_points = subd.patch_points[i_surface];
        subd.warpEvaluatePatchPoints( patch_points, i_surface );
    }
}

template <typename SUBD_T>
void launchSubdClusterTiling( TessellatorConfig tessConfig, const SUBD_T& subd, const EdgeSegments& metric, PatchGatherer& out )
{
    auto& coarse_tessellation_timer = stats::clusterAccelSamplers.clusterTilingTime;
    coarse_tessellation_timer.start();

    const uint32_t n_surfaces  = subd.number_of_surfaces();
    const dim3     grid_shape  = { div_up( n_surfaces, WARPS_PER_BLOCK ), 1, 1 };  // assign warps to surfaces
    const dim3     block_shape = { 32, WARPS_PER_BLOCK, 1 };                       // a block of warps

    patchPointsKernel<<<grid_shape, block_shape>>>( subd );

    surfaceTilingKernel<<<grid_shape, block_shape>>>( tessConfig, subd, metric, out );

    coarse_tessellation_timer.stop();
    CUDA_SYNC_CHECK();
}

template <typename SUBD_T>
void computeClusterTiling(const TessellatorConfig &tessConfig,
                          const SUBD_T &subd,
                          const otk::Matrix3x4 &localToWorld,
                          PatchGatherer &out)
{
    const float tessFactor = tessConfig.coarseTessellationRate / tessConfig.fineTessellationRate;

    SphereMetric metric = { tessConfig.camera->getEye(), static_cast<uint32_t>( tessConfig.viewport_size.y ),
                            tessConfig.fineTessellationRate, localToWorld };

    EdgeVisibility visibility_metric{ tessConfig.camera->getViewProjectionMatrix(),
                                      localToWorld,
                                      tessConfig.camera->getEye(),
                                      make_float2( tessConfig.viewport_size ),
                                      tessConfig.enableFrustumVisibility,
                                      tessConfig.enableBackfaceVisibility };

    EdgeSegments edgeSegments{ metric, visibility_metric, localToWorld, tessFactor };

    launchSubdClusterTiling( tessConfig, subd, edgeSegments, out );
}


