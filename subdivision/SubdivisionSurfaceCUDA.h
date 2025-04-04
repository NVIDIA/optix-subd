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

#include "SubdivisionPlanCUDA.h"

#include <material/materialCuda.h>
#include <texture/bicubicTexture.h>

#include "TopologyMap.h"
#include <opensubdiv/tmr/surfaceDescriptor.h>
#include <opensubdiv/tmr/types.h>

// clang-format on

using namespace OpenSubdiv;


template <typename PATCH_PT_T>
struct PatchPointsCUDA
{
    std::span<PATCH_PT_T> patch_points;
    std::span<uint32_t>   patch_points_offsets;

    __device__ PATCH_PT_T* operator[]( uint32_t i_surface )
    {
        const uint32_t support_offset = patch_points_offsets[i_surface];
        return ( support_offset >= patch_points.size() ) ? nullptr : &patch_points[support_offset];
    }

    __device__ const PATCH_PT_T* operator[]( uint32_t i_surface ) const
    {
        const uint32_t support_offset = patch_points_offsets[i_surface];
        return ( support_offset >= patch_points.size() ) ? nullptr : &patch_points[support_offset];
    }
};


constexpr void evalLinearBasis( float u, float v, float weights[4], float du_weights[4], float dv_weights[4] )
{
    weights[0] = ( 1.0f - u ) * ( 1.0f - v );
    weights[1] = u * ( 1.0f - v );
    weights[2] = u * v;
    weights[3] = ( 1.0f - u ) * v;

    du_weights[0] = ( -1.0f + v );
    du_weights[1] = ( 1.0f - v );
    du_weights[2] = v;
    du_weights[3] = -v;

    dv_weights[0] = ( -1.0f + u );
    dv_weights[1] = ( -u );
    dv_weights[2] = u;
    dv_weights[3] = ( 1.0f - u );
}

template <typename CONTROL_PT_T, typename L>
struct SubdLinearCUDA
{
    std::span<Tmr::LinearSurfaceDescriptor> surface_descriptors;
    std::span<CONTROL_PT_T>                 control_points;
    std::span<int>                          control_point_indices;
    PatchPointsCUDA<TexCoord>               patch_points;

    __device__ void warpEvaluatePatchPoints( CONTROL_PT_T* patch_points, uint32_t i_surface ) const
    {
        const Tmr::LinearSurfaceDescriptor desc    = surface_descriptors[i_surface];
        const Tmr::LocalIndex              subface = desc.GetQuadSubfaceIndex();

        if( subface == Tmr::LOCAL_INDEX_INVALID )
            return;  // no patch points needed for quad face

        const Tmr::Index* indices = &control_point_indices[desc.firstControlPoint];

        // Center point

        const uint16_t faceSize = desc.GetFaceSize();
        assert( faceSize <= 32 );  // warp reduce limited to 32 values
        float2 center = ( threadIdx.x < faceSize ) ? control_points[indices[threadIdx.x]].uv : float2{ 0, 0 };
        constexpr unsigned int WARP_ALL = 0xffffffff;
        for( int offset = 16; offset > 0; offset /= 2 )
        {
            center.x += __shfl_down_sync( WARP_ALL, center.x, offset );
            center.y += __shfl_down_sync( WARP_ALL, center.y, offset );
        }

       if( threadIdx.x == 0 )
            patch_points[0].uv = center / faceSize;

        // Edge midpoints

        if( threadIdx.x < faceSize )
        {
            CONTROL_PT_T a                   = control_points[indices[threadIdx.x]];
            CONTROL_PT_T b                   = control_points[indices[( threadIdx.x + 1 ) % faceSize]];
            patch_points[threadIdx.x + 1].uv = 0.5f * ( a.uv + b.uv );
        }
    }

    __device__ void evaluate( L& limit, float2 uv, uint32_t i_surface, const CONTROL_PT_T* patch_points ) const
    {
        const Tmr::LinearSurfaceDescriptor desc = surface_descriptors[i_surface];

        const uint16_t        faceSize = desc.GetFaceSize();
        const Tmr::LocalIndex subface  = desc.GetQuadSubfaceIndex();

        const Tmr::Index* indices = &control_point_indices[desc.firstControlPoint];

        float point_weights[4], du_weights[4], dv_weights[4];
        evalLinearBasis( uv.x, uv.y, point_weights, du_weights, dv_weights );

        limit.Clear();
        for( int k = 0; k < 4; ++k )
        {
            Tmr::Index patchPointIndex = GetPatchPoint( k, faceSize, subface );
            if( patchPointIndex < faceSize )
            {
                limit.AddWithWeight( control_points[indices[patchPointIndex]], point_weights[k], du_weights[k], dv_weights[k] );
            }
            else
            {
                limit.AddWithWeight( patch_points[patchPointIndex - faceSize], point_weights[k], du_weights[k], dv_weights[k] );
            }
        }
    }

  protected:
    __device__ inline Tmr::Index
    /*LinearSurfaceDescriptor::*/
    GetPatchPoint( int pointIndex, uint16_t faceSize, Tmr::LocalIndex subfaceIndex ) const
    {
        if( subfaceIndex == Tmr::LOCAL_INDEX_INVALID )
        {
            assert( pointIndex < faceSize );
            return pointIndex;
        }
        else
        {
            assert( pointIndex < 4 );
            // patch point indices layout (N = faceSize) :
            // [ N control points ]
            // [ 1 face-point ]
            // [ N edge-points ]
            int N = faceSize;
            switch( pointIndex )
            {
                case 0:
                    return subfaceIndex;
                case 1:
                    return N + 1 + subfaceIndex;  // edge-point after
                case 2:
                    return N;  // central face-point
                case 3:
                    return N + ( subfaceIndex > 0 ? subfaceIndex : N );
            }
        }
        return Tmr::INDEX_INVALID;
    }
};


struct PatchWeightArrays
{
    float* a_pt_weights;
    float* a_du_weights;
    float* a_dv_weights;
};

template <typename CONTROL_PT_T, typename PATCH_PT_T>
struct SubdCUDA
{
    std::span<Tmr::SurfaceDescriptor>            surface_descriptors;
    std::span<SubdivisionPlanCUDA>               plans;
    std::span<CONTROL_PT_T>                      control_points;
    std::span<int>                               control_point_indices;
    PatchPointsCUDA<PATCH_PT_T>                  patch_points;
    SubdLinearCUDA<TexCoord, TexCoordLimitFrame> texcoord_subd;

    static constexpr int isolation_level = 6;


    __host__ SubdCUDA( const SubdivisionSurface& subd )
        : surface_descriptors{ subd.m_vertexDeviceData.surface_descriptors.span() }
        , plans{ subd.getTopologyMap()->d_plans.span() }
        , control_points{ subd.d_positions.span() }
        , control_point_indices{ subd.m_vertexDeviceData.control_point_indices.span() }
        , patch_points{ subd.m_vertexDeviceData.patch_points.span(), subd.m_vertexDeviceData.patch_points_offsets.span() }
        , texcoord_subd{ subd.m_texcoordDeviceData.surface_descriptors.span(),
                         subd.d_texcoords.span(),
                         subd.m_texcoordDeviceData.control_point_indices.span(),
                         { subd.m_texcoordDeviceData.patch_points.span(), subd.m_texcoordDeviceData.patch_points_offsets.span() } }
    { ; }

    __host__ __device__ uint32_t number_of_surfaces() const { return surface_descriptors.size(); }
    __host__ __device__ uint32_t patchCount() const { return surface_descriptors.size(); }

    __device__ const SubdivisionPlanCUDA& getPlan( uint32_t i_surface ) const
    {
        return plans[surface_descriptors[i_surface].GetSubdivisionPlanIndex()];
    }

    __device__ bool hasLimit( uint32_t i_surface ) const { return surface_descriptors[i_surface].HasLimit(); }

    __device__ void evaluateBsplinePatch( LimitFrame& limit, float2 uv, uint32_t i_surface ) const
    {
        constexpr uint32_t patchSize{16};
        float   a_pt_weights[patchSize] = { 0 };
        float   a_du_weights[patchSize] = { 0 };
        float   a_dv_weights[patchSize] = { 0 };

        uint8_t                   quadrant = 0;
        SubdivisionPlanCUDA::Node node =
            getPlan( i_surface )
                .EvaluateBasis( uv, a_pt_weights, a_du_weights, a_dv_weights, &quadrant, isolation_level );

        const Tmr::SurfaceDescriptor& desc    = surface_descriptors[i_surface];
        const Tmr::Index*             indices = &control_point_indices[desc.firstControlPoint];

        limit.Clear();

        assert( patchSize == node.GetPatchSize( quadrant, isolation_level ) );

        for( int i_weight = 0; i_weight < patchSize; ++i_weight )
        {
            Far::Index patchPointIndex = node.GetPatchPoint( i_weight, quadrant, isolation_level );
            assert( patchPointIndex < getPlan( i_surface )._numControlPoints );

            Tmr::Index    cpi = indices[patchPointIndex];
            Vertex const* vtx = &control_points[cpi];  // regular face
            limit.AddWithWeight( *vtx, a_pt_weights[i_weight], a_du_weights[i_weight], a_dv_weights[i_weight] );
        }
    }

    __device__ void warpEvaluateBsplinePatch( std::span<LimitFrame> limits, std::span<const float2> sts, uint32_t i_surface ) const
    {
        assert( sts.size() <= 8 );
        const uint32_t i_lane = threadIdx.x;
        const float2   st     = i_lane % 8 < sts.size() ? sts[i_lane % 8] : make_float2( 0, 0 );
        assert( limits.size() >= sts.size() );

        LimitFrame                limit;
        SubdivisionPlanCUDA::Node node      = getPlan( i_surface ).getRootNode();
        assert( 16 == node.GetPatchSize( 0 ) );
        Tmr::NodeDescriptor       node_desc = node.GetDescriptor();

        const float sharpness = node_desc.HasSharpness() ? node.GetSharpness() : 0.f;

        Far::PatchParam param;
        param.Set( Tmr::INDEX_INVALID, node_desc.GetU(), node_desc.GetV(), 0, false, node_desc.GetBoundaryMask(), 0, true );
        const int boundaryMask = param.GetBoundary();

        float w_s[4], w_t[4], w_ds[4], w_dt[4];
        evalBSplineCurve( st.x, w_s, w_ds, 0 );
        evalBSplineCurve( st.y, w_t, w_dt, 0 );

        if( boundaryMask )
        {
            if( sharpness > 0.0f )
                adjustCreases( st, w_s, w_t, w_ds, w_dt, boundaryMask, sharpness );
            else
                adjustBoundaries( st, w_s, w_t, w_ds, w_dt, boundaryMask );
        }

        const Tmr::SurfaceDescriptor& desc    = surface_descriptors[i_surface];
        const Tmr::Index*             indices = &control_point_indices[desc.firstControlPoint];

        const auto&     plan            = getPlan( i_surface );
        const uint32_t* tree            = plan._tree.data();
        const int       i_patch_pt_base = tree[node.patchPointsOffset()];

        const auto i      = i_lane / 8;
        auto       w_t_i  = w_t[i];
        auto       w_dt_i = w_dt[i];

        for( auto j = 0; j < 4; ++j )
        {
            const int  i_weight        = 4 * i + j;
            Far::Index patchPointIndex = plan._patchPoints[i_patch_pt_base + i_weight];
            assert( patchPointIndex < plan._numControlPoints );
            Tmr::Index    cpi = indices[patchPointIndex];
            Vertex const* vtx = &control_points[cpi];  // regular face

            limit.point += ( w_t_i * w_s[j] ) * vtx->point;
            limit.deriv1 += ( w_t_i * w_ds[j] ) * vtx->point;
            limit.deriv2 += ( w_dt_i * w_s[j] ) * vtx->point;
        }
        for( uint32_t i_coord = 0; i_coord < 9; ++i_coord )
        {
            limit[i_coord] += __shfl_down_sync( 0xFFFFFFFF, limit[i_coord], 8 );
            limit[i_coord] += __shfl_down_sync( 0xFFFFFFFF, limit[i_coord], 16 );
        }
        if( i_lane < limits.size() )
            limits[i_lane] = limit;
    }

    __device__ void evaluatePureBsplinePatch( LimitFrame& limit, float2 uv, uint32_t i_surface ) const
    {
        constexpr uint32_t patchSize{16};
        float   a_pt_weights[patchSize] = { 0 };
        float   a_du_weights[patchSize] = { 0 };
        float   a_dv_weights[patchSize] = { 0 };

        EvalBasisBSpline<true>( uv, a_pt_weights, a_du_weights, a_dv_weights,
                                0,    // boundary mask
                                0.0f  // sharpness
        );

        const Tmr::SurfaceDescriptor& desc    = surface_descriptors[i_surface];
        const Tmr::Index*             indices = &control_point_indices[desc.firstControlPoint];

        limit.Clear();

        static const Far::Index patchPointIndices[patchSize] = { 6, 7, 8, 9, 5, 0, 1, 10, 4, 3, 2, 11, 15, 14, 13, 12 };

        for( int i_weight = 0; i_weight < patchSize; ++i_weight )
        {

            Far::Index patchPointIndex = patchPointIndices[i_weight];

            Tmr::Index    cpi = indices[patchPointIndex];
            Vertex const* vtx = &control_points[cpi];  // regular face
            limit.AddWithWeight( *vtx, a_pt_weights[i_weight], a_du_weights[i_weight], a_dv_weights[i_weight] );
        }
    }

    __device__ const Vertex& patchPoint( cuda::std::array<uint32_t, 2> ij, uint32_t i_surface ) const
    {
        static const Far::Index patchPointIndices[] = { 6, 7, 8, 9, 5, 0, 1, 10, 4, 3, 2, 11, 15, 14, 13, 12 };
        int                     i_weight            = 4 * ij[1] + ij[0];
        Far::Index              patchPointIndex     = patchPointIndices[i_weight];

        const Tmr::SurfaceDescriptor& desc    = surface_descriptors[i_surface];
        const Tmr::Index*             indices = &control_point_indices[desc.firstControlPoint];
        Tmr::Index                    cpi     = indices[patchPointIndex];
        return control_points[cpi];  // regular face
    }

    // Warp-parallel pure b-spline patch evaluator for up to 8 sample locations.
    //
    // Inputs:
    //    uvs - a span of up to 8 sample locations
    //    i_surface - subd object surface index
    //
    // Returns:
    //    limits - a span of LimitFrames with enough space to store the resulting limits (same a uvs.size()).
    //
    __device__ void warpEvaluatePureBsplinePatch8( std::span<LimitFrame> limits, std::span<const float2> uvs, uint32_t i_surface ) const
    {
        assert( uvs.size() <= 8 );
        const uint32_t i_lane = threadIdx.x;
        float2         uv     = i_lane % 8 < uvs.size() ? uvs[i_lane % 8] : make_float2( 0, 0 );
        assert( limits.size() >= uvs.size() );
        auto w_s = cubicBSplineWeight( uv.x, i_lane / 8 );  // weights for s-direction in warp w_s_0, ..., w_s_0, w_s_1, ..., w_s_1, ... w_s_3, ... w_s_3
        auto w_ds = cubicBSplineDerivativeWeight( uv.x, i_lane / 8 );  // weights for s_direction derivative, warp layout as for w_s

        LimitFrame limit;
        for( uint32_t j = 0; j < 4; ++j )
        {
            auto w_t  = cubicBSplineWeight( uv.y, j );
            auto w_dt = cubicBSplineDerivativeWeight( uv.y, j );

            const Vertex& vtx = patchPoint( { i_lane / 8, j }, i_surface );

            for( uint32_t i_coord = 0; i_coord < 3; ++i_coord )
            {
                auto vertex_component = vtx[i_coord];
                limit[0 + i_coord] += w_s * w_t * vertex_component;   // point
                limit[3 + i_coord] += w_ds * w_t * vertex_component;  // deriv1
                limit[6 + i_coord] += w_s * w_dt * vertex_component;  // deriv2
            }
        }

        for( uint32_t i_coord = 0; i_coord < 9; ++i_coord )
        {
            limit[i_coord] += __shfl_down_sync( 0xFFFFFFFF, limit[i_coord], 8 );
            limit[i_coord] += __shfl_down_sync( 0xFFFFFFFF, limit[i_coord], 16 );
        }
        if( i_lane < limits.size() )
            limits[i_lane] = limit;
    }

    __device__ void evaluateLimitSurface( LimitFrame& limit, float2 uv, uint32_t i_surface ) const
    {
        uint8_t quadrant            = 0;
        constexpr uint32_t patchSize{16};
        float   a_pt_weights[patchSize] = { 0 };
        float   a_du_weights[patchSize] = { 0 };
        float   a_dv_weights[patchSize] = { 0 };

        SubdivisionPlanCUDA::Node node =
            getPlan( i_surface )
                .EvaluateBasis( uv, a_pt_weights, a_du_weights, a_dv_weights, &quadrant, isolation_level );

        const uint16_t numControlPoints = getPlan( i_surface )._numControlPoints;

        limit.Clear();
        assert( patchSize == node.GetPatchSize( quadrant, isolation_level ) );

        for( int i_weight = 0; i_weight < patchSize; ++i_weight )
        {
            Far::Index    patchPointIndex = node.GetPatchPoint( i_weight, quadrant, isolation_level );
            int           index           = patchPointIndex - numControlPoints;
            Vertex const* patchPoint      = &patch_points[i_surface][index];  // not a regular face
            limit.AddWithWeight( *patchPoint, a_pt_weights[i_weight], a_du_weights[i_weight], a_dv_weights[i_weight] );
        }
    }

    __device__ LimitFrame evaluate( uint32_t i_surface, float2 uv, float du = 0.0f, float dv = 0.0f ) const
    {
        LimitFrame limit;
        if( isPureBSplinePatch( i_surface ) )
            evaluatePureBsplinePatch( limit, uv, i_surface );
        else if( isBSplinePatch( i_surface ) )
            evaluateBsplinePatch( limit, uv, i_surface );
        else
            evaluateLimitSurface( limit, uv, i_surface );
        return limit;
    }

    // Evaluate up to 8 limits with one warp
    __device__ void warpEvaluateBSplinePatch8( std::span<LimitFrame>   limits,
                                               uint32_t                      i_surface,
                                               std::span<const float2> uvs ) const
    {
        if( isPureBSplinePatch( i_surface ) )
            warpEvaluatePureBsplinePatch8( limits, uvs, i_surface );
        else if( isBSplinePatch( i_surface ) )
            warpEvaluateBsplinePatch( limits, uvs, i_surface );
        else  // there is no warp parallel implementation for non-bspline patches falling back to single thread
        {
            const auto i_lane = threadIdx.x;
            if( i_lane < uvs.size() )  // no warp parallel limit eval use single lane
            {
                evaluateLimitSurface( limits[i_lane], uvs[i_lane], i_surface );
            }
        }
    }

    __device__ bool isBSplinePatch( uint32_t i_surface ) const
    {
        return getPlan( i_surface ).isBSplinePatch( isolation_level );
    }

    __device__ bool isPureBSplinePatch( uint32_t i_surface ) const
    {
        const Tmr::SurfaceDescriptor& desc = surface_descriptors[i_surface];
        return desc.GetSubdivisionPlanIndex() == 0;
    }

    __device__ void warpEvaluatePatchPoints( PATCH_PT_T* patch_points, uint32_t i_surface ) const
    {
        if( isPureBSplinePatch( i_surface ) || isBSplinePatch( i_surface ) )
        {
            // No patch points needed for bsplines
            return;
        }

        const Tmr::SurfaceDescriptor& desc    = surface_descriptors[i_surface];
        const Tmr::Index*             indices = &control_point_indices[desc.firstControlPoint];

        getPlan( i_surface ).WarpEvaluatePatchPoints( patch_points, control_points, indices, isolation_level );
    }

    __device__ uint16_t materialId(uint32_t i_surface) const
    {
        return 0;
    }
};


template <typename CONTROL_PT_T, typename PATCH_PT_T>
struct DisplacedSubdCUDA: SubdCUDA<CONTROL_PT_T, PATCH_PT_T>
{
    const float global_scale = 1.0f;
    const float global_bias  = 0.0f;
    const float global_displacement_filter_scale = 0.0f;
    const float global_displacement_filter_mip_bias = 0.0f;

    const MaterialCuda* materials = nullptr;
    std::span<uint16_t> material_bindings;

    __host__ DisplacedSubdCUDA( const SubdivisionSurface& subd, float dispScale, float dispBias, float dispFilterScale, float dispFilterBias, const MaterialCuda* materials )
        : SubdCUDA<CONTROL_PT_T, PATCH_PT_T>( subd )
        , global_scale{ dispScale }
        , global_bias{ dispBias }
        , global_displacement_filter_scale{ dispFilterScale }
        , global_displacement_filter_mip_bias{ dispFilterBias }
        , materials{ materials }
        , material_bindings{ subd.d_materialBindings.span() }
    { ; }

    __device__ LimitFrame displace( const LimitFrame&             limit,
                                    uint32_t                      i_surface,
                                    const DisplacementSampler&    displacement_sampler,
                                    float2                        uv,
                                    float                         du = 0.0f,
                                    float                         dv = 0.0f ) const
    {

        auto sample = [&]( DisplacementSampler sampler, float2 uv, float2 duv0, float2 duv1, float  mipBias ) -> float
        {
            float d = BicubicSampling::tex2D_bicubic_grad_fast<float>( *sampler.tex, uv.x, uv.y, duv0, duv1, mipBias );
            return sampler.scale * global_scale * ( sampler.bias + global_bias + d );
        };

        // compute subd limit and normal
        const float3 normal = normalize( cross( limit.deriv1, limit.deriv2 ) );

        // get displacement
        TexCoordLimitFrame texcoord;
        this->texcoord_subd.evaluate( texcoord, uv, i_surface, this->texcoord_subd.patch_points[i_surface] );
        
        const float filterScale = global_displacement_filter_scale;
        float2 grad_du = filterScale * du * texcoord.deriv1;
        float2 grad_dv = filterScale * dv * texcoord.deriv2;
        const float filterBias = global_displacement_filter_mip_bias;
        float displacement = sample( displacement_sampler, texcoord.uv, grad_du, grad_dv, filterBias );

        
        // compute derivatives of displacement map, (dD/du) and (dD/dv) from finite differences
        const float2 delta = { std::max( du, 0.01f), std::max( dv, 0.01f ) };
        float2 texcoord_du         = texcoord.uv + delta.x * texcoord.deriv1;
        float2 texcoord_dv         = texcoord.uv + delta.y * texcoord.deriv2;
        float  displacement1       = sample( displacement_sampler, texcoord_du, grad_du, grad_dv, filterBias );
        float  displacement2       = sample( displacement_sampler, texcoord_dv, grad_du, grad_dv, filterBias );
        float  dDdu = ( displacement1 - displacement ) / delta.x;
        float  dDdv = ( displacement2 - displacement ) / delta.y;

        // compute displaced partial derivatives
        const float3 dpdu = limit.deriv1 + dDdu * normal;
        const float3 dpdv = limit.deriv2 + dDdv * normal;

        return { limit.point + displacement * normal, dpdu, dpdv };
    }

    __device__ LimitFrame evaluate( uint32_t iSurface, float2 uv, float du = 0.0f, float dv = 0.0f ) const
    {
        const LimitFrame limit = SubdCUDA<CONTROL_PT_T, PATCH_PT_T>::evaluate(iSurface, uv);
        const auto displacement = displacementSampler(iSurface);
        if (!displacement.tex) // if no sampler texture return undisplaced
            return limit;

        return displace( limit, iSurface, displacement, uv, du, dv );
    }

        // Evaluate up to 8 limits with one warp
    __device__ void warpEvaluateBSplinePatch8( std::span<LimitFrame>   limits,
                                               uint32_t                      iSurface,
                                               std::span<const float2> uvs ) const
    {
        SubdCUDA<CONTROL_PT_T, PATCH_PT_T>::warpEvaluateBSplinePatch8( limits, iSurface, uvs );
        const auto displacement = displacementSampler( iSurface );
        const auto iLane = threadIdx.x;
        if( iLane < uvs.size() )  // no warp parallel limit eval use single lane
        {
            if( !displacement.tex ) // no displacement sampler
                return;
            auto limit = displace( limits[iLane], iSurface, displacement, uvs[iLane] );
            limits[iLane] = limit;
        }
    }

    __device__ uint16_t materialId(uint32_t i_surface) const
    {
        return this->material_bindings[i_surface];
    }

    __device__ DisplacementSampler displacementSampler(uint32_t i_surface) const
    {
        auto id = materialId( i_surface );
        const MaterialCuda& material = materials[id];
        return material.displacementSampler;
    }
};

