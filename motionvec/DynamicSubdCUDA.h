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

#include <subdivision/SubdivisionPlanCUDA.h>
#include <subdivision/SubdivisionSurface.h>
#include <subdivision/TopologyMap.h>

#include <opensubdiv/tmr/surfaceDescriptor.h>
#include <opensubdiv/tmr/types.h>


using namespace OpenSubdiv;

// this class is based on SubdivisionSurfaceCUDA but computes patch points dynamically per lane,
// and also contains two sets of control points (usually for two frames of animation)

template <typename CONTROL_PT_T>
struct DynamicSubdCUDA
{
    std::span<Tmr::SurfaceDescriptor>            surface_descriptors;
    std::span<SubdivisionPlanCUDA>               plans;
    std::span<CONTROL_PT_T>                      control_points0;
    std::span<CONTROL_PT_T>                      control_points1;
    std::span<int>                               control_point_indices;
    std::span<uint16_t>                          material_bindings;

    static constexpr int isolation_level = 6; 

    __host__ 
    DynamicSubdCUDA( const SubdivisionSurface& subd, const std::span<CONTROL_PT_T>& positions0, const std::span<CONTROL_PT_T>& positions1 )
        : surface_descriptors{ subd.m_vertexDeviceData.surface_descriptors.span() }
        , plans{ subd.getTopologyMap()->d_plans.span() }
        , control_points0{ positions0 }
        , control_points1{ positions1 }
        , control_point_indices{ subd.m_vertexDeviceData.control_point_indices.span() }
        , material_bindings{ subd.d_materialBindings.span() }
        
    { ; }


    __device__ 
    const SubdivisionPlanCUDA& getPlan( uint32_t i_surface ) const
    {
        return plans[surface_descriptors[i_surface].GetSubdivisionPlanIndex()];
    }

    __device__ 
    bool hasLimit( uint32_t i_surface ) const { return surface_descriptors[i_surface].HasLimit(); }


    // evaluate both sets of control points; pass null pointer to skip
    __device__ 
    void evaluatePureBsplinePatch( LimitFrame* limit0, LimitFrame* limit1, float2 uv, uint32_t i_surface ) const
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

        if ( limit0 ) limit0->Clear();
        if ( limit1 ) limit1->Clear();

        static const Far::Index patchPointIndices[patchSize] = { 6, 7, 8, 9, 5, 0, 1, 10, 4, 3, 2, 11, 15, 14, 13, 12 };

        for( int i_weight = 0; i_weight < patchSize; ++i_weight )
        {

            Far::Index patchPointIndex = patchPointIndices[i_weight];

            Tmr::Index cpi = indices[patchPointIndex];
            if ( limit0 )
                limit0->AddWithWeight( control_points0[cpi], a_pt_weights[i_weight], a_du_weights[i_weight], a_dv_weights[i_weight] );
            if ( limit1 )
                limit1->AddWithWeight( control_points1[cpi], a_pt_weights[i_weight], a_du_weights[i_weight], a_dv_weights[i_weight] );
        }
    }

    
    __device__ 
    void evaluateBsplinePatch( LimitFrame* limit0, LimitFrame* limit1, float2 uv, uint32_t i_surface ) const
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

        if ( limit0 ) limit0->Clear();
        if ( limit1 ) limit1->Clear();

        assert( patchSize == node.GetPatchSize( quadrant, isolation_level ) );

        for( int i_weight = 0; i_weight < patchSize; ++i_weight )
        {
            Far::Index patchPointIndex = node.GetPatchPoint( i_weight, quadrant, isolation_level );
            assert( patchPointIndex < getPlan( i_surface )._numControlPoints );

            Tmr::Index cpi = indices[patchPointIndex];
            if ( limit0 )
                limit0->AddWithWeight( control_points0[cpi], a_pt_weights[i_weight], a_du_weights[i_weight], a_dv_weights[i_weight] );
            if ( limit1 )
                limit1->AddWithWeight( control_points1[cpi], a_pt_weights[i_weight], a_du_weights[i_weight], a_dv_weights[i_weight] );
        }
    }


    __device__ 
    void evaluateLimitSurface( LimitFrame* limit0, LimitFrame* limit1, float2 uv, uint32_t i_surface ) const
    {
        uint8_t quadrant            = 0;
        constexpr uint32_t numPatchPoints = 16;
        float   a_pt_weights[numPatchPoints] = { 0 };
        float   a_du_weights[numPatchPoints] = { 0 };
        float   a_dv_weights[numPatchPoints] = { 0 };

        const SubdivisionPlanCUDA& plan = getPlan( i_surface );

        SubdivisionPlanCUDA::Node node =
            plan.EvaluateBasis( uv, a_pt_weights, a_du_weights, a_dv_weights, &quadrant, isolation_level );

        if ( limit0 ) limit0->Clear();
        if ( limit1 ) limit1->Clear();

        assert( numPatchPoints == node.GetPatchSize( quadrant, isolation_level ) );

        const int numControlPoints = plan._numControlPoints;
        const auto stencilMatrix = cuda::std::mdspan( plan._stencilMatrix.data(), numPatchPoints, numControlPoints );

        const Tmr::SurfaceDescriptor& desc                     = surface_descriptors[i_surface];
        const Tmr::Index*             localControlPointIndices = &control_point_indices[desc.firstControlPoint];

        for ( int point = 0; point < numPatchPoints; ++point ) 
        {
            Far::Index patchPointIndex = node.GetPatchPoint( point, quadrant, isolation_level );
            assert( patchPointIndex >= numControlPoints );  // not a regular bspline surface

            // Build patch point and accumulate it

            if ( limit0 )
            {
                Vertex patchPoint0 = { 0.0f };
                for ( int i = 0; i < numControlPoints; ++i )
                {
                    patchPoint0 +=
                        control_points0[localControlPointIndices[i]] * stencilMatrix( patchPointIndex - numControlPoints, i );
                }
                limit0->AddWithWeight( patchPoint0, a_pt_weights[point], a_du_weights[point], a_dv_weights[point] );
            }

            if ( limit1 )
            {
                Vertex patchPoint1 = { 0.0f };
                for ( int i = 0; i < numControlPoints; ++i )
                {
                    patchPoint1 +=
                        control_points1[localControlPointIndices[i]] * stencilMatrix( patchPointIndex - numControlPoints, i );
                }
                limit1->AddWithWeight( patchPoint1, a_pt_weights[point], a_du_weights[point], a_dv_weights[point] );
            }
        }

    }

    __device__ 
    bool isBSplinePatch( uint32_t i_surface ) const
    {
        return getPlan( i_surface ).isBSplinePatch( isolation_level );
    }

    __device__ 
    bool isPureBSplinePatch( uint32_t i_surface ) const
    {
        const Tmr::SurfaceDescriptor& desc = surface_descriptors[i_surface];
        return desc.GetSubdivisionPlanIndex() == 0;
    }

    __device__ 
    void evaluate( uint32_t i_surface, float2 uv, LimitFrame& limit0, LimitFrame& limit1 ) const
    {
        
        if ( isPureBSplinePatch( i_surface) )
        {
            evaluatePureBsplinePatch( &limit0, &limit1, uv, i_surface );
        }
        else if( isBSplinePatch( i_surface ) )
        {
            evaluateBsplinePatch( &limit0, &limit1, uv, i_surface );
        }
        else {
            evaluateLimitSurface( &limit0, &limit1, uv, i_surface );
        }
    }
    
    __device__ 
    void evaluateSecond( uint32_t i_surface, float2 uv, LimitFrame& limit ) const
    {
        
        if ( isPureBSplinePatch( i_surface) )
        {
            evaluatePureBsplinePatch( nullptr, &limit, uv, i_surface );
        }
        else if( isBSplinePatch( i_surface ) )
        {
            evaluateBsplinePatch( nullptr, &limit, uv, i_surface );
        }
        else {
            evaluateLimitSurface( nullptr, &limit, uv, i_surface );
        }
    }


};




