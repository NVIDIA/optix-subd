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

#include "motionvec.h"

#include "DynamicSubdCUDA.h"
#include "GBuffer.cuh"

#include <material/materialCuda.h>
#include <shadingTypes.h>
#include <material/materialCache.h>
#include <scene/scene.h>
#include <statistics.h>
#include <texture/bicubicTexture.h>
#include <utils.h>

#include <OptiXToolkit/Gui/Camera.h>
#include <OptiXToolkit/Util/Exception.h>

#include <array>
#include <iostream>
#include <span>
#include <vector>

struct MotionVecPass::SubdInstance
{
    const otk::Matrix3x4 localToWorld = otk::Matrix3x4::affineIdentity();
    const otk::Matrix3x4 worldToLocal = otk::Matrix3x4::affineIdentity();
    const uint32_t       meshIndex = ~uint32_t( 0 );
};

struct DeviceCamera
{
    otk::Matrix4x4 view;
    otk::Matrix4x4 viewInv;
    otk::Matrix4x4 proj;
    otk::Matrix4x4 projInv;
    uint2          dims   = { 0, 0 };

    __host__ 
    DeviceCamera( const otk::Camera& cam, uint2 dims )
        : dims( dims )
    {
        view = cam.getViewMatrix();
        viewInv = view.inverse();
        proj = cam.getProjectionMatrix();
        projInv = proj.inverse();
    }

    __device__
    float3 unprojectPixelToWorld_lineardepth(float2 pixel, float z )
    {
        float4 v_ndc  = make_float4( ( pixel / make_float2( dims.x, dims.y ) ) * 2.f - 1.f, 0.f, 1.f );
        float4 v_clip = v_ndc * z;
        float4 v_view = projInv * v_clip;
        float4 vw     = viewInv * make_float4( v_view.x, v_view.y, v_view.z, 1.0f );

        return make_float3( vw.x, vw.y, vw.z );
    }

    __device__ float3 unprojectPixelToWorld_hwdepth( float2 pixel, float zNDC )
    {
        // zNDC --> zCam
        float* m = proj.getData();
        float  A = m[10];
        float  B = m[11];
        float  C = m[14];
        float  zLinear = -B / ( C * zNDC - A );

        return unprojectPixelToWorld_lineardepth( pixel, zLinear );
    }

    __device__
    float3 unprojectPixelToWorldDirection( float2 pixel )
    {
        float4 v_ndc =
            make_float4( ( pixel / make_float2( dims.x, dims.y ) ) * 2.f - 1.f,
                         0.f, 1.f );
        float4 v_clip = v_ndc;
        float4 v_cam  = projInv * v_clip;
        float4 vw = viewInv * make_float4( v_cam.x, v_cam.y, v_cam.z, 0.0f );

        return make_float3( vw.x, vw.y, vw.z );
    }


    __device__ float2 projectWorldToPixel( float3 p )
    {
        const float4 p_cam    = view * make_float4( p.x, p.y, p.z, 1.0f );
        const float4 p_clip   = proj * p_cam;
        const float4 p_ndc    = p_clip / p_clip.w;
        const float2 p_screen = 0.5f * ( make_float2( p_ndc.x, p_ndc.y ) + 1.0f );
        const float2 pixel    = { p_screen.x * dims.x, p_screen.y * dims.y };

        return pixel;
    }

    __device__ float2 projectWorldDirectionToPixel( float3 v )
    {
        const float4 v_cam    = view * make_float4( v.x, v.y, v.z, 0.0f );
        const float4 v_clip   = proj * v_cam;
        const float4 v_ndc    = v_clip / v_clip.w;
        const float2 v_screen = 0.5f * ( make_float2( v_ndc.x, v_ndc.y ) + 1.f );
        const float2 pixel    = { v_screen.x * dims.x, v_screen.y * dims.y };

        return pixel;
    }


};

__device__ 
float3 transformPoint( float3 p, const otk::Matrix3x4& mat)
{
    float3 out = mat * make_float4( p.x, p.y, p.z, 1.0f );
    return out;
}


__device__ 
DisplacementSampler resolveDisplacementSampler( const MaterialCuda& mtl, float displacementScale, float displacementBias )
{
    if ( mtl.displacementSampler.tex ) 
    {
        return DisplacementSampler { 
            .tex = mtl.displacementSampler.tex, 
            .bias = mtl.displacementSampler.bias + displacementBias,
            .scale = mtl.displacementSampler.scale * displacementScale
        };
    }
    return DisplacementSampler{0};
}

template <typename T, MotionVecDisplacementMode DisplacementMode >
__global__ void motionVecKernel( const std::span<DynamicSubdCUDA<T>> subds,
                                 const std::span<MotionVecPass::SubdInstance> instances,
                                 DeviceCamera                        camera,
                                 DeviceCamera                        prevCamera,
                                 float2                              jitter,
                                 std::span<HitResult>                hits,
                                 const RwFloat                       depths,
                                 const MaterialCuda*                 materials,
                                 float                               displacementScale,
                                 float                               displacementBias,
                                 RwFloat2                   motionvecs )
{
    const uint2 idx = { blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y };
    if( idx.x >= motionvecs.m_size.x || idx.y >= motionvecs.m_size.y )
        return;

    const HitResult& hit = hits[idx.y * motionvecs.m_size.x + idx.x];

    const float2 curPixel = make_float2( idx.x + jitter.x, idx.y + jitter.y );

    // Check for miss
    if ( hit.instanceId == ~uint32_t(0)  ) 
    {
        // Re-project env map direction
        const float3 vw  = camera.unprojectPixelToWorldDirection( curPixel );
        const float2 prevPixel = prevCamera.projectWorldDirectionToPixel( vw );
        const float2 flow = curPixel - prevPixel;
        gbuffer::write( curPixel - prevPixel, motionvecs, idx);
        return;
    }

    const float depth = gbuffer::read( depths, idx );
    const float3 Pw    = camera.unprojectPixelToWorld_lineardepth( curPixel, depth );


    // Check for non-subd geometry
    if( hit.surfaceIndex == ~uint32_t( 0 ) )
    {
        //  No deformation, only camera motion
        float2 prevPixel = prevCamera.projectWorldToPixel( Pw );
        gbuffer::write( curPixel - prevPixel, motionvecs, idx);
        return;
    }

    const MotionVecPass::SubdInstance& instance = instances[hit.instanceId];

    assert( instance.meshIndex < subds.size() );
    const DynamicSubdCUDA<T>& subd = subds[instance.meshIndex];

    if ( !subd.hasLimit( hit.surfaceIndex ) ) 
    {
        gbuffer::write( make_float2(0), motionvecs, idx );
        return;
    }

    float2 prevPixel = { 0 };
    
    // Check for deforming subd
    if( subd.control_points1.size() )
    {

        if ( DisplacementMode == MOTION_VEC_DISPLACEMENT_FROM_MATERIAL )
        {
            LimitFrame limit1;
            subd.evaluateSecond( hit.surfaceIndex, { hit.u, hit.v }, limit1 );

            float3 displacementVec = { 0 };

            // Resolve displacement sampler; can be global or material w/ global overrides
            DisplacementSampler sampler = {0};
            if( materials && subd.material_bindings.size() )
            {
                uint16_t materialIndex = subd.material_bindings[hit.surfaceIndex];
                if( materialIndex != uint16_t( ~uint16_t( 0 ) ) )
                {
                    const MaterialCuda& mtl = materials[materialIndex];
                    sampler = resolveDisplacementSampler( mtl, displacementScale, displacementBias );
                }
            }
            // Evaluate displacement using the resolved sampler. TODO: reconstruct gradients?
            if( sampler.tex )
            {
                float d = BicubicSampling::tex2D_bicubic_grad_fast<float>( *sampler.tex, hit.texcoord.x, hit.texcoord.y );

                float  displacement = sampler.scale * ( d + sampler.bias );
                float3 normal       = normalize( cross( limit1.deriv1, limit1.deriv2 ) );
                displacementVec     = displacement * normal;
            }

            float3 PdispW = transformPoint( limit1.point + displacementVec, instance.localToWorld );
            prevPixel     = prevCamera.projectWorldToPixel( PdispW );

        }
        else 
        {
            LimitFrame limit0, limit1;
            subd.evaluate( hit.surfaceIndex, { hit.u, hit.v }, limit0, limit1 );

            // Evaluate displacement using the delta between hit point and subd limit point
            // This captures rigid transforms but maybe tangent space would be better?
            float3 displacementVec = transformPoint( Pw, instance.worldToLocal) - limit0.point;

            float3 PdispW = transformPoint( limit1.point + displacementVec, instance.localToWorld );
            prevPixel     = prevCamera.projectWorldToPixel( PdispW );
        }

        
    }
    else
    {
        // No deformation, only camera motion
        prevPixel = prevCamera.projectWorldToPixel( Pw );
    }

    gbuffer::write( curPixel - prevPixel, motionvecs, idx );

}

MotionVecPass::MotionVecPass( MotionVecDisplacementMode mode )
    : m_displacementMode( mode )
{}

MotionVecPass::~MotionVecPass()
{}


static inline otk::Matrix3x4 inverse3x4( const otk::Matrix3x4& m )
{
    return otk::Matrix3x4::makeFrom( otk::Matrix4x4::makeFrom( m ).inverse() );
}

void MotionVecPass::run( const Scene&               scene,
                         float                      displacementScale,
                         float                      displacementBias,
                         const otk::Camera&         camera,
                         const otk::Camera&         prevCamera,
                         float2                     jitter,
                         const CuBuffer<HitResult>& hits,
                         GBuffer&       gbuffer )
{

    OTK_REQUIRE( gbuffer.m_motionvecs.m_size.x * gbuffer.m_motionvecs.m_size.y == hits.size() );

    std::span<Instance const> instances = scene.getSubdMeshInstances();

    auto& subds = scene.getSubdMeshes();

    {
        // Convert subds and instances to device data
        m_subds.clear();
        m_subds.reserve( subds.size() );
        for ( size_t i = 0; i < subds.size(); ++i )
        {
            if( subds[i]->d_positionsCached.size() )
            {
                m_subds.push_back( DynamicSubdCUDA<Vertex>( *subds[i], subds[i]->d_positions.span(),
                                                            subds[i]->d_positionsCached.span() ) );
            }
            else
            {
                // Placeholder for non-deforming subd, kernel won't evaluate this
                m_subds.push_back( DynamicSubdCUDA<Vertex>( *subds[i], std::span<Vertex>(), std::span<Vertex>() ) );
            }
        }
        m_deviceSubds.upload( m_subds );

        m_instances.clear();
        m_instances.reserve( instances.size() );
        for ( size_t i = 0; i < instances.size(); ++i )
        {
            const Instance& inst = instances[i];
            m_instances.push_back( SubdInstance { 
                    .localToWorld= inst.localToWorld.matrix3x4(), 
                    .worldToLocal = inverse3x4( inst.localToWorld.matrix3x4() ),
                    .meshIndex = inst.meshID 
                    } );
        }
        m_deviceInstances.upload( m_instances );


        // Evaluate motion vecs

        uint2 dims        = gbuffer.m_motionvecs.m_size;
        dim3 block_shape = { 16u, 16u, 1u };
        dim3 grid_shape  = { div_up( uint32_t( dims.x ), block_shape.x ),
                             div_up( uint32_t( dims.y ), block_shape.y ), 1u };

        stats::frameSamplers.motionVecTime.start();
        
        const MaterialCuda* d_materials = scene.getMaterialCache().getDeviceData().data();

        if ( m_displacementMode == MOTION_VEC_DISPLACEMENT_FROM_MATERIAL )
        {
            motionVecKernel< Vertex, MOTION_VEC_DISPLACEMENT_FROM_MATERIAL > <<<grid_shape, block_shape>>>( m_deviceSubds.span(), m_deviceInstances.span(),
                    DeviceCamera( camera, dims ), DeviceCamera( prevCamera, dims ),
                    jitter, hits.span(), gbuffer.m_depth, d_materials, displacementScale, displacementBias, gbuffer.m_motionvecs );
        }
        else {
            motionVecKernel< Vertex, MOTION_VEC_DISPLACEMENT_FROM_SUBD_EVAL > <<<grid_shape, block_shape>>>( m_deviceSubds.span(), m_deviceInstances.span(),
                    DeviceCamera( camera, dims ), DeviceCamera( prevCamera, dims ),
                    jitter, hits.span(), gbuffer.m_depth, d_materials, displacementScale, displacementBias, gbuffer.m_motionvecs );
        }
        stats::frameSamplers.motionVecTime.stop();

    }
}

