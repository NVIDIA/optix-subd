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

#include "GBuffer.cuh"
#include "shadingTypes.h"
#include "utils.cuh"

#include <OptiXToolkit/ShaderUtil/color.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <optix.h>

#include <cfloat>

extern "C" {
    extern __constant__ Params params;
}



// Note: we have both primary and shadow ray types, but only primary rays use CH so the 
// stride between SBT entries is 1, not 2
constexpr unsigned int SBT_STRIDE = 1;

__device__ inline
void makeCameraRay( const uint2& idx, const uint2& dims, const float2& subpixel_jitter, float3& out_ray_origin, float3& out_ray_direction )
{
    float2 d       = ( ( make_float2( idx.x, idx.y ) + subpixel_jitter ) / make_float2( dims.x, dims.y ) ) * 2.f - 1.f;
    out_ray_origin = params.eye;
    out_ray_direction = normalize( d.x * params.U + d.y * params.V + params.W );
}

__device__ inline
void traceRadiance( OptixTraversableHandle handle,
                    float3                 rayOrigin,
                    float3                 rayDirection,
                    unsigned&              seed,
                    float3&                pathWeight )
{
    unsigned u0 = pathWeight.x;
    unsigned u1 = pathWeight.y;
    unsigned u2 = pathWeight.z;
    unsigned u3 = seed;

    optixTrace( params.handle, rayOrigin, rayDirection,
                0.0f,     // tmin
                FLT_MAX,  // tmax
                0.0f,     // time
                OptixVisibilityMask( 1 ), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                0,        // SBT offset
                SBT_STRIDE,
                RAY_TYPE_RADIANCE,
                u0, u1, u2, u3 );

    pathWeight = {  __uint_as_float(u0), __uint_as_float(u1), __uint_as_float(u2) };
    seed       = u3;
}


extern "C" __global__
void __raygen__pinhole()
{
    const uint2 idx = make_uint2( optixGetLaunchIndex() );
    const uint2 dims = make_uint2( optixGetLaunchDimensions() );

    unsigned int image_index = dims.x * idx.y + idx.x;
    unsigned int seed        = tea<16>( image_index, params.frame_index );
    float3 rayOrigin, rayDirection;

    float3 result = make_float3( 0.f );

    makeCameraRay( idx, dims, params.jitter, rayOrigin, rayDirection );

    float3 pathWeight = make_float3( 1.f );
    traceRadiance( params.handle, rayOrigin, rayDirection, seed, pathWeight );
    result += pathWeight;

    gbuffer::write( make_float4( result, 1.0f ), params.aovColor, idx );

    {
        // HACK: since DLSS is not hooked up yet, write high res depth buffer from here
        // rather than in a separate pass.  This is a shortcut.
        makeCameraRay(idx, dims, {0.5, 0.5}, rayOrigin, rayDirection);
        optixTraverse(params.handle, rayOrigin, rayDirection,
                      0.0f,    // tmin
                      FLT_MAX, // tmax
                      0.0f,    // time
                      OptixVisibilityMask(1), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                      0, // SBToffset
                      SBT_STRIDE,
                      RAY_TYPE_RADIANCE);

        float depth = std::numeric_limits<float>::infinity();
        if (optixHitObjectIsHit())
        {  
            float t = optixHitObjectGetRayTmax();
            depth = dot( normalize( params.W ), t*rayDirection );
        }
        gbuffer::write( depth, params.aovDepthHires, idx );
    }
}

extern "C" __global__ 
void __miss__radiance()
{
    optixSetPayload_0( __float_as_uint( params.missColor.x ) );
    optixSetPayload_1( __float_as_uint( params.missColor.y ) );
    optixSetPayload_2( __float_as_uint( params.missColor.z ) );

    // write AOVs
    {
        const uint2 idx = make_uint2( optixGetLaunchIndex() );
        gbuffer::write( std::numeric_limits<float>::infinity(), params.aovDepth, idx );

        const float4 albedo{ 0.f, 0.f, 0.f, 1.f };
        const float4 normal{ 0.f, 0.f, 0.f, 1.f };

        gbuffer::write( normal, params.aovNormals, idx );
        gbuffer::write( albedo, params.aovAlbedo, idx );

        const uint2  dims            = make_uint2( optixGetLaunchDimensions() );
        unsigned int linearIdx       = dims.x * idx.y + idx.x;
        params.hit_buffer[linearIdx] = HitResult{};
    }

}
