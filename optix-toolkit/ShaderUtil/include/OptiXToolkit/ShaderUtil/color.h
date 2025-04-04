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

#include <vector_types.h>
#include <OptiXToolkit/ShaderUtil/Matrix.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>

#ifndef USE_ACES
    #define USE_ACES 1
#endif

__forceinline__ __device__ float3 apply_exposure( float3 rgb, float exposure )
{
    return rgb * exposure;
}

__forceinline__ __device__ float3 uncharted2_partial( float3 x )
{
    float A = 0.15f; float B = 0.50f; float C = 0.10f;
    float D = 0.20f; float E = 0.02f; float F = 0.30f;
    return ( ( x * ( A * x + C * B ) + D * E ) / ( x * ( A * x + B ) + D * F ) ) - E / F;
}

__forceinline__ __device__ float3 Uncharted2_tonemap( float3 rgb )
{
    // reference : https://64.github.io/tonemapping/
    static constexpr float const exposure_bias = 2.0f;
    float3 curr = uncharted2_partial( rgb * exposure_bias );

    static constexpr float3 const W = { 11.2f, 11.2f, 11.2f };
    float3 white_scale = float3{ 1.f, 1.f, 1.f } / uncharted2_partial(W);
    
    float3 c = clamp( curr * white_scale, 0.f, 1.f );

    float e = 1.f / 2.2f;

    return { powf(c.x, e), powf(c.y, e), powf(c.z, e) };
}

__forceinline__ __device__ float3 ACES_tonemap( float3 rgb )
{
    // reference : https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
    // sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
    otk::Matrix3x3 m1;
    m1.setCol( 0, { 0.59719f, 0.07600f, 0.02840f } );
    m1.setCol( 1, { 0.35458f, 0.90834f, 0.13383f } );
    m1.setCol( 2, { 0.04823f, 0.01566f, 0.83777f } );

    // ODT_SAT => XYZ => D60_2_D65 => sRGB
    otk::Matrix3x3 m2;
    m2.setCol( 0, {  1.60475f, -0.10208f, -0.00327f } );
    m2.setCol( 1, { -0.53108f,  1.10813f, -0.07276f } );
    m2.setCol( 2, { -0.07367f, -0.00605f,  1.07602f } );

    float3 v = m1 * float3{ rgb.x, rgb.y, rgb.z };
    float3 a = v * ( v + 0.0245786f ) - 0.000090537f;
    float3 b = v * ( 0.983729f * v + 0.4329510f ) + 0.238081f;

    float3 c = clamp( m2 * ( a / b ), 0.0f, 1.0f );

    float e = 1.f / 2.2f;

    return { powf( c.x, e ), powf( c.y, e ), powf( c.z, e ) };
}

__forceinline__ __device__ float3 toSRGB( const float3& c )
{
    // reference: https://www.color.org/chardata/rgb/srgb.xalter
    float  invGamma = 1.0f / 2.4f;
    float3 powed    = make_float3( powf( c.x, invGamma ), powf( c.y, invGamma ), powf( c.z, invGamma ) );
    return make_float3(
        c.x < 0.0031308f ? 12.92f * c.x : 1.055f * powed.x - 0.055f,
        c.y < 0.0031308f ? 12.92f * c.y : 1.055f * powed.y - 0.055f,
        c.z < 0.0031308f ? 12.92f * c.z : 1.055f * powed.z - 0.055f );
}

__forceinline__ __device__ float3 fromSRGB( const float3& c )
{
    // reference: https://www.color.org/chardata/rgb/srgb.xalter
    return make_float3(
        c.x <= 0.04045f ? c.x / 12.92f : powf( ( c.x + 0.055f ) / 1.055f, 2.4f ),
        c.y <= 0.04045f ? c.y / 12.92f : powf( ( c.y + 0.055f ) / 1.055f, 2.4f ),
        c.z <= 0.04045f ? c.z / 12.92f : powf( ( c.z + 0.055f ) / 1.055f, 2.4f )
    );
}

//__forceinline__ __device__ float dequantizeUnsigned8Bits( const unsigned char i )
//{
//    enum { N = (1 << 8) - 1 };
//    return min((float)i / (float)N), 1.f)
//}
__forceinline__ __device__ unsigned char quantizeUnsigned8Bits( float x )
{
    x = clamp( x, 0.0f, 1.0f );
    enum { N = (1 << 8) - 1, Np1 = (1 << 8) };
    return (unsigned char)min((unsigned int)(x * (float)Np1), (unsigned int)N);
}

__forceinline__ __device__ uchar4 make_color( const float3& c )
{
    float3 srgb = toSRGB( clamp( c, 0.0f, 1.0f ) );
    return make_uchar4( quantizeUnsigned8Bits( srgb.x ), quantizeUnsigned8Bits( srgb.y ), quantizeUnsigned8Bits( srgb.z ), 255u );
}

__forceinline__ __device__ uchar4 make_color( const float4& c )
{
    return make_color( make_float3( c.x, c.y, c.z ) );
}
