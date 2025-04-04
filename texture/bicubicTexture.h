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

#include "textureCuda.h"

#ifdef __CUDACC__
/*
    Bicubic filtering
    See GPU Gems 2: "Fast Third-Order Texture Filtering", Sigg & Hadwiger
    http://http.developer.nvidia.com/GPUGems2/gpugems2_chapter20.html

    Reformulation thanks to Keenan Crane
*/
namespace BicubicSampling {


// w0, w1, w2, and w3 are the four cubic B-spline basis functions
inline __device__ float w0( float a )
{
    //    return (1.0f/6.0f)*(-a*a*a + 3.0f*a*a - 3.0f*a + 1.0f);
    return ( 1.0f / 6.0f ) * ( a * ( a * ( -a + 3.0f ) - 3.0f ) + 1.0f );  // optimized
}

inline __device__ float w1( float a )
{
    //    return (1.0f/6.0f)*(3.0f*a*a*a - 6.0f*a*a + 4.0f);
    return ( 1.0f / 6.0f ) * ( a * a * ( 3.0f * a - 6.0f ) + 4.0f );
}

inline __device__ float w2( float a )
{
    //    return (1.0f/6.0f)*(-3.0f*a*a*a + 3.0f*a*a + 3.0f*a + 1.0f);
    return ( 1.0f / 6.0f ) * ( a * ( a * ( -3.0f * a + 3.0f ) + 3.0f ) + 1.0f );
}

inline __device__ float w3( float a )
{
    return ( 1.0f / 6.0f ) * ( a * a * a );
}

// g0 and g1 are the two amplitude functions
inline __device__ float g0( float a )
{
    return w0( a ) + w1( a );
}

inline __device__ float g1( float a )
{
    return w2( a ) + w3( a );
}

// h0 and h1 are the two offset functions
inline __device__ float h0( float a )
{
    // note +0.5 offset to compensate for CUDA linear filtering convention
    return -1.0f + w1( a ) / ( w0( a ) + w1( a ) ) + 0.5f;
}

inline __device__ float h1( float a )
{
    return 1.0f + w3( a ) / ( w2( a ) + w3( a ) ) + 0.5f;
}


inline __device__ float mipLevel(float2 dx, float2 dy) 
{
    float d = fmaxf(dot(dx, dx), dot(dy, dy));
    return fmaxf(0.5f * log2f(d), 0.f);
}



template <class R>  // texture data type, return type
__device__ R tex2D_bicubic_lod_fast( cudaTextureObject_t tex, float2 dim, float x, float y, float level )
{
    const float s = powf(2.f, level);
    dim /= s;
    const float2 invdim = make_float2( 1.f / dim.x, 1.f / dim.y );

    x *= dim.x;
    y *= dim.y;

    x -= 0.5f;
    y -= 0.5f;

    float px = floorf( x );
    float py = floorf( y );
    float fx = x - px;
    float fy = y - py;

    // note: we could store these functions in a lookup table texture, but maths is cheap
    float g0x = g0( fx );
    float g1x = g1( fx );
    float h0x = h0( fx );
    float h1x = h1( fx );
    float h0y = h0( fy );
    float h1y = h1( fy );

    const float px_h0x = invdim.x * (px + h0x);
    const float px_h1x = invdim.x * (px + h1x);

    const float py_h0y = invdim.y * (py + h0y);
    const float py_h1y = invdim.y * (py + h1y);

    const float2 uv0 = make_float2( px_h0x, py_h0y );
    const float2 uv1 = make_float2( px_h1x, py_h0y );
    const float2 uv2 = make_float2( px_h0x, py_h1y );
    const float2 uv3 = make_float2( px_h1x, py_h1y );

    R r = g0( fy ) * ( g0x * tex2DLod<R>(tex, uv0.x, uv0.y, level ) + g1x * tex2DLod<R>(tex, uv1.x, uv1.y, level ) ) +
          g1( fy ) * ( g0x * tex2DLod<R>(tex, uv2.x, uv2.y, level ) + g1x * tex2DLod<R>(tex, uv3.x, uv3.y, level ) );
    return r;
}


template <class R>  // texture data type, return type
__device__ R tex2D_bicubic_grad_fast( const TextureCuda& tex, float x, float y, float2 dx = {0.0f, 0.0f}, float2 dy = {0.0f, 0.0f}, float mipBias = 0.0f )
{
    dx *= tex.getDim().x;
    dy *= tex.getDim().y;
    const float level = mipLevel( dx, dy ) + mipBias;
    return tex2D_bicubic_lod_fast<R>( tex.getTexref(), tex.getDim(), x, y, level );
}


};  // namespace BicubicSampling

#endif // #ifdef __CUDACC__
